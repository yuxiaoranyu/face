#include <fstream>
#include <iostream>
#include <sstream>
#include <numeric>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "include/utils.hpp"
#include "preprocess.h"
#include "postprocess.h"
#define MAX_IMAGE_INPUT_SIZE_THRESH 5000 * 5000
#define MAX_OBJECTS 2048

struct affineMatrix // letter_box  仿射变换矩阵
{
    float i2d[6]; // 仿射变换正变换
    float d2i[6]; // 仿射变换逆变换
};

struct bbox
{
    float x1, x2, y1, y2;
    float landmarks[10]; // 5个关键点
    float score;
};

#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

#define DEVICE 0 // GPU id
#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.3

using namespace nvinfer1;

static const int INPUT_W = 640;
static const int INPUT_H = 640;
static const int NUM_CLASSES = 1; // 类别数
static const int CKPT_NUM = 5;    // 关键点个数

const char *INPUT_BLOB_NAME = "input";   // onnx 输入  名字
const char *OUTPUT_BLOB_NAME = "output"; // onnx 输出 名字
static Logger gLogger;

extern "C"
{

    void affine_project(float *d2i, float x, float y, float *ox, float *oy) // 通过仿射变换逆矩阵，恢复成原图的坐标
    {
        *ox = d2i[0] * x + d2i[1] * y + d2i[2];
        *oy = d2i[3] * x + d2i[4] * y + d2i[5];
    }

    const float color_list[5][3] =
        {
            {255, 0, 0},
            {0, 255, 0},
            {0, 0, 255},
            {0, 255, 255},
            {255, 255, 0},
    };

    void getd2i(affineMatrix &afmt, cv::Size to, cv::Size from) // 计算仿射变换的矩阵和逆矩阵
    {
        float scale = std::min(1.0 * to.width / from.width, 1.0 * to.height / from.height);
        afmt.i2d[0] = scale;
        afmt.i2d[1] = 0;
        afmt.i2d[2] = -scale * from.width * 0.5 + to.width * 0.5;
        afmt.i2d[3] = 0;
        afmt.i2d[4] = scale;
        afmt.i2d[5] = -scale * from.height * 0.5 + to.height * 0.5;
        cv::Mat i2d_mat(2, 3, CV_32F, afmt.i2d);
        cv::Mat d2i_mat(2, 3, CV_32F, afmt.d2i);
        cv::invertAffineTransform(i2d_mat, d2i_mat);
        memcpy(afmt.d2i, d2i_mat.ptr<float>(0), sizeof(afmt.d2i));
    }

    bbox infer(unsigned char *input_img,int width,int height)
    {
        int NUM_BOX_ELEMENT = 7 + CKPT_NUM * 2;
        cudaSetDevice(DEVICE);
        char *trtModelStreamDet{nullptr};
        size_t size{0};
        const std::string engine_file_path = "/home/ambition/face_detect/yolov7-face-tensorrt-master/build/yolov5s-face.trt";
        std::ifstream file(engine_file_path, std::ios::binary);
        int batch_size = 1;
        if (file.good())
        {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStreamDet = new char[size];
            assert(trtModelStreamDet);
            file.read(trtModelStreamDet, size);
            file.close();
        }

        // det模型trt初始化
        IRuntime *runtime_det = createInferRuntime(gLogger);
        assert(runtime_det != nullptr);

        ICudaEngine *engine_det = runtime_det->deserializeCudaEngine(trtModelStreamDet, size);
        assert(engine_det != nullptr);
        IExecutionContext *context_det = engine_det->createExecutionContext();
        assert(context_det != nullptr);
        delete[] trtModelStreamDet;

        float *buffers[2];
        const int inputIndex = engine_det->getBindingIndex(INPUT_BLOB_NAME);
        const int outputIndex = engine_det->getBindingIndex(OUTPUT_BLOB_NAME);
        assert(inputIndex == 0);
        assert(outputIndex == 1);
        // Create GPU buffers on device

        auto out_dims = engine_det->getBindingDimensions(1);
        auto output_size = 1;
        int OUTPUT_CANDIDATES = out_dims.d[1];
        std::cout << "OUTPUT_CANDIDATES: " << out_dims.d[0] << std::endl;

        for (int j = 0; j < out_dims.nbDims; j++)
        {
            output_size *= out_dims.d[j];
        }

        CHECK(cudaMalloc((void **)&buffers[inputIndex], 3 * INPUT_H * INPUT_W * sizeof(float)));
        CHECK(cudaMalloc((void **)&buffers[outputIndex], output_size * sizeof(float)));

        // Create stream
        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));
        uint8_t *img_host = nullptr;
        uint8_t *img_device = nullptr;
        float *affine_matrix_d2i_host = nullptr;
        float *affine_matrix_d2i_device = nullptr;
        float *decode_ptr_device = nullptr;
        float *decode_ptr_host = nullptr;
        decode_ptr_host = new float[1 + MAX_OBJECTS * NUM_BOX_ELEMENT];
        // prepare input data cache in pinned memory
        CHECK(cudaMallocHost((void **)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
        // prepare input data cache in device memory
        CHECK(cudaMalloc((void **)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3));
        CHECK(cudaMallocHost(&affine_matrix_d2i_host, sizeof(float) * 6));
        CHECK(cudaMalloc(&affine_matrix_d2i_device, sizeof(float) * 6));
        CHECK(cudaMalloc(&decode_ptr_device, sizeof(float) * (1 + MAX_OBJECTS * NUM_BOX_ELEMENT)));

        static float *prob = new float[output_size];

        double sumTime = 0;
        int index = 0;
        cv::Size to(INPUT_W, INPUT_H);
        affineMatrix afmt;

        cv::Mat img(width, height, CV_8UC3, input_img);

        getd2i(afmt, to, cv::Size(img.cols, img.rows));
        double begin_time = cv::getTickCount();
        float *buffer_idx = (float *)buffers[inputIndex];
        size_t size_image = img.cols * img.rows * 3;
        size_t size_image_dst = INPUT_H * INPUT_W * 3;
        memcpy(affine_matrix_d2i_host, afmt.d2i, sizeof(afmt.d2i));
        memcpy(img_host, img.data, size_image);
        std::cout << "max5 buff" << std::endl;
        CHECK(cudaMemcpyAsync(img_device, img_host, size_image, cudaMemcpyHostToDevice, stream));
        CHECK(cudaMemcpyAsync(affine_matrix_d2i_device, affine_matrix_d2i_host, sizeof(afmt.d2i), cudaMemcpyHostToDevice, stream));
        preprocess_kernel_img(img_device, img.cols, img.rows, buffer_idx, INPUT_W, INPUT_H, affine_matrix_d2i_device, stream); // 前处理 ，相当于letter_box
        double time_pre = cv::getTickCount();
        double time_pre_ = (time_pre - begin_time) / cv::getTickFrequency() * 1000;
        (*context_det).enqueueV2((void **)buffers, stream, nullptr);
        float *predict = (float *)buffers[outputIndex];
        std::cout << "max6 buff" << std::endl;
        CHECK(cudaMemsetAsync(decode_ptr_device, 0, sizeof(int), stream));
        decode_kernel_invoker(predict, NUM_BOX_ELEMENT, OUTPUT_CANDIDATES, NUM_CLASSES, CKPT_NUM, BBOX_CONF_THRESH, affine_matrix_d2i_device, decode_ptr_device, MAX_OBJECTS, stream); // cuda 后处理
        nms_kernel_invoker(decode_ptr_device, NMS_THRESH, MAX_OBJECTS, stream, NUM_BOX_ELEMENT);                                                                                       // cuda nms
        CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device, sizeof(float) * (1 + MAX_OBJECTS * NUM_BOX_ELEMENT), cudaMemcpyDeviceToHost, stream));
        cudaStreamSynchronize(stream);
        double end_time = cv::getTickCount();
        std::vector<bbox> boxes;
        std::cout << "max7 buff" << std::endl;
        int boxes_count = 0;
        int count = std::min((int)*decode_ptr_host, MAX_OBJECTS);

        for (int i = 0; i < count; i++)
        {
            int basic_pos = 1 + i * NUM_BOX_ELEMENT;
            int keep_flag = decode_ptr_host[basic_pos + 6];
            if (keep_flag == 1)
            {
                boxes_count += 1;
                bbox box;
                box.x1 = decode_ptr_host[basic_pos + 0];
                box.y1 = decode_ptr_host[basic_pos + 1];
                box.x2 = decode_ptr_host[basic_pos + 2];
                box.y2 = decode_ptr_host[basic_pos + 3];
                box.score = decode_ptr_host[basic_pos + 4];
                int landmark_pos = basic_pos + 7;
                for (int id = 0; id < CKPT_NUM; id += 1)
                {
                    box.landmarks[2 * id] = decode_ptr_host[landmark_pos + 2 * id];
                    box.landmarks[2 * id + 1] = decode_ptr_host[landmark_pos + 2 * id + 1];
                }
                boxes.push_back(box);
            }
        }
        std::cout<<"landmarks: "<<boxes[0].landmarks[0];
        std::cout<<" "<<boxes[0].landmarks[1];
        std::cout<<" "<<boxes[0].landmarks[2];
        std::cout<<" "<<boxes[0].landmarks[3];
        std::cout<<" "<<boxes[0].landmarks[4];
        std::cout<<" "<<boxes[0].landmarks[5];
        std::cout<<" "<<boxes[0].landmarks[6];
        std::cout<<" "<<boxes[0].landmarks[7];
        std::cout<<" "<<boxes[0].landmarks[8];
        std::cout<<" "<<boxes[0].landmarks[9]<<std::endl;
        std::cout<<"boxes size: "<<boxes.size()<<std::endl;
        context_det->destroy();
        engine_det->destroy();
        runtime_det->destroy();

        cudaStreamDestroy(stream);
        CHECK(cudaFree(affine_matrix_d2i_device));
        CHECK(cudaFreeHost(affine_matrix_d2i_host));
        CHECK(cudaFree(img_device));
        CHECK(cudaFreeHost(img_host));
        CHECK(cudaFree(buffers[inputIndex]));
        CHECK(cudaFree(buffers[outputIndex]));
        CHECK(cudaFree(decode_ptr_device));
        delete[] decode_ptr_host;
        return boxes[0];
    }
}