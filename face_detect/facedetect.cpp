#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"
#include "preprocess.h"
#include "postprocess.h"
#include "logging.h"
#include "facedetect.h"

using namespace nvinfer1;

FaceDetect::FaceDetect() :
    bInitOk(false)
{
}

FaceDetect::~FaceDetect()
{
    if (bInitOk)
    {
        m_IExecutionContext->destroy();
        m_ICudaEngine->destroy();
        m_IRuntime->destroy();
        cudaStreamDestroy(m_cudaStream);
        cudaFree(m_buffers[0]);
        cudaFree(m_buffers[1]);
        cudaFree(affine_matrix_d2i_device);
        cudaFreeHost(affine_matrix_d2i_host);
        cudaFree(img_device);
        cudaFreeHost(img_host);
        cudaFree(decode_ptr_device);
        delete[] decode_ptr_host;
    }
}

void FaceDetect::getd2i(affineMatrix &afmt, cv::Size to, cv::Size from)
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

void FaceDetect::load(char* model_path)
{
    static Logger gLogger;
    char *trtModelStream = nullptr;
    std::ifstream file(model_path, std::ios::binary);
    int fileSize;

    cudaSetDevice(0);
    if (file.good())
    {
        file.seekg(0, file.end);
        fileSize = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[fileSize];
        file.read(trtModelStream, fileSize);
        file.close();
    }
    else
    {
        std::cerr << "can not open model file " << std::endl;
        return;
    }

    m_IRuntime = createInferRuntime(gLogger);
    if (m_IRuntime == nullptr)
    {
        std::cout << "face detect createInferRuntime error" << std::endl;
        return;
    }
    m_ICudaEngine = m_IRuntime->deserializeCudaEngine(trtModelStream, fileSize);
    if (m_ICudaEngine == nullptr)
    {
        std::cout << "face detect deserializeCudaEngine error" << std::endl;
        return;
    }
    delete[] trtModelStream;
    m_IExecutionContext = m_ICudaEngine->createExecutionContext();
    if (m_IExecutionContext == nullptr)
    {
        std::cout << "face detect createExecutionContext error" << std::endl;
        return;
    }
    Dims out_dims = m_ICudaEngine->getBindingDimensions(1); //获取输出维度
    m_num_bboxes = out_dims.d[1];
    m_output_size = 1;
    for (int i = 0; i < out_dims.nbDims; i++)
    {
        m_output_size *= out_dims.d[i];
    }

    cudaStreamCreate(&m_cudaStream);
    cudaMalloc(&m_buffers[0], 1 * 3 * INPUT_H * INPUT_W * sizeof(float));
    cudaMalloc(&m_buffers[1], 1 * m_output_size * sizeof(float));

    cudaMallocHost((void **)&img_host, MAX_IMAGE_INPUT_SIZE_THRESH * 3);
    cudaMalloc((void **)&img_device, MAX_IMAGE_INPUT_SIZE_THRESH * 3);

    cudaMallocHost(&affine_matrix_d2i_host, sizeof(float) * 6);
    cudaMalloc(&affine_matrix_d2i_device, sizeof(float) * 6);
    decode_ptr_host = new float[1 + MAX_OBJECTS * NUM_BOX_ELEMENT];
    cudaMalloc(&decode_ptr_device, sizeof(float) * (1 + MAX_OBJECTS * NUM_BOX_ELEMENT));

    std::cout << "face detect load ok" << std::endl;
    bInitOk = true;
}

bbox FaceDetect::infer(unsigned char *input_img, int width, int height)
{
    affineMatrix afmt;

    if (!bInitOk)
    {
        bbox box;

        return box;
    }

    cv::Mat img(width, height, CV_8UC3, input_img);
    cv::Size to(INPUT_W, INPUT_H);

    getd2i(afmt, to, cv::Size(img.cols, img.rows));
    memcpy(affine_matrix_d2i_host, afmt.d2i, sizeof(afmt.d2i));

    size_t size_image = img.cols * img.rows * 3;
    size_t size_image_dst = INPUT_H * INPUT_W * 3;
    memcpy(img_host, img.data, size_image);

    CHECK(cudaMemcpyAsync(img_device, img_host, size_image, cudaMemcpyHostToDevice, m_cudaStream));
    CHECK(cudaMemcpyAsync(affine_matrix_d2i_device, affine_matrix_d2i_host, sizeof(afmt.d2i), cudaMemcpyHostToDevice, m_cudaStream));
    preprocess_kernel_img(img_device, img.cols, img.rows, m_buffers[0], INPUT_W, INPUT_H, affine_matrix_d2i_device, m_cudaStream); // 前处理 ，相当于letter_box
    m_IExecutionContext->enqueueV2((void **)m_buffers, m_cudaStream, nullptr);
    float *predict = (float *)m_buffers[1];
    CHECK(cudaMemsetAsync(decode_ptr_device, 0, sizeof(int), m_cudaStream));
    decode_kernel_invoker(predict, NUM_BOX_ELEMENT, m_num_bboxes, NUM_CLASSES, CKPT_NUM, BBOX_CONF_THRESH, affine_matrix_d2i_device, decode_ptr_device, MAX_OBJECTS, m_cudaStream); // cuda 后处理
    nms_kernel_invoker(decode_ptr_device, NMS_THRESH, MAX_OBJECTS, m_cudaStream, NUM_BOX_ELEMENT);                                                                                       // cuda nms
    CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device, sizeof(float) * (1 + MAX_OBJECTS * NUM_BOX_ELEMENT), cudaMemcpyDeviceToHost, m_cudaStream));
    cudaStreamSynchronize(m_cudaStream);

    std::vector<bbox> boxes;
    int count = std::min((int)*decode_ptr_host, MAX_OBJECTS);
    for (int i = 0; i < count; i++)
    {
        int basic_pos = 1 + i * NUM_BOX_ELEMENT;
        int keep_flag = decode_ptr_host[basic_pos + 6];
        if (keep_flag == 1)
        {
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
    if (boxes.size() > 0)
    {
        return boxes[0];
    }
    else
    {
        bbox box;

        return box;
    }
}

extern "C"
{

FaceDetect faceDetect;

void load(char* model_path)
{
    faceDetect.load(model_path);
}

bbox infer(unsigned char *input_img, int width, int height)
{
    return faceDetect.infer(input_img, width, height);
}

}
