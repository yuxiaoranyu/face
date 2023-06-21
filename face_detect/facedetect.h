#pragma once

#define MAX_IMAGE_INPUT_SIZE_THRESH 25000000  //推理图片最大尺寸
#define MAX_OBJECTS 2048  //图片最多检测目标数

#define NMS_THRESH 0.45
#define BBOX_CONF_THRESH 0.3

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

struct affineMatrix // 变换矩阵
{
    float i2d[6]; // 正变换
    float d2i[6]; // 逆变换
};

struct bbox
{
    bbox() : x1(0), x2(0), y1(0), y2(0) {}
    float x1, x2, y1, y2; //人脸位置矩形
    float landmarks[10]; // 5个关键点
    float score; //得分
};

class FaceDetect
{
    public:
        FaceDetect();
        ~FaceDetect();
        void load(char* model_path);
        bbox infer(unsigned char *input_img, int width, int height);

    private:
        void getd2i(affineMatrix &afmt, cv::Size to, cv::Size from); //构建变换矩阵

    private:
        static const int INPUT_W = 640;
        static const int INPUT_H = 640;
        static const int NUM_CLASSES = 1; // 类别数
        static const int CKPT_NUM = 5;    // 关键点个数
        static const int NUM_BOX_ELEMENT = 7 + CKPT_NUM * 2;

        bool bInitOk;
        int m_num_bboxes;
        int m_output_size;
        nvinfer1::IRuntime *m_IRuntime;
        nvinfer1::ICudaEngine *m_ICudaEngine;
        nvinfer1::IExecutionContext *m_IExecutionContext;
        cudaStream_t m_cudaStream;
        float *m_buffers[2];
        uint8_t *img_host;
        uint8_t *img_device;
        float *affine_matrix_d2i_host;
        float *affine_matrix_d2i_device;
        float *decode_ptr_device;
        float *decode_ptr_host;
};

