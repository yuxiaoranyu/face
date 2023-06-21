#pragma once

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

struct Rec_result
{
    float rec_score;
    char* person_id;
};

class FaceRecognition
{
    public:
        FaceRecognition();
        ~FaceRecognition();
        void load(char *model_path);
        Rec_result infer(unsigned char *input_img, char *facedb_path);

    private:
        void doInference(float *inputBuf, float *outputBuf, int batchSize);
        void get_filenames_in_dir(const char *p_dir_name, std::vector<std::string> &file_names);

    private:
        static const int INPUT_H = 112;
        static const int INPUT_W = 112;
        static const int OUTPUT_SIZE = 512;
        static const int BATCH_SIZE = 1;

        bool bInitOk;
        float input_data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
        float output_data[BATCH_SIZE * OUTPUT_SIZE];

        nvinfer1::IRuntime *m_IRuntime;
        nvinfer1::ICudaEngine *m_ICudaEngine;
        nvinfer1::IExecutionContext *m_IExecutionContext;
        cudaStream_t m_cudaStream;
        float *m_buffers[2];
};

