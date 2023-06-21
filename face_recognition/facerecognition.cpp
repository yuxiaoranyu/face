#include <fstream>
#include <dirent.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include "facerecognition.h"

using namespace nvinfer1;

FaceRecognition::FaceRecognition() : bInitOk(false)
{
}

FaceRecognition::~FaceRecognition()
{
    if (bInitOk)
    {
        cudaStreamDestroy(m_cudaStream);
        CHECK(cudaFree(m_buffers[0]));
        CHECK(cudaFree(m_buffers[1]));
        m_IExecutionContext->destroy();
        m_ICudaEngine->destroy();
        m_IRuntime->destroy();
    }
}

void FaceRecognition::load(char *model_path)
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
        std::cout << "face recognition createInferRuntime error" << std::endl;
        return;
    }
    m_ICudaEngine = m_IRuntime->deserializeCudaEngine(trtModelStream, fileSize);
    if (m_ICudaEngine == nullptr)
    {
        std::cout << "face recognition deserializeCudaEngine error" << std::endl;
        return;
    }
    delete[] trtModelStream;
    m_IExecutionContext = m_ICudaEngine->createExecutionContext();
    if (m_IExecutionContext == nullptr)
    {
        std::cout << "face recognition createExecutionContext error" << std::endl;
        return;
    }

    cudaStreamCreate(&m_cudaStream);
    cudaMalloc((void **)&m_buffers[0], 1 * 3 * INPUT_H * INPUT_W * sizeof(float));
    cudaMalloc((void **)&m_buffers[1], 1 * OUTPUT_SIZE * sizeof(float));
    std::cout << "face recognition load ok" << std::endl;
    bInitOk = true;
}

void FaceRecognition::doInference(float *inputBuf, float *outputBuf, int batchSize)
{
    CHECK(cudaMemcpyAsync(m_buffers[0], inputBuf, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, m_cudaStream));
    m_IExecutionContext->enqueue(batchSize, (void **)m_buffers, m_cudaStream, nullptr);
    CHECK(cudaMemcpyAsync(outputBuf, m_buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, m_cudaStream));
    cudaStreamSynchronize(m_cudaStream);
}

void FaceRecognition::get_filenames_in_dir(const char *p_dir_name, std::vector<std::string> &file_names)
{
    DIR *p_dir = opendir(p_dir_name);

    if (p_dir == nullptr)
    {
        return;
    }

    struct dirent *p_file;
    while ((p_file = readdir(p_dir)) != nullptr)
    {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0)
        {
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
}

Rec_result FaceRecognition::infer(unsigned char *input_img, char *facedb_path)
{
    std::string img_dir = facedb_path;
    std::string label = "";
    std::vector<std::string> file_names;
    Rec_result rec_result;
    if (!bInitOk)
    {
        rec_result.rec_score=0;
        rec_result.person_id=(char *)label.c_str();
        return rec_result;
    }

    get_filenames_in_dir(facedb_path, file_names);
    std::map<std::string, cv::Mat> facedbMap;
    for (auto it : file_names)
    {
        std::string person_name = it;
        std::string img_name = img_dir + "/" + it;
        cv::Mat img = cv::imread(img_name);
        for (int i = 0; i < INPUT_H * INPUT_W; i++)
        {
            input_data[i] = ((float)img.at<cv::Vec3b>(i)[2] - 127.5) * 0.0078125;
            input_data[i + INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[1] - 127.5) * 0.0078125;
            input_data[i + 2 * INPUT_H * INPUT_W] = ((float)img.at<cv::Vec3b>(i)[0] - 127.5) * 0.0078125;
        }

//        auto start = std::chrono::system_clock::now();
        doInference(input_data, output_data, BATCH_SIZE);
//        auto end = std::chrono::system_clock::now();
//        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        cv::Mat outMat(512, 1, CV_32FC1, output_data);
        cv::Mat out_norm;
        cv::normalize(outMat, out_norm);
        facedbMap[person_name] = out_norm;
    }

    cv::Mat detect_img(112, 112, CV_8UC3, input_img);
    for (int i = 0; i < INPUT_H * INPUT_W; i++)
    {
        input_data[i] = ((float)detect_img.at<cv::Vec3b>(i)[2] - 127.5) * 0.0078125;
        input_data[i + INPUT_H * INPUT_W] = ((float)detect_img.at<cv::Vec3b>(i)[1] - 127.5) * 0.0078125;
        input_data[i + 2 * INPUT_H * INPUT_W] = ((float)detect_img.at<cv::Vec3b>(i)[0] - 127.5) * 0.0078125;
    }

    doInference(input_data, output_data, BATCH_SIZE);

    cv::Mat outMat2(1, 512, CV_32FC1, output_data);
    cv::Mat out_norm2;
    cv::normalize(outMat2, out_norm2);

    std::map<std::string, float> face_result;
    for (auto it : facedbMap)
    {
        cv::Mat res = out_norm2 * it.second;
        face_result[it.first] = *(float *)res.data;
        std::cout << "==>" << it.first << ": " << *(float *)res.data << std::endl;
    }

    auto result = max_element(face_result.begin(), face_result.end(),
                              [](std::pair<std::string, float> l, std::pair<std::string, float> r)
                              { return l.second < r.second; });
    
    label = result->first;
    if (result->second < 0.45)
    {
        label = "";
    }
    rec_result.rec_score=result->second;
    rec_result.person_id=(char *)label.c_str();
    return rec_result;
}

extern "C"
{

    FaceRecognition faceRecognition;

    void load(char *model_path)
    {
        faceRecognition.load(model_path);
    }

    Rec_result infer(unsigned char *input_img, char *facedb_path)
    {
        return faceRecognition.infer(input_img, facedb_path);
    }
}
