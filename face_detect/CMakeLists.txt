cmake_minimum_required(VERSION 3.10)
project(facedetect)

add_definitions(-std=c++11)
add_definitions(-w)
# option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
find_package(CUDA REQUIRED)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

#cuda 
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

#tensorrt 
include_directories(/home/ai/build_group/TensorRT/TensorRT-7.1.3.4/include)
link_directories(/home/ai/build_group/TensorRT/TensorRT-7.1.3.4/lib)

#opencv 
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

#onnx2trt
#add_subdirectory(${PROJECT_SOURCE_DIR}/onnx2trt)

cuda_add_library(facedetect SHARED facedetect.cpp preprocess.cu postprocess.cu)
target_link_libraries(facedetect nvinfer)
target_link_libraries(facedetect cudart)
target_link_libraries(facedetect nvonnxparser)
target_link_libraries(facedetect ${OpenCV_LIBS})

add_definitions(-O0 -pthread)
