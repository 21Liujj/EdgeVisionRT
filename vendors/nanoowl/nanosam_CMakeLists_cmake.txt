cmake_minimum_required(VERSION 3.18)
project(nanosam_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# --- CUDA ---
find_path(CUDA_INCLUDE_DIR cuda_runtime.h
    PATHS /usr/local/cuda/targets/aarch64-linux/include
          /usr/local/cuda/include
)
find_library(CUDA_RUNTIME_LIB cudart
    PATHS /usr/local/cuda/targets/aarch64-linux/lib
          /usr/local/cuda/lib64
)
if(NOT CUDA_INCLUDE_DIR OR NOT CUDA_RUNTIME_LIB)
    message(FATAL_ERROR "CUDA runtime not found")
endif()

# --- TensorRT ---
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/aarch64-linux-gnu /usr/include
)
find_library(TENSORRT_NVINFER_LIB nvinfer
    PATHS /usr/lib/aarch64-linux-gnu /usr/lib
)
find_library(TENSORRT_NVINFER_PLUGIN_LIB nvinfer_plugin
    PATHS /usr/lib/aarch64-linux-gnu /usr/lib
)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIB)
    message(FATAL_ERROR "TensorRT not found")
endif()

# --- OpenCV ---
find_package(OpenCV REQUIRED core imgcodecs imgproc)

add_executable(nanosam_cpp
    src/main.cpp
    src/trt_engine.cpp
    src/nanosam_runtime.cpp
    src/preprocessing.cpp
)

target_include_directories(nanosam_cpp PRIVATE
    include
    ${CUDA_INCLUDE_DIR}
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(nanosam_cpp PRIVATE
    ${TENSORRT_NVINFER_LIB}
    ${TENSORRT_NVINFER_PLUGIN_LIB}
    ${CUDA_RUNTIME_LIB}
    ${OpenCV_LIBS}
)

target_compile_options(nanosam_cpp PRIVATE -O2 -Wall -Wextra)

# Preprocessing test tool (no TensorRT dependency)
add_executable(preprocessing_test
    src/preprocessing_test.cpp
    src/preprocessing.cpp
)
target_include_directories(preprocessing_test PRIVATE
    include
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(preprocessing_test PRIVATE ${OpenCV_LIBS})

# Encoder-only test tool for tensor-mode parity verification
add_executable(encoder_test
    src/encoder_test.cpp
    src/trt_engine.cpp
)
target_include_directories(encoder_test PRIVATE
    include
    ${CUDA_INCLUDE_DIR}
    ${TENSORRT_INCLUDE_DIR}
)
target_link_libraries(encoder_test PRIVATE
    ${TENSORRT_NVINFER_LIB}
    ${CUDA_RUNTIME_LIB}
)

message(STATUS "CUDA include: ${CUDA_INCLUDE_DIR}")
message(STATUS "TensorRT include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
