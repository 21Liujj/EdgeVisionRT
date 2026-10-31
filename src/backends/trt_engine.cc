#include "edgevision/trt_engine.h"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

namespace edgevision {

namespace {

class TrtLogger : public nvinfer1::ILogger {
 public:
  static TrtLogger& instance() {
    static TrtLogger l;
    return l;
  }
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      std::fprintf(stderr, "[TRT] %s\n", msg);
    }
  }
};

std::string trt_shape_str(const nvinfer1::Dims& d) {
  std::ostringstream oss;
  for (int i = 0; i < d.nbDims; ++i) {
    if (i) oss << 'x';
    oss << d.d[i];
  }
  return oss.str();
}

}  // namespace

TrtEngine::~TrtEngine() { cleanup(); }

bool TrtEngine::load(const std::string& engine_path) {
  cleanup();

  std::ifstream f(engine_path, std::ios::binary | std::ios::ate);
  if (!f) {
    last_error_ = "TrtEngine: cannot open engine file: " + engine_path;
    return false;
  }
  engine_size_ = static_cast<std::size_t>(f.tellg());
  f.seekg(0, std::ios::beg);
  engine_data_ = new uint8_t[engine_size_];
  f.read(reinterpret_cast<char*>(engine_data_), static_cast<std::streamsize>(engine_size_));
  f.close();

  runtime_ = nvinfer1::createInferRuntime(TrtLogger::instance());
  if (!runtime_) {
    last_error_ = "TrtEngine: createInferRuntime failed";
    cleanup();
    return false;
  }

  auto* rt = static_cast<nvinfer1::IRuntime*>(runtime_);
  engine_ = rt->deserializeCudaEngine(engine_data_, engine_size_);
  if (!engine_) {
    last_error_ = "TrtEngine: deserializeCudaEngine failed (bad engine file?)";
    cleanup();
    return false;
  }

  auto* eng = static_cast<nvinfer1::ICudaEngine*>(engine_);
  int nb = eng->getNbIOTensors();
  if (nb != 2) {
    last_error_ = "TrtEngine: expected 2 I/O tensors, got " + std::to_string(nb);
    cleanup();
    return false;
  }

  for (int i = 0; i < nb; ++i) {
    const char* name = eng->getIOTensorName(i);
    auto mode = eng->getTensorIOMode(name);
    auto shape = eng->getTensorShape(name);
    auto dtype = eng->getTensorDataType(name);

    if (mode == nvinfer1::TensorIOMode::kINPUT) {
      input_name_ = name;
      int nb_d = shape.nbDims;
      if (nb_d == 4) {
        input_dims_.n = shape.d[0];
        input_dims_.c = shape.d[1];
        input_dims_.h = shape.d[2];
        input_dims_.w = shape.d[3];
      }
      std::fprintf(stderr, "[TrtEngine] input  %s [%s] dtype=%d\n",
                   name, trt_shape_str(shape).c_str(), static_cast<int>(dtype));
    } else if (mode == nvinfer1::TensorIOMode::kOUTPUT) {
      output_name_ = name;
      int nb_d = shape.nbDims;
      if (nb_d >= 2) {
        output_dims_.n = 1;
        output_dims_.c = shape.d[shape.nbDims - 2];
        output_dims_.h = 1;
        output_dims_.w = shape.d[shape.nbDims - 1];
      }
      std::fprintf(stderr, "[TrtEngine] output %s [%s] dtype=%d\n",
                   name, trt_shape_str(shape).c_str(), static_cast<int>(dtype));
    }
  }

  context_ = eng->createExecutionContext();
  if (!context_) {
    last_error_ = "TrtEngine: createExecutionContext failed";
    cleanup();
    return false;
  }

  return true;
}

bool TrtEngine::prepare_io() {
  if (!context_) {
    last_error_ = "TrtEngine: no context (call load first)";
    return false;
  }

  auto* ctx = static_cast<nvinfer1::IExecutionContext*>(context_);

  cudaError_t cr;

  nvinfer1::Dims4 in_dims(static_cast<int32_t>(input_dims_.n),
                           static_cast<int32_t>(input_dims_.c),
                           static_cast<int32_t>(input_dims_.h),
                           static_cast<int32_t>(input_dims_.w));
  if (!ctx->setInputShape(input_name_.c_str(), in_dims)) {
    last_error_ = "TrtEngine: setInputShape failed";
    return false;
  }

  input_bytes_ = static_cast<std::size_t>(input_dims_.total()) * sizeof(float);
  output_bytes_ = static_cast<std::size_t>(output_dims_.total()) * sizeof(float);

  cr = cudaMalloc(&input_gpu_, input_bytes_);
  if (cr != cudaSuccess) {
    last_error_ = "TrtEngine: cudaMalloc input failed: " + std::string(cudaGetErrorString(cr));
    return false;
  }

  cr = cudaMalloc(&output_gpu_, output_bytes_);
  if (cr != cudaSuccess) {
    last_error_ = "TrtEngine: cudaMalloc output failed: " + std::string(cudaGetErrorString(cr));
    cudaFree(input_gpu_);
    input_gpu_ = nullptr;
    return false;
  }

  ctx->setTensorAddress(input_name_.c_str(), input_gpu_);
  ctx->setTensorAddress(output_name_.c_str(), output_gpu_);

  return true;
}

bool TrtEngine::infer(void* stream) {
  if (!context_) {
    last_error_ = "TrtEngine: no context";
    return false;
  }
  auto* ctx = static_cast<nvinfer1::IExecutionContext*>(context_);
  cudaStream_t cu_stream = reinterpret_cast<cudaStream_t>(stream);

  if (!ctx->enqueueV3(cu_stream)) {
    last_error_ = "TrtEngine: enqueueV3 failed";
    return false;
  }

  cudaError_t cr = cudaStreamSynchronize(cu_stream);
  if (cr != cudaSuccess) {
    last_error_ = "TrtEngine: cuda sync failed: " + std::string(cudaGetErrorString(cr));
    return false;
  }

  return true;
}

void TrtEngine::cleanup() {
  if (input_gpu_) { cudaFree(input_gpu_); input_gpu_ = nullptr; }
  if (output_gpu_) { cudaFree(output_gpu_); output_gpu_ = nullptr; }

  if (context_) {
    delete static_cast<nvinfer1::IExecutionContext*>(context_);
    context_ = nullptr;
  }
  if (engine_) {
    delete static_cast<nvinfer1::ICudaEngine*>(engine_);
    engine_ = nullptr;
  }
  if (runtime_) {
    delete static_cast<nvinfer1::IRuntime*>(runtime_);
    runtime_ = nullptr;
  }
  if (engine_data_) {
    delete[] static_cast<uint8_t*>(engine_data_);
    engine_data_ = nullptr;
  }
}

}  // namespace edgevision