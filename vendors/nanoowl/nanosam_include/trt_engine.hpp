#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace nanosam {

// Simple TensorRT logger
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

// RAII wrapper for CUDA device memory
class DeviceBuffer {
public:
    DeviceBuffer() : ptr_(nullptr), size_(0) {}
    explicit DeviceBuffer(size_t size);
    ~DeviceBuffer();

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept;
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;

    void* get() const { return ptr_; }
    size_t size() const { return size_; }

    void copyFromHost(const void* host_ptr, size_t size);
    void copyToHost(void* host_ptr, size_t size) const;

private:
    void* ptr_;
    size_t size_;
};

// TensorRT engine wrapper with enqueueV3 API
class TrtEngine {
public:
    TrtEngine();
    ~TrtEngine();

    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    // Load engine from file
    bool loadEngine(const std::string& engine_path);

    // Get tensor info
    int getNumIOTensors() const;
    std::string getTensorName(int index) const;
    nvinfer1::Dims getTensorShape(const std::string& name) const;
    nvinfer1::DataType getTensorDataType(const std::string& name) const;
    bool isTensorInput(const std::string& name) const;

    // Set concrete shape for a dynamic input tensor (must call before allocateBuffers)
    void setInputShape(const std::string& name, const nvinfer1::Dims& dims);

    // Allocate device buffers for all I/O tensors (resolves dynamic shapes set via setInputShape)
    void allocateBuffers();

    // Set tensor address for execution
    void setTensorAddress(const std::string& name, void* device_ptr);

    // Execute inference
    bool infer();

    // Helper: get total elements in a tensor
    size_t getTensorSize(const std::string& name) const;

    // Get device buffer by name (after allocateBuffers)
    DeviceBuffer* getBuffer(const std::string& name);

private:
    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_;

    std::unordered_map<std::string, DeviceBuffer> buffers_;
};

} // namespace nanosam
