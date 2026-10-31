#include "trt_engine.hpp"
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

namespace nanosam {

void Logger::log(Severity severity, const char* msg) noexcept {
    // Only print WARNING, ERROR, and necessary INFO — avoid TensorRT log spam.
    switch (severity) {
        case Severity::kINTERNAL_ERROR:
        case Severity::kERROR:
            std::cerr << "[TRT ERROR] " << msg << std::endl;
            break;
        case Severity::kWARNING:
            std::cerr << "[TRT WARN] " << msg << std::endl;
            break;
        default:
            break; // suppress INFO/VERBOSE
    }
}

// ---------------- DeviceBuffer ----------------

DeviceBuffer::DeviceBuffer(size_t size) : ptr_(nullptr), size_(size) {
    if (size_ > 0) {
        cudaError_t err = cudaMalloc(&ptr_, size_);
        if (err != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(err));
        }
    }
}

DeviceBuffer::~DeviceBuffer() {
    if (ptr_) {
        cudaFree(ptr_);
        ptr_ = nullptr;
    }
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_) {
    other.ptr_ = nullptr;
    other.size_ = 0;
}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
        if (ptr_) cudaFree(ptr_);
        ptr_ = other.ptr_;
        size_ = other.size_;
        other.ptr_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void DeviceBuffer::copyFromHost(const void* host_ptr, size_t size) {
    if (size > size_) {
        throw std::runtime_error("copyFromHost: size exceeds buffer capacity");
    }
    cudaError_t err = cudaMemcpy(ptr_, host_ptr, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpy H2D failed: ") + cudaGetErrorString(err));
    }
}

void DeviceBuffer::copyToHost(void* host_ptr, size_t size) const {
    if (size > size_) {
        throw std::runtime_error("copyToHost: size exceeds buffer capacity");
    }
    cudaError_t err = cudaMemcpy(host_ptr, ptr_, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMemcpy D2H failed: ") + cudaGetErrorString(err));
    }
}

// ---------------- TrtEngine ----------------

TrtEngine::TrtEngine() : stream_(nullptr) {
    cudaError_t err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaStreamCreate failed: ") + cudaGetErrorString(err));
    }
}

TrtEngine::~TrtEngine() {
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

bool TrtEngine::loadEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        std::cerr << "[TrtEngine] Failed to open engine file: " << engine_path << std::endl;
        return false;
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> engine_data(size);
    if (!file.read(engine_data.data(), size)) {
        std::cerr << "[TrtEngine] Failed to read engine file: " << engine_path << std::endl;
        return false;
    }
    file.close();

    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        std::cerr << "[TrtEngine] createInferRuntime failed" << std::endl;
        return false;
    }

    engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (!engine_) {
        std::cerr << "[TrtEngine] deserializeCudaEngine failed for: " << engine_path << std::endl;
        return false;
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        std::cerr << "[TrtEngine] createExecutionContext failed" << std::endl;
        return false;
    }

    return true;
}

int TrtEngine::getNumIOTensors() const {
    return engine_->getNbIOTensors();
}

std::string TrtEngine::getTensorName(int index) const {
    return std::string(engine_->getIOTensorName(index));
}

nvinfer1::Dims TrtEngine::getTensorShape(const std::string& name) const {
    return engine_->getTensorShape(name.c_str());
}

nvinfer1::DataType TrtEngine::getTensorDataType(const std::string& name) const {
    return engine_->getTensorDataType(name.c_str());
}

bool TrtEngine::isTensorInput(const std::string& name) const {
    return engine_->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
}

void TrtEngine::setInputShape(const std::string& name, const nvinfer1::Dims& dims) {
    if (!context_->setInputShape(name.c_str(), dims)) {
        throw std::runtime_error("setInputShape failed for tensor " + name);
    }
}

size_t TrtEngine::getTensorSize(const std::string& name) const {
    // Query the execution context for the resolved (post dynamic-shape) dims,
    // falling back to the engine's static shape when the context isn't set yet.
    nvinfer1::Dims dims = context_->getTensorShape(name.c_str());
    if (dims.nbDims < 0) {
        dims = getTensorShape(name);
    }
    size_t elems = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        int64_t d = dims.d[i];
        if (d < 0) {
            throw std::runtime_error("getTensorSize: dynamic dimension not resolved for tensor " + name);
        }
        elems *= static_cast<size_t>(d);
    }
    nvinfer1::DataType dtype = getTensorDataType(name);
    size_t elem_bytes = 4; // default FLOAT
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT: elem_bytes = 4; break;
        case nvinfer1::DataType::kHALF: elem_bytes = 2; break;
        case nvinfer1::DataType::kINT8: elem_bytes = 1; break;
        case nvinfer1::DataType::kINT32: elem_bytes = 4; break;
        case nvinfer1::DataType::kBOOL: elem_bytes = 1; break;
        default: elem_bytes = 4; break;
    }
    return elems * elem_bytes;
}

void TrtEngine::allocateBuffers() {
    int n = getNumIOTensors();
    for (int i = 0; i < n; ++i) {
        std::string name = getTensorName(i);
        size_t byte_size = getTensorSize(name);
        buffers_.emplace(name, DeviceBuffer(byte_size));
        context_->setTensorAddress(name.c_str(), buffers_.at(name).get());
    }
}

void TrtEngine::setTensorAddress(const std::string& name, void* device_ptr) {
    context_->setTensorAddress(name.c_str(), device_ptr);
}

DeviceBuffer* TrtEngine::getBuffer(const std::string& name) {
    auto it = buffers_.find(name);
    if (it == buffers_.end()) return nullptr;
    return &it->second;
}

bool TrtEngine::infer() {
    bool status = context_->enqueueV3(stream_);
    if (!status) {
        std::cerr << "[TrtEngine] enqueueV3 failed" << std::endl;
        return false;
    }
    cudaError_t err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess) {
        std::cerr << "[TrtEngine] cudaStreamSynchronize failed: " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}

} // namespace nanosam
