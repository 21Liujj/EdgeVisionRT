#pragma once
// TrtEngine: minimal RAII wrapper for TensorRT inference.
// Loads a serialized engine, allocates I/O on GPU, runs synchronous inference.
// Single-context, single-stream. No batching at this layer.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace edgevision {

class TrtEngine {
 public:
  struct Dims {
    int64_t n = 1, c = 3, h = 640, w = 640;
    int64_t total() const { return n * c * h * w; }
  };

  TrtEngine() = default;
  ~TrtEngine();

  TrtEngine(const TrtEngine&) = delete;
  TrtEngine& operator=(const TrtEngine&) = delete;

  bool load(const std::string& engine_path);

  bool prepare_io();

  bool infer(void* stream = nullptr);

  const Dims& input_dims() const { return input_dims_; }
  const Dims& output_dims() const { return output_dims_; }

  void* input_gpu() const { return input_gpu_; }
  void* output_gpu() const { return output_gpu_; }
  const std::string& last_error() const { return last_error_; }

  const std::string& input_name() const { return input_name_; }
  const std::string& output_name() const { return output_name_; }

 private:
  void cleanup();

  void* engine_data_ = nullptr;
  std::size_t engine_size_ = 0;
  void* runtime_ = nullptr;
  void* engine_ = nullptr;
  void* context_ = nullptr;

  Dims input_dims_;
  Dims output_dims_;
  std::string input_name_;
  std::string output_name_;

  void* input_gpu_ = nullptr;
  void* output_gpu_ = nullptr;
  std::size_t input_bytes_ = 0;
  std::size_t output_bytes_ = 0;

  std::string last_error_;
};

}  // namespace edgevision