#include "preprocessing.hpp"
#include <fstream>
#include <stdexcept>
#include <cmath>

namespace nanosam {

// Mirrors third_party/nanosam/nanosam/utils/predictor.py::preprocess_image
//
// Python reference:
//   aspect_ratio = width / height
//   if aspect_ratio >= 1: resize_width = size; resize_height = size / aspect_ratio
//   else: resize_height = size; resize_width = size * aspect_ratio
//   PIL resize(resize_width, resize_height)   [PIL default resample filter]
//   normalize: (pixel - mean) / std, mean=[123.675,116.28,103.53], std=[58.395,57.12,57.375]
//   HWC -> CHW, zero-pad into a (size, size) canvas anchored at top-left.
//
// NOTE: PIL.Image.resize's default resampling filter differs from OpenCV's
// INTER_LINEAR. This is a known, documented source of preprocessing delta
// (see docs/cpp_runtime.md and results/stage_cpp_runtime/preprocess_parity.json).
// We use cv::INTER_LINEAR here per project scope (no custom CUDA resize kernel).
std::vector<float> preprocessImage(const std::string& image_path, int target_size) {
    cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
        throw std::runtime_error("preprocessImage: failed to read image " + image_path);
    }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    const int width = rgb.cols;
    const int height = rgb.rows;
    const double aspect_ratio = static_cast<double>(width) / static_cast<double>(height);

    int resize_width, resize_height;
    if (aspect_ratio >= 1.0) {
        resize_width = target_size;
        resize_height = static_cast<int>(target_size / aspect_ratio);
    } else {
        resize_height = target_size;
        resize_width = static_cast<int>(target_size * aspect_ratio);
    }
    resize_width = std::max(1, resize_width);
    resize_height = std::max(1, resize_height);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(resize_width, resize_height), 0, 0, cv::INTER_LINEAR);

    const float mean[3] = {123.675f, 116.28f, 103.53f};
    const float std_[3] = {58.395f, 57.12f, 57.375f};

    // Output tensor: [1, 3, target_size, target_size], CHW, zero-padded.
    std::vector<float> tensor(static_cast<size_t>(3) * target_size * target_size, 0.0f);

    for (int y = 0; y < resize_height; ++y) {
        const uint8_t* row = resized.ptr<uint8_t>(y);
        for (int x = 0; x < resize_width; ++x) {
            for (int c = 0; c < 3; ++c) {
                float pixel = static_cast<float>(row[x * 3 + c]);
                float norm = (pixel - mean[c]) / std_[c];
                size_t idx = static_cast<size_t>(c) * target_size * target_size
                           + static_cast<size_t>(y) * target_size + x;
                tensor[idx] = norm;
            }
        }
    }

    return tensor;
}

std::vector<float> loadTensorFromFile(const std::string& tensor_path) {
    std::ifstream file(tensor_path, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        throw std::runtime_error("loadTensorFromFile: failed to open " + tensor_path);
    }
    std::streamsize size_bytes = file.tellg();
    file.seekg(0, std::ios::beg);

    if (size_bytes % sizeof(float) != 0) {
        throw std::runtime_error("loadTensorFromFile: file size not a multiple of float size");
    }
    size_t num_floats = static_cast<size_t>(size_bytes) / sizeof(float);
    std::vector<float> data(num_floats);
    if (!file.read(reinterpret_cast<char*>(data.data()), size_bytes)) {
        throw std::runtime_error("loadTensorFromFile: failed to read " + tensor_path);
    }
    return data;
}

} // namespace nanosam
