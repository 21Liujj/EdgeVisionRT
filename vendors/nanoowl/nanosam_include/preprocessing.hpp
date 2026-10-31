#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace nanosam {

// Matches Python NanoSAM predictor preprocessing:
// - RGB conversion
// - letterbox resize (long edge = target_size, maintain aspect ratio)
// - normalize with mean/std
// - HWC -> CHW
// - zero-padding to square target_size x target_size
// - returns FP32 [1, 3, H, W] in CHW layout, H=W=target_size
std::vector<float> preprocessImage(const std::string& image_path, int target_size = 1024);

// For direct tensor input mode (--input-tensor) to bypass preprocessing
std::vector<float> loadTensorFromFile(const std::string& tensor_path);

} // namespace nanosam
