#include "nanosam_runtime.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

namespace nanosam {

using Clock = std::chrono::steady_clock;

static double elapsedMs(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

NanoSamRuntime::NanoSamRuntime() {}

bool NanoSamRuntime::loadEncoder(const std::string& encoder_path) {
    if (!encoder_.loadEngine(encoder_path)) return false;
    encoder_.allocateBuffers();
    return true;
}

bool NanoSamRuntime::loadDecoder(const std::string& decoder_path) {
    if (!decoder_.loadEngine(decoder_path)) return false;
    // point_coords / point_labels have a dynamic "num points" dim (-1).
    // Fix it to 2 (top-left, bottom-right box prompt) before allocating buffers.
    nvinfer1::Dims coords_dims;
    coords_dims.nbDims = 3;
    coords_dims.d[0] = 1; coords_dims.d[1] = 2; coords_dims.d[2] = 2;
    decoder_.setInputShape("point_coords", coords_dims);

    nvinfer1::Dims labels_dims;
    labels_dims.nbDims = 2;
    labels_dims.d[0] = 1; labels_dims.d[1] = 2;
    decoder_.setInputShape("point_labels", labels_dims);

    decoder_.allocateBuffers();
    decoder_shapes_set_ = true;
    return true;
}

std::vector<float> NanoSamRuntime::runEncoder(const std::vector<float>& input_tensor, double* latency_ms) {
    DeviceBuffer* image_buf = encoder_.getBuffer("image");
    DeviceBuffer* emb_buf = encoder_.getBuffer("image_embeddings");
    if (!image_buf || !emb_buf) {
        throw std::runtime_error("runEncoder: expected tensors 'image' / 'image_embeddings' not found");
    }
    size_t expected_elems = static_cast<size_t>(1) * 3 * 1024 * 1024;
    if (input_tensor.size() != expected_elems) {
        throw std::runtime_error("runEncoder: input tensor size mismatch, expected 1x3x1024x1024");
    }

    image_buf->copyFromHost(input_tensor.data(), input_tensor.size() * sizeof(float));

    auto t0 = Clock::now();
    if (!encoder_.infer()) {
        throw std::runtime_error("runEncoder: enqueueV3 failed");
    }
    auto t1 = Clock::now();
    if (latency_ms) *latency_ms = elapsedMs(t0, t1);

    size_t emb_elems = static_cast<size_t>(256) * 64 * 64;
    std::vector<float> embeddings(emb_elems);
    emb_buf->copyToHost(embeddings.data(), embeddings.size() * sizeof(float));
    return embeddings;
}

InferenceResult NanoSamRuntime::runDecoder(const std::vector<float>& embeddings,
                                            const Box& box_orig_image,
                                            int orig_width, int orig_height,
                                            double* latency_ms) {
    if (!decoder_shapes_set_) {
        throw std::runtime_error("runDecoder: decoder input shapes not set (loadDecoder not called)");
    }

    // Mirrors preprocess_points(): scale = size / max(height, width), size=1024
    // (orig_image_encoder_size), applied uniformly to both axes.
    const float size = 1024.0f;
    const float scale = size / static_cast<float>(std::max(orig_width, orig_height));

    std::array<float, 4> point_coords = {
        box_orig_image.x1 * scale, box_orig_image.y1 * scale,
        box_orig_image.x2 * scale, box_orig_image.y2 * scale
    };
    // labels: 2 = top-left, 3 = bottom-right (matches predictor.py box prompt convention)
    std::array<float, 2> point_labels = {2.0f, 3.0f};

    std::vector<float> mask_input(static_cast<size_t>(1) * 1 * 256 * 256, 0.0f);
    std::array<float, 1> has_mask_input = {0.0f};

    DeviceBuffer* emb_buf = decoder_.getBuffer("image_embeddings");
    DeviceBuffer* coords_buf = decoder_.getBuffer("point_coords");
    DeviceBuffer* labels_buf = decoder_.getBuffer("point_labels");
    DeviceBuffer* mask_input_buf = decoder_.getBuffer("mask_input");
    DeviceBuffer* has_mask_buf = decoder_.getBuffer("has_mask_input");
    DeviceBuffer* iou_buf = decoder_.getBuffer("iou_predictions");
    DeviceBuffer* low_res_buf = decoder_.getBuffer("low_res_masks");

    if (!emb_buf || !coords_buf || !labels_buf || !mask_input_buf || !has_mask_buf || !iou_buf || !low_res_buf) {
        throw std::runtime_error("runDecoder: one or more expected decoder tensors not found");
    }

    emb_buf->copyFromHost(embeddings.data(), embeddings.size() * sizeof(float));
    coords_buf->copyFromHost(point_coords.data(), point_coords.size() * sizeof(float));
    labels_buf->copyFromHost(point_labels.data(), point_labels.size() * sizeof(float));
    mask_input_buf->copyFromHost(mask_input.data(), mask_input.size() * sizeof(float));
    has_mask_buf->copyFromHost(has_mask_input.data(), has_mask_input.size() * sizeof(float));

    auto t0 = Clock::now();
    if (!decoder_.infer()) {
        throw std::runtime_error("runDecoder: enqueueV3 failed");
    }
    auto t1 = Clock::now();
    if (latency_ms) *latency_ms = elapsedMs(t0, t1);

    InferenceResult result;
    result.low_res_masks.resize(static_cast<size_t>(4) * 256 * 256);
    low_res_buf->copyToHost(result.low_res_masks.data(), result.low_res_masks.size() * sizeof(float));
    iou_buf->copyToHost(result.iou_predictions.data(), result.iou_predictions.size() * sizeof(float));

    return result;
}

cv::Mat NanoSamRuntime::postprocessMask(const std::vector<float>& low_res_masks_flat,
                                         int orig_width, int orig_height) {
    // Fixed slot0: first 256*256 elements of [4,256,256].
    const int lr = 256;
    cv::Mat slot0(lr, lr, CV_32F, const_cast<float*>(low_res_masks_flat.data()));

    // Mirrors upscale_mask(): crop out the zero-padded region using the same
    // aspect-ratio formula as preprocessing, but evaluated at size=256.
    int lim_x, lim_y;
    if (orig_width > orig_height) {
        lim_x = lr;
        lim_y = static_cast<int>(lr * static_cast<double>(orig_height) / static_cast<double>(orig_width));
    } else {
        lim_x = static_cast<int>(lr * static_cast<double>(orig_width) / static_cast<double>(orig_height));
        lim_y = lr;
    }
    lim_x = std::clamp(lim_x, 1, lr);
    lim_y = std::clamp(lim_y, 1, lr);

    cv::Mat cropped = slot0(cv::Rect(0, 0, lim_x, lim_y)).clone();

    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(orig_width, orig_height), 0, 0, cv::INTER_LINEAR);

    // Python: mask logits > 0 -> binary foreground (matches eval protocol slot0 threshold).
    cv::Mat binary;
    cv::threshold(resized, binary, 0.0, 255.0, cv::THRESH_BINARY);
    cv::Mat out8u;
    binary.convertTo(out8u, CV_8U);
    return out8u;
}

} // namespace nanosam
