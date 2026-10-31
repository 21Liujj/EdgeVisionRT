#pragma once

#include "trt_engine.hpp"
#include <opencv2/opencv.hpp>
#include <array>
#include <string>
#include <vector>

namespace nanosam {

struct Box {
    float x1, y1, x2, y2;
};

struct LatencyStats {
    double mean_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
};

struct InferenceResult {
    // low_res_masks: [4, 256, 256] flattened, slot0 = first 256*256 elements
    std::vector<float> low_res_masks;
    // iou_predictions: [4]
    std::array<float, 4> iou_predictions;
    double encoder_latency_ms = 0.0;
    double decoder_latency_ms = 0.0;
};

// Encapsulates encoder + decoder engines and reproduces the Python NanoSAM
// Predictor contract (third_party/nanosam/nanosam/utils/predictor.py) exactly:
// box prompt as two points labeled [2, 3] (top-left, bottom-right),
// fixed mask_input=zeros / has_mask_input=0, fixed slot0 mask selection.
class NanoSamRuntime {
public:
    NanoSamRuntime();

    bool loadEncoder(const std::string& encoder_path);
    bool loadDecoder(const std::string& decoder_path);

    // Run encoder on a preprocessed [1,3,1024,1024] tensor (CHW, normalized).
    // Returns image_embeddings [256,64,64] flattened, and sets encoder latency.
    std::vector<float> runEncoder(const std::vector<float>& input_tensor, double* latency_ms = nullptr);

    // Run decoder given embeddings and a box prompt (image-space pixel coords
    // in the *original* image, matching preprocess_points scaling contract).
    InferenceResult runDecoder(const std::vector<float>& embeddings,
                                const Box& box_orig_image,
                                int orig_width, int orig_height,
                                double* latency_ms = nullptr);

    // Postprocess: slot0 low_res_mask [256,256] -> binary mask at original
    // image resolution, replicating Python upscale_mask() cropping + bilinear
    // resize + threshold > 0.
    cv::Mat postprocessMask(const std::vector<float>& low_res_masks_flat,
                             int orig_width, int orig_height);

    TrtEngine& encoderEngine() { return encoder_; }
    TrtEngine& decoderEngine() { return decoder_; }

private:
    TrtEngine encoder_;
    TrtEngine decoder_;
    bool decoder_shapes_set_ = false;
};

} // namespace nanosam
