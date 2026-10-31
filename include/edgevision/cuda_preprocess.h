#pragma once
// CUDA preprocess: BGR888 HWC uint8 → planar CHW float32 normalized to [0,1]
// Bilinear resize from arbitrary src (sw,sh) to fixed dst (dw,dh).
// Fused kernel: minimize launch overhead.
#include <cstdint>

namespace edgevision {
namespace cuda_preprocess {

bool init();

void preprocess(const uint8_t* src_bgr, int src_w, int src_h,
                float* dst_planar, int dst_w, int dst_h,
                void* stream = nullptr);

}  // namespace cuda_preprocess
}  // namespace edgevision