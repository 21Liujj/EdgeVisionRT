#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

namespace edgevision {
namespace cuda_preprocess {

__global__ void kernel_bgr_resize_norm(const uint8_t* __restrict__ src,
                                        int src_w, int src_h, int src_stride,
                                        float* __restrict__ dst,
                                        int dst_w, int dst_h) {
  int dx = blockIdx.x * blockDim.x + threadIdx.x;
  int dy = blockIdx.y * blockDim.y + threadIdx.y;
  if (dx >= dst_w || dy >= dst_h) return;

  float sx = (static_cast<float>(dx) + 0.5f) * static_cast<float>(src_w) /
                 static_cast<float>(dst_w) -
             0.5f;
  float sy = (static_cast<float>(dy) + 0.5f) * static_cast<float>(src_h) /
                 static_cast<float>(dst_h) -
             0.5f;

  int x0 = static_cast<int>(floorf(sx));
  x0 = x0 < 0 ? 0 : x0;
  int x1 = x0 + 1;
  x1 = x1 >= src_w ? src_w - 1 : x1;
  int y0 = static_cast<int>(floorf(sy));
  y0 = y0 < 0 ? 0 : y0;
  int y1 = y0 + 1;
  y1 = y1 >= src_h ? src_h - 1 : y1;

  float dx0 = sx - static_cast<float>(x0);
  float dy0 = sy - static_cast<float>(y0);

  int off00 = (y0 * src_w + x0) * 3;
  int off01 = (y0 * src_w + x1) * 3;
  int off10 = (y1 * src_w + x0) * 3;
  int off11 = (y1 * src_w + x1) * 3;

  int plane_size = dst_w * dst_h;
  int idx = dy * dst_w + dx;

  for (int c = 0; c < 3; ++c) {
    float v00 = static_cast<float>(src[off00 + c]);
    float v01 = static_cast<float>(src[off01 + c]);
    float v10 = static_cast<float>(src[off10 + c]);
    float v11 = static_cast<float>(src[off11 + c]);

    float v0 = v00 * (1.0f - dx0) + v01 * dx0;
    float v1 = v10 * (1.0f - dx0) + v11 * dx0;
    float v = (v0 * (1.0f - dy0) + v1 * dy0) / 255.0f;

    dst[c * plane_size + idx] = v;
  }
}

bool init() {
  int count;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess || count == 0) {
    std::fprintf(stderr, "[cuda_preprocess] no CUDA device\n");
    return false;
  }
  return true;
}

void preprocess(const uint8_t* src_bgr, int src_w, int src_h,
                float* dst_planar, int dst_w, int dst_h,
                void* stream) {
  dim3 block(16, 16);
  dim3 grid((dst_w + block.x - 1) / block.x,
            (dst_h + block.y - 1) / block.y);

  kernel_bgr_resize_norm<<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      src_bgr, src_w, src_h, src_w * 3, dst_planar, dst_w, dst_h);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::fprintf(stderr, "[cuda_preprocess] launch error: %s\n",
                 cudaGetErrorString(err));
  }
}

}  // namespace cuda_preprocess
}  // namespace edgevision