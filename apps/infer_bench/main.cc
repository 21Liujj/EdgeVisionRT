#ifdef __linux__
// infer_bench: single-camera YOLOv8-n TensorRT inference benchmark.
// Usage: infer_bench [--engine models/yolov8n_fp16.engine] [--device /dev/video0]
//                     [--frames 100] [--conf 0.25]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "edgevision/cuda_preprocess.h"
#include "edgevision/mjpeg_decoder.h"
#include "edgevision/stream_session.h"
#include "edgevision/trt_engine.h"

namespace {

struct Det {
  float x, y, w, h, conf;
  int cls;
};

std::vector<Det> nms(const std::vector<Det>& dets, float iou_thresh) {
  std::vector<Det> result;
  std::vector<bool> suppressed(dets.size(), false);

  for (std::size_t i = 0; i < dets.size(); ++i) {
    if (suppressed[i]) continue;
    result.push_back(dets[i]);
    for (std::size_t j = i + 1; j < dets.size(); ++j) {
      if (suppressed[j]) continue;

      float x1_i = dets[i].x - dets[i].w / 2.0f;
      float y1_i = dets[i].y - dets[i].h / 2.0f;
      float x2_i = dets[i].x + dets[i].w / 2.0f;
      float y2_i = dets[i].y + dets[i].h / 2.0f;

      float x1_j = dets[j].x - dets[j].w / 2.0f;
      float y1_j = dets[j].y - dets[j].h / 2.0f;
      float x2_j = dets[j].x + dets[j].w / 2.0f;
      float y2_j = dets[j].y + dets[j].h / 2.0f;

      float ix1 = std::max(x1_i, x1_j);
      float iy1 = std::max(y1_i, y1_j);
      float ix2 = std::min(x2_i, x2_j);
      float iy2 = std::min(y2_i, y2_j);
      float iw = std::max(0.0f, ix2 - ix1);
      float ih = std::max(0.0f, iy2 - iy1);
      float inter = iw * ih;

      float area_i = (x2_i - x1_i) * (y2_i - y1_i);
      float area_j = (x2_j - x1_j) * (y2_j - y1_j);
      float iou = inter / (area_i + area_j - inter + 1e-6f);

      if (iou > iou_thresh) suppressed[j] = true;
    }
  }
  return result;
}

std::vector<Det> postprocess(const float* output, int num_anchors, int num_classes,
                              float conf_thresh, float iou_thresh) {
  std::vector<Det> dets;
  for (int i = 0; i < num_anchors; ++i) {
    const float* row = output + i;
    float max_conf = 0;
    int max_cls = -1;
    for (int c = 4; c < 4 + num_classes; ++c) {
      float score = row[c * num_anchors];
      if (score > max_conf) {
        max_conf = score;
        max_cls = c - 4;
      }
    }
    if (max_conf < conf_thresh) continue;

    float cx = row[0 * num_anchors];
    float cy = row[1 * num_anchors];
    float w = row[2 * num_anchors];
    float h = row[3 * num_anchors];

    dets.push_back({cx, cy, w, h, max_conf, max_cls});
  }
  return nms(dets, iou_thresh);
}

}  // namespace

int main(int argc, char** argv) {
  std::string engine_path = "models/yolov8n_fp16.engine";
  std::string device = "/dev/video0";
  int num_frames = 100;
  int warmup = 20;
  float conf_thresh = 0.25f;
  float iou_thresh = 0.45f;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--engine") == 0 && i + 1 < argc) engine_path = argv[++i];
    else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) device = argv[++i];
    else if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) num_frames = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--conf") == 0 && i + 1 < argc) conf_thresh = std::atof(argv[++i]);
    else if (std::strcmp(argv[i], "--iou") == 0 && i + 1 < argc) iou_thresh = std::atof(argv[++i]);
  }

  std::printf("[infer_bench] engine=%s device=%s frames=%d warmup=%d conf=%.2f iou=%.2f\n",
              engine_path.c_str(), device.c_str(), num_frames, warmup, conf_thresh, iou_thresh);

  edgevision::TrtEngine engine;
  if (!engine.load(engine_path)) {
    std::fprintf(stderr, "[infer_bench] FAIL load engine: %s\n", engine.last_error().c_str());
    return 1;
  }
  std::printf("[infer_bench] engine loaded: input [%lldx%lldx%lldx%lld] output [%lldx%lldx%lldx%lld]\n",
              (long long)engine.input_dims().n, (long long)engine.input_dims().c,
              (long long)engine.input_dims().h, (long long)engine.input_dims().w,
              (long long)engine.output_dims().n, (long long)engine.output_dims().c,
              (long long)engine.output_dims().h, (long long)engine.output_dims().w);

  if (!engine.prepare_io()) {
    std::fprintf(stderr, "[infer_bench] FAIL prepare_io: %s\n", engine.last_error().c_str());
    return 1;
  }

  edgevision::StreamSession sess;
  if (!sess.start(device, 1280, 720, 0x47504A4D, 6, 4, 4)) {
    std::fprintf(stderr, "[infer_bench] FAIL start session: %s\n", sess.last_error().c_str());
    return 1;
  }
  auto stats0 = sess.stats();
  std::printf("[infer_bench] streaming %ux%u driver_fps=%.2f\n",
              stats0.negotiated_w, stats0.negotiated_h, stats0.driver_fps);

  int target_w = static_cast<int>(engine.input_dims().w);
  int target_h = static_cast<int>(engine.input_dims().h);
  int num_classes = static_cast<int>(engine.output_dims().c) - 4;
  int num_anchors = static_cast<int>(engine.output_dims().w);

  if (!edgevision::cuda_preprocess::init()) {
    std::fprintf(stderr, "[infer_bench] FAIL cuda preprocess init\n");
    return 1;
  }

  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);

  void* bgr_gpu = nullptr;
  cudaMalloc(&bgr_gpu, 1280 * 720 * 3);

  std::vector<double> latencies;
  latencies.reserve(static_cast<std::size_t>(num_frames));
  int total_dets = 0;
  int ok = 0;
  int fail_decode = 0;
  int fail_infer = 0;

  for (int f = 0; f < num_frames + warmup; ++f) {
    edgevision::FrameLease lease;
    if (!sess.consume(lease, 2000)) {
      std::printf("[infer_bench] frame %d TIMEOUT\n", f);
      continue;
    }
    if (!lease.valid()) continue;

    std::vector<uint8_t> bgr;
    int dec_w = 0, dec_h = 0;
    std::string dec_err;
    if (!edgevision::mjpeg_decode_to_bgr(lease->data.data(), lease->data.size(),
                                          bgr, dec_w, dec_h, dec_err)) {
      fail_decode++;
      std::fprintf(stderr, "[infer_bench] frame %d decode FAIL: %s\n", f, dec_err.c_str());
      continue;
    }

    cudaMemcpy(bgr_gpu, bgr.data(),
                    static_cast<std::size_t>(dec_w) * dec_h * 3,
                    cudaMemcpyHostToDevice);

    auto t0 = std::chrono::steady_clock::now();

    edgevision::cuda_preprocess::preprocess(static_cast<const uint8_t*>(bgr_gpu),
                                             dec_w, dec_h,
                                             static_cast<float*>(engine.input_gpu()),
                                             target_w, target_h, stream);

    if (!engine.infer(stream)) {
      fail_infer++;
      std::fprintf(stderr, "[infer_bench] frame %d infer FAIL: %s\n", f, engine.last_error().c_str());
      continue;
    }

    cudaStreamSynchronize(stream);

    auto t1 = std::chrono::steady_clock::now();
    double lat_ms = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;

    std::vector<float> output(static_cast<std::size_t>(engine.output_dims().total()));
    cudaMemcpy(output.data(), engine.output_gpu(),
               output.size() * sizeof(float), cudaMemcpyDeviceToHost);

    auto dets = postprocess(output.data(), num_anchors, num_classes, conf_thresh, iou_thresh);

    if (f >= warmup) {
      latencies.push_back(lat_ms);
      total_dets += static_cast<int>(dets.size());
      ok++;

      if (f % 20 == 0 || f == warmup) {
        std::printf("[infer_bench] frame %4d  lat=%.2fms  dets=%zu\n",
                    f, lat_ms, dets.size());
      }
    }
  }

  cudaStreamDestroy(stream);
  cudaFree(bgr_gpu);
  sess.stop();

  if (latencies.empty()) {
    std::printf("[infer_bench] FAIL: no valid inference results\n");
    return 1;
  }

  std::sort(latencies.begin(), latencies.end());
  double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
  double avg = sum / static_cast<double>(latencies.size());
  double p50 = latencies[latencies.size() * 50 / 100];
  double p95 = latencies[latencies.size() * 95 / 100];
  double p99 = latencies[latencies.size() * 99 / 100];

  std::printf("[infer_bench] SUMMARY frames=%d ok=%d decode_fail=%d infer_fail=%d\n",
              num_frames, ok, fail_decode, fail_infer);
  std::printf("[infer_bench] SUMMARY latency_ms  avg=%.2f p50=%.2f p95=%.2f p99=%.2f "
              "min=%.2f max=%.2f\n",
              avg, p50, p95, p99, latencies.front(), latencies.back());
  std::printf("[infer_bench] SUMMARY dets_total=%d dets_per_frame=%.1f\n",
              total_dets, ok ? static_cast<double>(total_dets) / ok : 0);
  std::printf("[infer_bench] PASS\n");
  return 0;
}
#else
#include <cstdio>
int main() {
  std::printf("[infer_bench] SKIP (requires Linux + CUDA)\n");
  return 0;
}
#endif