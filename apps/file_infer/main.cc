// file_infer: replay images/video frames through the SAME pipeline as edgevisiond.
// Usage:
//   file_infer --engine models/yolov8n_fp16.engine --input img_or_dir_or_list [--out replay_xxx]
// Output dir is replay_<stamp>/ (explicitly NOT sessions/): mock/replay never
// counts as on-device acceptance (repo rule).
//
// Deps: libjpeg (decode input jpgs) + CUDA + TRT. No V4L2, no camera.
#ifdef __linux__

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <numeric>
#include <string>
#include <sys/stat.h>
#include <vector>

#include <cuda_runtime.h>

#include "edgevision/cuda_preprocess.h"
#include "edgevision/mjpeg_decoder.h"
#include "edgevision/trt_engine.h"

namespace {

struct Det {
  float x, y, w, h, conf;
  int cls;
};

std::vector<Det> nms(std::vector<Det> dets, float iou_thresh) {
  std::vector<Det> result;
  std::vector<bool> suppressed(dets.size(), false);
  for (std::size_t i = 0; i < dets.size(); ++i) {
    if (suppressed[i]) continue;
    result.push_back(dets[i]);
    float x1_i = dets[i].x - dets[i].w / 2.0f, y1_i = dets[i].y - dets[i].h / 2.0f;
    float x2_i = dets[i].x + dets[i].w / 2.0f, y2_i = dets[i].y + dets[i].h / 2.0f;
    float area_i = (x2_i - x1_i) * (y2_i - y1_i);
    for (std::size_t j = i + 1; j < dets.size(); ++j) {
      if (suppressed[j]) continue;
      float x1_j = dets[j].x - dets[j].w / 2.0f, y1_j = dets[j].y - dets[j].h / 2.0f;
      float x2_j = dets[j].x + dets[j].w / 2.0f, y2_j = dets[j].y + dets[j].h / 2.0f;
      float ix1 = std::max(x1_i, x1_j), iy1 = std::max(y1_i, y1_j);
      float ix2 = std::min(x2_i, x2_j), iy2 = std::min(y2_i, y2_j);
      float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
      float area_j = (x2_j - x1_j) * (y2_j - y1_j);
      if (inter / (area_i + area_j - inter + 1e-6f) > iou_thresh) suppressed[j] = true;
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
    dets.push_back({row[0 * num_anchors], row[1 * num_anchors], row[2 * num_anchors],
                    row[3 * num_anchors], max_conf, max_cls});
  }
  return nms(dets, iou_thresh);
}

bool endswith(const std::string& s, const std::string& suf) {
  return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

void collect_inputs(const std::string& in, std::vector<std::string>& out) {
  struct stat st;
  if (stat(in.c_str(), &st) != 0) return;
  if (S_ISREG(st.st_mode)) {
    out.push_back(in);
    return;
  }
  if (S_ISDIR(st.st_mode)) {
    DIR* d = opendir(in.c_str());
    if (!d) return;
    struct dirent* e;
    while ((e = readdir(d)) != nullptr) {
      std::string n = e->d_name;
      if (endswith(n, ".jpg") || endswith(n, ".jpeg") || endswith(n, ".JPG"))
        out.push_back(in + "/" + n);
    }
    closedir(d);
    std::sort(out.begin(), out.end());
  }
}

std::vector<uint8_t> read_file(const std::string& p) {
  FILE* fp = std::fopen(p.c_str(), "rb");
  if (!fp) return {};
  std::fseek(fp, 0, SEEK_END);
  long n = std::ftell(fp);
  std::fseek(fp, 0, SEEK_SET);
  std::vector<uint8_t> b(n > 0 ? n : 0);
  if (n > 0) std::fread(b.data(), 1, n, fp);
  std::fclose(fp);
  return b;
}

std::string stamp() {
  std::time_t t = std::time(nullptr);
  struct tm tmv;
  localtime_r(&t, &tmv);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tmv);
  return std::string(buf);
}

}  // namespace

int main(int argc, char** argv) {
  std::string engine_path = "models/yolov8n_fp16.engine";
  std::string input;
  std::string outdir;
  int warmup = 5;
  float conf_thresh = 0.25f, iou_thresh = 0.45f;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--engine") == 0 && i + 1 < argc) engine_path = argv[++i];
    else if (std::strcmp(argv[i], "--input") == 0 && i + 1 < argc) input = argv[++i];
    else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) outdir = argv[++i];
    else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) warmup = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--conf") == 0 && i + 1 < argc)
      conf_thresh = std::atof(argv[++i]);
    else if (std::strcmp(argv[i], "--iou") == 0 && i + 1 < argc) iou_thresh = std::atof(argv[++i]);
  }
  if (input.empty()) {
    std::fprintf(stderr, "usage: file_infer --input img|dir [--engine ...] [--out replay_x]\n");
    return 1;
  }
  std::vector<std::string> files;
  collect_inputs(input, files);
  if (files.empty()) {
    std::fprintf(stderr, "[file_infer] no jpg inputs under %s\n", input.c_str());
    return 1;
  }
  if (outdir.empty()) outdir = "replay_" + stamp();
  std::string mk = "mkdir -p '" + outdir + "'";
  if (std::system(mk.c_str()) != 0) return 1;

  edgevision::TrtEngine engine;
  if (!engine.load(engine_path) || !engine.prepare_io()) {
    std::fprintf(stderr, "[file_infer] FAIL engine: %s\n", engine.last_error().c_str());
    return 1;
  }
  int TW = static_cast<int>(engine.input_dims().w);
  int TH = static_cast<int>(engine.input_dims().h);
  int NC = static_cast<int>(engine.output_dims().c) - 4;
  int NA = static_cast<int>(engine.output_dims().w);
  if (!edgevision::cuda_preprocess::init()) return 1;
  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);
  void* bgr_gpu = nullptr;
  cudaMalloc(&bgr_gpu, 1920 * 1080 * 3);

  std::vector<float> output(static_cast<std::size_t>(NA * (4 + NC)));
  std::vector<double> lats;
  uint64_t dets_total = 0;
  FILE* ev = std::fopen((outdir + "/events.jsonl").c_str(), "w");
  std::printf("[file_infer] replay %zu files -> %s/ (REPLAY, not on-device)\n", files.size(),
              outdir.c_str());
  for (std::size_t fi = 0; fi < files.size(); ++fi) {
    auto jpg = read_file(files[fi]);
    std::vector<uint8_t> bgr;
    int w = 0, h = 0;
    std::string derr;
    if (!edgevision::mjpeg_decode_to_bgr(jpg.data(), jpg.size(), bgr, w, h, derr)) {
      std::fprintf(stderr, "[file_infer] decode FAIL %s: %s\n", files[fi].c_str(),
                   derr.c_str());
      continue;
    }
    cudaMemcpy(bgr_gpu, bgr.data(), static_cast<std::size_t>(w) * h * 3,
               cudaMemcpyHostToDevice);
    auto t0 = std::chrono::steady_clock::now();
    edgevision::cuda_preprocess::preprocess(static_cast<const uint8_t*>(bgr_gpu), w, h,
                                            static_cast<float*>(engine.input_gpu()), TW, TH,
                                            stream);
    if (!engine.infer(stream)) continue;
    cudaStreamSynchronize(stream);
    auto t1 = std::chrono::steady_clock::now();
    double ms =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    cudaMemcpy(output.data(), engine.output_gpu(), output.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    auto dets = postprocess(output.data(), NA, NC, conf_thresh, iou_thresh);
    bool rec = (static_cast<int>(fi) >= warmup);
    if (rec) {
      lats.push_back(ms);
      dets_total += dets.size();
      if (ev) {
        std::fprintf(ev, "{\"file\":\"%s\",\"lat_ms\":%.2f,\"dets\":%zu,\"boxes\":[",
                     files[fi].c_str(), ms, dets.size());
        for (std::size_t k = 0; k < dets.size(); ++k) {
          if (k) std::fprintf(ev, ",");
          std::fprintf(ev, "{\"x\":%.4f,\"y\":%.4f,\"w\":%.4f,\"h\":%.4f,\"conf\":%.3f,"
                           "\"cls\":%d}",
                       dets[k].x, dets[k].y, dets[k].w, dets[k].h, dets[k].conf,
                       dets[k].cls);
        }
        std::fprintf(ev, "]}\n");
      }
    }
    if (fi % 20 == 0) std::printf("[file_infer] %zu/%zu lat=%.2fms dets=%zu %s\n", fi,
                                 files.size(), ms, dets.size(), files[fi].c_str());
  }
  if (ev) std::fclose(ev);
  std::sort(lats.begin(), lats.end());
  double avg = 0, p50 = 0, p99 = 0;
  if (!lats.empty()) {
    avg = std::accumulate(lats.begin(), lats.end(), 0.0) / lats.size();
    p50 = lats[lats.size() * 50 / 100];
    p99 = lats[lats.size() * 99 / 100];
  }
  FILE* sf = std::fopen((outdir + "/summary.json").c_str(), "w");
  if (sf) {
    std::fprintf(sf, "{\"replay\":true,\"files\":%zu,\"frames_ok\":%zu,\"lat_avg_ms\":%.2f,"
                     "\"lat_p50_ms\":%.2f,\"lat_p99_ms\":%.2f,\"dets_total\":%llu}\n",
                 files.size(), lats.size(), avg, p50, p99,
                 (unsigned long long)dets_total);
    std::fclose(sf);
  }
  std::printf("[file_infer] REPLAY SUMMARY files=%zu ok=%zu avg=%.2f p50=%.2f p99=%.2f "
              "dets=%llu out=%s/\n",
              files.size(), lats.size(), avg, p50, p99, (unsigned long long)dets_total,
              outdir.c_str());
  cudaFree(bgr_gpu);
  cudaStreamDestroy(stream);
  std::printf("[file_infer] REPLAY DONE (not on-device acceptance)\n");
  return 0;
}
#else
#include <cstdio>
int main() {
  std::printf("[file_infer] SKIP (requires Linux + CUDA)\n");
  return 0;
}
#endif
