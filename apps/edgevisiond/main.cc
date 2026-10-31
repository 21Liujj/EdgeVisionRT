// edgevisiond: persistent inference daemon (M6 service layer).
//
// infer_bench runs N frames and exits. edgevisiond instead owns the camera
// for its whole lifetime and answers control commands over a Unix socket:
//
//   STATUS  -> {"ok":true,"state":"IDLE|STARTING|RUNNING|STOPPING","frames":N,...}
//   START   -> begin a session (accepted = StreamSession::start returned true)
//   STOP    -> end session, flush summary.json atomically, then report completed
//
// Iron rule: sent != accepted != completed. START replies accepted only after
// the V4L2 session is really streaming; STOP replies completed only after the
// session stopped AND summary.json was renamed into place. STATUS frames_captured
// is the only proof that frames really flowed.
//
// Session artifacts: ./sessions/<YYYYMMDD_HHMMSS>/events.jsonl + summary.json
// (written to .tmp then renamed, so a crash never leaves a half file as final).
//
// Usage:
//   edgevisiond [--engine models/yolov8n_fp16.engine] [--device /dev/video0]
//               [--sock /tmp/edgevisiond.sock] [--sessions ./sessions]
//               [--conf 0.25] [--iou 0.45] [--warmup 10]
#ifdef __linux__

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

extern "C" {
#include <jpeglib.h>
}

#include <cuda_runtime.h>

#include "edgevision/cuda_preprocess.h"
#include "edgevision/mjpeg_decoder.h"
#include "edgevision/stream_session.h"
#include "edgevision/trt_engine.h"

namespace {

volatile std::sig_atomic_t g_stop_process = 0;
void on_signal(int) { g_stop_process = 1; }

enum class DaemonState { IDLE, STARTING, RUNNING, STOPPING };

const char* state_str(DaemonState s) {
  switch (s) {
    case DaemonState::IDLE: return "IDLE";
    case DaemonState::STARTING: return "STARTING";
    case DaemonState::RUNNING: return "RUNNING";
    case DaemonState::STOPPING: return "STOPPING";
  }
  return "UNKNOWN";
}

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

uint64_t wall_ns_now() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + ts.tv_nsec;
}

std::string session_stamp() {
  std::time_t t = std::time(nullptr);
  struct tm tmv;
  localtime_r(&t, &tmv);
  char buf[32];
  std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tmv);
  return std::string(buf);
}

struct Daemon {
  // config
  std::string engine_path = "models/yolov8n_fp16.engine";
  std::string device = "/dev/video0";
  std::string sock_path = "/tmp/edgevisiond.sock";
  std::string sessions_dir = "./sessions";
  float conf_thresh = 0.25f;
  float iou_thresh = 0.45f;
  int warmup = 10;
  int http_port = 8080;  // 0 = disable web UI
  // ---- M8a review queue: streak trigger dumps crop+meta here (atomic .tmp->rename).
  // Workers (off-board) scan for dirs containing meta.json (final, not .tmp).
  std::string review_dir = "./pending_review";
  int review_keep = 20;  // prune oldest review dirs beyond this count

  // runtime
  std::mutex mu;
  DaemonState state = DaemonState::IDLE;
  uint64_t frames_ok = 0;
  uint64_t decode_fail = 0;
  uint64_t infer_fail = 0;
  uint64_t timeouts = 0;
  std::string session_dir;
  std::string last_error;

  edgevision::TrtEngine engine;
  edgevision::StreamSession sess;
  int target_w = 640, target_h = 640;
  int num_classes = 80, num_anchors = 8400;
  cudaStream_t stream = nullptr;
  void* bgr_gpu = nullptr;

  std::atomic<bool> stop_session{false};
  std::thread pipeline_thread;
  std::thread control_thread;
  std::atomic<bool> stop_control{false};
  int listen_fd = -1;

  std::vector<double> latencies;  // guarded by mu
  uint64_t dets_total = 0;        // guarded by mu
  // ---- M7 rule state (scheduler placeholder): which model is "active" and why.
  // Daemon still runs the single YOLO engine; RULE only flips this metadata so
  // the web UI can visualize the scheduler's decision. Never touches pipeline.
  std::string active_model = "yolo";                    // guarded by mu
  std::string rule_reason = "default: yolo everyday";   // guarded by mu
  double low_conf_thresh = 0.50;  // config (startup only, read without lock)
  uint64_t low_conf_frames = 0;   // guarded by mu: recorded frames with 0 dets or max_conf < thresh
  uint64_t low_conf_streak = 0;   // guarded by mu: consecutive low-conf recorded frames
  double last_max_conf = 0;       // guarded by mu: max det conf of last recorded frame
  std::string worker_state = "idle";   // guarded by mu: idle|owl-working|qwen-working|done:<summary>
  std::string worker_detail;           // guarded by mu: short summary (<=60 chars shown on web)
  uint64_t reviews_written = 0;  // guarded by mu
  FILE* events_fp = nullptr;      // pipeline thread only

  // ---- web snapshot slot (M7a): latest annotated small frame ----
  std::mutex snap_mu;
  std::vector<uint8_t> snap_bgr;  // 480px-wide BGR, boxes drawn
  int snap_w = 0, snap_h = 0;
  uint64_t snap_frame = 0;
  std::vector<Det> snap_dets;  // boxes in snap pixel coords

  bool init_cuda() {
    if (!edgevision::cuda_preprocess::init()) {
      last_error = "cuda_preprocess init failed";
      return false;
    }
    if (cudaStreamCreate(&stream) != cudaSuccess) {
      last_error = "cudaStreamCreate failed";
      return false;
    }
    // Worst case: 1080p BGR staging, exact size decided per frame.
    if (cudaMalloc(&bgr_gpu, 1920 * 1080 * 3) != cudaSuccess) {
      last_error = "cudaMalloc bgr staging failed";
      return false;
    }
    return true;
  }

  bool load_engine() {
    if (!engine.load(engine_path)) {
      last_error = "load engine: " + engine.last_error();
      return false;
    }
    if (!engine.prepare_io()) {
      last_error = "prepare_io: " + engine.last_error();
      return false;
    }
    target_w = static_cast<int>(engine.input_dims().w);
    target_h = static_cast<int>(engine.input_dims().h);
    num_classes = static_cast<int>(engine.output_dims().c) - 4;
    num_anchors = static_cast<int>(engine.output_dims().w);
    return true;
  }

  // ---- M8a review dump: atomic crop.jpg + meta.json for worker escalation ----
  // Called from pipeline thread when low_conf_streak hits exactly 3 (edge trigger,
  // once per streak). Uses full-res decoded BGR (not the 480px snapshot).
  static bool write_file_atomic(const std::string& final_path,
                                const uint8_t* data, std::size_t n) {
    std::string tmp = final_path + ".tmp";
    FILE* fp = std::fopen(tmp.c_str(), "wb");
    if (!fp) return false;
    if (n && std::fwrite(data, 1, n, fp) != n) {
      std::fclose(fp);
      return false;
    }
    std::fclose(fp);
    return std::rename(tmp.c_str(), final_path.c_str()) == 0;
  }

  // BGR -> JPEG file (quality 85 for review crops; snapshot path uses q70 in-mem).
  static bool write_bgr_jpeg(const std::string& path, const uint8_t* bgr, int w, int h) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    FILE* fp = std::fopen((path + ".tmp").c_str(), "wb");
    if (!fp) {
      jpeg_destroy_compress(&cinfo);
      return false;
    }
    jpeg_stdio_dest(&cinfo, fp);
    cinfo.image_width = w;
    cinfo.image_height = h;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_EXT_BGR;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 85, TRUE);
    jpeg_start_compress(&cinfo, TRUE);
    std::vector<uint8_t> row(static_cast<std::size_t>(w) * 3);
    while (cinfo.next_scanline < cinfo.image_height) {
      const uint8_t* src = bgr + static_cast<std::size_t>(cinfo.next_scanline) * w * 3;
      std::memcpy(row.data(), src, row.size());
      JSAMPROW rp = row.data();
      jpeg_write_scanlines(&cinfo, &rp, 1);
    }
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    std::fclose(fp);
    return std::rename((path + ".tmp").c_str(), path.c_str()) == 0;
  }

  static void prune_oldest(const std::string& dir, int keep) {
    // best-effort: keep newest `keep` subdirs, remove the rest
    std::string cmd = "ls -1 '" + dir + "' 2>/dev/null | head -n 1000 | "
                      "while read d; do echo \"$d\"; done | "
                      "sort | head -n -" + std::to_string(keep) +
                      " 2>/dev/null | while read d; do rm -rf '" + dir + "/'\"$d\"; done";
    std::system(cmd.c_str());
  }

  // Returns review stamp on success, "" when skipped/failed.
  std::string maybe_dump_review(uint64_t frame_no, const std::vector<uint8_t>& bgr,
                                int w, int h, const std::vector<Det>& dets,
                                double max_conf, bool empty_streak,
                                uint64_t streak_after) {
    if (streak_after != 3) return "";  // edge trigger only
    std::string stamp = session_stamp() + "_f" + std::to_string(frame_no);
    std::string rdir = review_dir + "/" + stamp;
    std::string cmd = "mkdir -p '" + rdir + "'";
    if (std::system(cmd.c_str()) != 0) return "";
    // crop: best det expanded 20%, else center 50%
    int cx0 = w / 4, cy0 = h / 4, cx1 = w * 3 / 4, cy1 = h * 3 / 4;
    float best = -1;
    const Det* bd = nullptr;
    for (const auto& d : dets) {
      if (d.conf > best) {
        best = d.conf;
        bd = &d;
      }
    }
    if (bd) {
      // model coords are PIXELS in target_w x target_h space (NOT normalized!).
      // Normalize first: preprocess is a direct full-frame stretch (no letterbox).
      // Low-conf boxes may exceed range: clamp + finite check before indexing bgr.
      if (!std::isfinite(bd->x) || !std::isfinite(bd->y) ||
          !std::isfinite(bd->w) || !std::isfinite(bd->h))
        return "";
      float nx = bd->x / target_w, ny = bd->y / target_h;
      float nw = bd->w / target_w, nh = bd->h / target_h;
      int bx0 = static_cast<int>((nx - nw / 2) * w);
      int by0 = static_cast<int>((ny - nh / 2) * h);
      int bx1 = static_cast<int>((nx + nw / 2) * w);
      int by1 = static_cast<int>((ny + nh / 2) * h);
      int pad_x = (bx1 - bx0) / 5, pad_y = (by1 - by0) / 5;
      cx0 = std::max(0, bx0 - pad_x);
      cy0 = std::max(0, by0 - pad_y);
      cx1 = std::min(w, bx1 + pad_x);
      cy1 = std::min(h, by1 + pad_y);
    }
    // hard clamp: degenerate/out-of-range rect => skip dump (no SEGV, no dir)
    cx0 = std::max(0, std::min(w - 1, cx0));
    cy0 = std::max(0, std::min(h - 1, cy0));
    cx1 = std::max(cx0 + 1, std::min(w, cx1));
    cy1 = std::max(cy0 + 1, std::min(h, cy1));
    int cw = cx1 - cx0, ch = cy1 - cy0;
    if (cw <= 0 || ch <= 0 || cw > w || ch > h) return "";
    std::vector<uint8_t> crop(static_cast<std::size_t>(cw) * ch * 3);
    for (int y = 0; y < ch; ++y) {
      int sy = cy0 + y;
      if (sy < 0 || sy >= h) return "";
      const uint8_t* s = &bgr[(static_cast<std::size_t>(sy) * w + cx0) * 3];
      uint8_t* d = &crop[static_cast<std::size_t>(y) * cw * 3];
      std::memcpy(d, s, static_cast<std::size_t>(cw) * 3);
    }
    std::string crop_path = rdir + "/crop.jpg";
    if (!write_bgr_jpeg(crop_path, crop.data(), cw, ch)) return "";
    // meta.json (atomic). Keep schema in sync with docs/m8_integration_plan.md §5.1.
    std::string trig = empty_streak ? "empty streak>=3" : "low_conf streak>=3";
    std::string model = empty_streak ? "yolo+vlm?" : "yolo+nanoowl?";
    std::string meta = "{\"frame_no\":" + std::to_string(frame_no) +
                       ",\"trigger\":\"" + trig + "\",\"active_model\":\"" + model + "\"," +
                       "\"crop\":\"crop.jpg\",\"crop_rect\":[" + std::to_string(cx0) + "," +
                       std::to_string(cy0) + "," + std::to_string(cw) + "," +
                       std::to_string(ch) + "],\"orig_wh\":[" + std::to_string(w) + "," +
                       std::to_string(h) + "],\"last_max_conf\":" + std::to_string(max_conf) +
                       ",\"yolo_boxes\":[";
    bool first = true;
    char bb[128];
    for (const auto& d : dets) {
      if (!first) meta += ",";
      first = false;
      // normalize to [0,1] of the letterboxed 640 frame so workers (OWL/SAM prompts)
      // can map back to any crop size without knowing target_w/target_h.
      std::snprintf(bb, sizeof(bb), "{\"x\":%.4f,\"y\":%.4f,\"w\":%.4f,\"h\":%.4f,"
                                   "\"conf\":%.3f,\"cls\":%d}",
                    d.x / target_w, d.y / target_h, d.w / target_w, d.h / target_h,
                    d.conf, d.cls);
      meta += bb;
    }
    meta += "],\"prompt_owl\":\"unknown object in crop, describe and localize\"," +
            std::string("\"prompt_qwen\":\" describe the cropped object in one sentence, is it abnormal?\"}");
    if (!write_file_atomic(rdir + "/meta.json",
                            reinterpret_cast<const uint8_t*>(meta.data()), meta.size()))
      return "";
    prune_oldest(review_dir, review_keep);
    return stamp;
  }

  // ---- session control (called from control thread) ----
  bool do_start(std::string& err) {
    std::lock_guard<std::mutex> lk(mu);
    if (state != DaemonState::IDLE) {
      err = std::string("already ") + state_str(state);
      return false;
    }
    state = DaemonState::STARTING;
    session_dir = sessions_dir + "/" + session_stamp();
    // mkdir -p sessions/<stamp>
    std::string cmd = "mkdir -p '" + session_dir + "'";
    if (std::system(cmd.c_str()) != 0) {
      state = DaemonState::IDLE;
      err = "mkdir session dir failed";
      return false;
    }
    std::string ev_tmp = session_dir + "/events.jsonl.tmp";
    events_fp = std::fopen(ev_tmp.c_str(), "w");
    if (!events_fp) {
      state = DaemonState::IDLE;
      err = "open events.jsonl.tmp failed";
      return false;
    }
    if (!sess.start(device, 1280, 720, 0x47504A4D /*MJPG*/, 6, 4, 4)) {
      std::fclose(events_fp);
      events_fp = nullptr;
      state = DaemonState::IDLE;
      err = "session start: " + sess.last_error();
      return false;
    }
    // accepted: driver really streaming now
    frames_ok = 0;
    decode_fail = 0;
    infer_fail = 0;
    timeouts = 0;
    dets_total = 0;
    low_conf_frames = 0;
    low_conf_streak = 0;
    last_max_conf = 0;
    active_model = "yolo";
    rule_reason = "default: yolo everyday";
    worker_state = "idle";
    worker_detail.clear();
    reviews_written = 0;
    latencies.clear();
    stop_session = false;
    state = DaemonState::RUNNING;
    pipeline_thread = std::thread(&Daemon::pipeline_loop, this);
    return true;
  }

  bool do_stop(std::string& err) {
    {
      std::lock_guard<std::mutex> lk(mu);
      if (state != DaemonState::RUNNING) {
        err = std::string("not running (state=") + state_str(state) + ")";
        return false;
      }
      state = DaemonState::STOPPING;
    }
    stop_session = true;
    if (pipeline_thread.joinable()) pipeline_thread.join();
    // completed: session stopped AND summary renamed into place
    sess.stop();
    if (events_fp) {
      std::fflush(events_fp);
      std::fclose(events_fp);
      events_fp = nullptr;
    }
    if (!write_summary(err)) {
      std::lock_guard<std::mutex> lk(mu);
      state = DaemonState::IDLE;
      return false;
    }
    {
      std::lock_guard<std::mutex> lk(mu);
      state = DaemonState::IDLE;
    }
    return true;
  }

  bool write_summary(std::string& err) {
    auto st = sess.stats();
    std::vector<double> lat;
    uint64_t dets = 0, ok = 0;
    {
      std::lock_guard<std::mutex> lk(mu);
      lat = latencies;
      dets = dets_total;
      ok = frames_ok;
    }
    std::sort(lat.begin(), lat.end());
    double avg = 0, p50 = 0, p99 = 0, mn = 0, mx = 0;
    if (!lat.empty()) {
      avg = std::accumulate(lat.begin(), lat.end(), 0.0) / lat.size();
      p50 = lat[lat.size() * 50 / 100];
      p99 = lat[lat.size() * 99 / 100];
      mn = lat.front();
      mx = lat.back();
    }
    std::string tmp = session_dir + "/summary.json.tmp";
    std::string fin = session_dir + "/summary.json";
    FILE* fp = std::fopen(tmp.c_str(), "w");
    if (!fp) {
      err = "open summary.json.tmp failed";
      return false;
    }
    std::fprintf(fp,
                 "{\"frames_ok\":%llu,\"decode_fail\":%llu,\"infer_fail\":%llu,"
                 "\"timeouts\":%llu,\"pool_drop\":%llu,\"queue_drop\":%llu,"
                 "\"lat_avg_ms\":%.2f,\"lat_p50_ms\":%.2f,\"lat_p99_ms\":%.2f,"
                 "\"lat_min_ms\":%.2f,\"lat_max_ms\":%.2f,\"dets_total\":%llu}\n",
                 (unsigned long long)ok, (unsigned long long)decode_fail,
                 (unsigned long long)infer_fail, (unsigned long long)timeouts,
                 (unsigned long long)st.frames_dropped_pool.load(),
                 (unsigned long long)st.frames_dropped_queue.load(), avg, p50, p99, mn,
                 mx, (unsigned long long)dets);
    std::fclose(fp);
    if (std::rename(tmp.c_str(), fin.c_str()) != 0) {
      err = "rename summary.json failed";
      return false;
    }
    std::string ev_tmp = session_dir + "/events.jsonl.tmp";
    std::string ev_fin = session_dir + "/events.jsonl";
    if (std::rename(ev_tmp.c_str(), ev_fin.c_str()) != 0) {
      err = "rename events.jsonl failed";
      return false;
    }
    return true;
  }

  void pipeline_loop() {
    std::vector<float> output(static_cast<std::size_t>(num_anchors * (4 + num_classes)));
    uint64_t frame_no = 0;
    while (!stop_session && !g_stop_process) {
      edgevision::FrameLease lease;
      if (!sess.consume(lease, 2000)) {
        std::lock_guard<std::mutex> lk(mu);
        timeouts++;
        continue;
      }
      if (!lease.valid()) continue;
      if (frame_no >= static_cast<uint64_t>(warmup)) {
        // steady-state path with timing; warmup frames still run (sensor settle)
        run_one(lease, output, frame_no, true);
      } else {
        run_one(lease, output, frame_no, false);
      }
      frame_no++;
    }
  }

  void run_one(edgevision::FrameLease& lease, std::vector<float>& output, uint64_t frame_no,
               bool record) {
    std::vector<uint8_t> bgr;
    int dec_w = 0, dec_h = 0;
    std::string dec_err;
    if (!edgevision::mjpeg_decode_to_bgr(lease->data.data(), lease->data.size(), bgr, dec_w,
                                         dec_h, dec_err)) {
      std::lock_guard<std::mutex> lk(mu);
      decode_fail++;
      return;
    }
    std::size_t need = static_cast<std::size_t>(dec_w) * dec_h * 3;
    cudaMemcpy(bgr_gpu, bgr.data(), need, cudaMemcpyHostToDevice);
    auto t0 = std::chrono::steady_clock::now();
    edgevision::cuda_preprocess::preprocess(static_cast<const uint8_t*>(bgr_gpu), dec_w,
                                            dec_h,
                                            static_cast<float*>(engine.input_gpu()), target_w,
                                            target_h, stream);
    if (!engine.infer(stream)) {
      std::lock_guard<std::mutex> lk(mu);
      infer_fail++;
      return;
    }
    cudaStreamSynchronize(stream);
    auto t1 = std::chrono::steady_clock::now();
    double lat_ms =
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1000.0;
    cudaMemcpy(output.data(), engine.output_gpu(), output.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    auto dets = postprocess(output.data(), num_anchors, num_classes, conf_thresh, iou_thresh);
    update_snapshot(bgr, dec_w, dec_h, dets);
    if (!record) return;
    // ---- M7 rule + M8a review dump (metadata + throttled disk I/O) ----
    {
      std::lock_guard<std::mutex> lk(mu);
      frames_ok++;
      dets_total += dets.size();
      latencies.push_back(lat_ms);
      // ---- M7 rule hint (metadata only, no pipeline change) ----
      double mx = 0;
      for (const auto& d : dets) mx = std::max(mx, static_cast<double>(d.conf));
      last_max_conf = mx;
      bool low = (dets.empty() || mx < low_conf_thresh);
      if (low) {
        low_conf_frames++;
        low_conf_streak++;
      } else {
        low_conf_streak = 0;
      }
      // NanoOWL placeholder: streak>=3 low-conf frames => "candidate for 2nd-stage".
      // VLM placeholder:  completely empty frame in a streak => flag for review.
      // Still runs YOLO only; scheduler (off-board) decides real escalation.
      if (low_conf_streak >= 3 && dets.empty()) {
        active_model = "yolo+vlm?";
        rule_reason = "empty streak>=3: needs VLM review (placeholder)";
      } else if (low_conf_streak >= 3) {
        active_model = "yolo+nanoowl?";
        rule_reason = "low-conf streak>=3: candidate for NanoOWL 2nd-stage (placeholder)";
      } else {
        active_model = "yolo";
        rule_reason = "default: yolo everyday";
      }
      // ---- M8a: edge-triggered review dump (streak 0->3 exactly once) ----
      // Dump AFTER rule fields are final; writes crop.jpg + meta.json atomically.
      // JPEG encode may take ~10-30ms: acceptable (once per streak, not per frame).
      std::string review_stamp;
      bool want_dump = (record && low && low_conf_streak == 3);
      uint64_t wall_ns = wall_ns_now();
      if (events_fp) {
        std::fprintf(events_fp, "{\"frame\":%llu,\"ts_wall_ns\":%llu,\"lat_ms\":%.2f,"
                                "\"dets\":%zu,\"max_conf\":%.3f,\"low_conf\":%d,"
                                "\"rule\":\"%s\"}\n",
                     (unsigned long long)frame_no, (unsigned long long)wall_ns, lat_ms,
                     dets.size(), mx, low ? 1 : 0, active_model.c_str());
      }
      if (want_dump) reviews_written++;  // count attempt; result row follows
      uint64_t my_reviews = reviews_written;
      (void)my_reviews;
      if (want_dump) {
        // NOTE: disk I/O intentionally under mu for M8a (simple + race-free review
        // counting). JPEG encode ~10-30ms once per streak, not per frame; socket/HTTP
        // threads take mu only briefly, no deadlock (no lock ordering inversion:
        // this is the only place pipeline takes mu across I/O).
        review_stamp = maybe_dump_review(frame_no, bgr, dec_w, dec_h, dets, mx,
                                         dets.empty(), 3);
        if (!review_stamp.empty()) {
          if (events_fp) {
            std::fprintf(events_fp, "{\"frame\":%llu,\"ts_wall_ns\":%llu,"
                                    "\"escalated\":true,\"review\":\"%s\"}\n",
                         (unsigned long long)frame_no, (unsigned long long)wall_ns_now(),
                         review_stamp.c_str());
          }
        } else {
          // dump failed: keep streak/stats, just no review dir this time
          reviews_written--;
        }
      }
    }
  }

  // ---- web snapshot: downscale to 480px wide, draw boxes, store ----
  void update_snapshot(const std::vector<uint8_t>& bgr, int w, int h,
                       const std::vector<Det>& dets) {
    if (http_port <= 0 || w <= 0 || h <= 0) return;
    const int dst_w = 480;
    const int dst_h = std::max(1, h * dst_w / w);
    std::vector<uint8_t> small(static_cast<std::size_t>(dst_w) * dst_h * 3);
    // nearest-neighbor downscale (cheap; snapshot only, not inference path)
    for (int y = 0; y < dst_h; ++y) {
      int sy = y * h / dst_h;
      for (int x = 0; x < dst_w; ++x) {
        int sx = x * w / dst_w;
        const uint8_t* s = &bgr[(static_cast<std::size_t>(sy) * w + sx) * 3];
        uint8_t* d = &small[(static_cast<std::size_t>(y) * dst_w + x) * 3];
        d[0] = s[0];
        d[1] = s[1];
        d[2] = s[2];
      }
    }
    float sx = static_cast<float>(dst_w) / target_w;
    float sy = static_cast<float>(dst_h) / target_h;
    std::vector<Det> px;
    px.reserve(dets.size());
    for (const auto& d : dets) {
      // model coords are normalized cx,cy,w,h in [0,1]
      float cx = d.x * dst_w, cy = d.y * dst_h;
      float bw = d.w * dst_w, bh = d.h * dst_h;
      int x0 = std::max(0, static_cast<int>(cx - bw / 2));
      int y0 = std::max(0, static_cast<int>(cy - bh / 2));
      int x1 = std::min(dst_w - 1, static_cast<int>(cx + bw / 2));
      int y1 = std::min(dst_h - 1, static_cast<int>(cy + bh / 2));
      // green 2px box (BGR: 0,255,0)
      for (int x = x0; x <= x1; ++x)
        for (int t = 0; t < 2; ++t) {
          if (y0 + t < dst_h) {
            uint8_t* p = &small[(static_cast<std::size_t>(y0 + t) * dst_w + x) * 3];
            p[0] = 0;
            p[1] = 255;
            p[2] = 0;
          }
          if (y1 - t >= 0) {
            uint8_t* p = &small[(static_cast<std::size_t>(y1 - t) * dst_w + x) * 3];
            p[0] = 0;
            p[1] = 255;
            p[2] = 0;
          }
        }
      for (int y = y0; y <= y1; ++y)
        for (int t = 0; t < 2; ++t) {
          if (x0 + t < dst_w) {
            uint8_t* p = &small[(static_cast<std::size_t>(y) * dst_w + x0 + t) * 3];
            p[0] = 0;
            p[1] = 255;
            p[2] = 0;
          }
          if (x1 - t >= 0) {
            uint8_t* p = &small[(static_cast<std::size_t>(y) * dst_w + x1 - t) * 3];
            p[0] = 0;
            p[1] = 255;
            p[2] = 0;
          }
        }
      px.push_back({static_cast<float>(x0), static_cast<float>(y0),
                    static_cast<float>(x1 - x0), static_cast<float>(y1 - y0), d.conf,
                    d.cls});
      (void)sx;
      (void)sy;
    }
    std::lock_guard<std::mutex> lk(snap_mu);
    snap_bgr.swap(small);
    snap_w = dst_w;
    snap_h = dst_h;
    snap_frame++;
    snap_dets.swap(px);
  }

  // BGR -> JPEG (libjpeg-turbo, quality 70). Returns false on error.
  static bool encode_jpeg(const uint8_t* bgr, int w, int h, int quality,
                          std::vector<uint8_t>& out) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    unsigned char* mem = nullptr;
    unsigned long mem_size = 0;
    jpeg_mem_dest(&cinfo, &mem, &mem_size);
    cinfo.image_width = w;
    cinfo.image_height = h;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_EXT_BGR;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);
    jpeg_start_compress(&cinfo, TRUE);
    std::vector<uint8_t> row(static_cast<std::size_t>(w) * 3);
    while (cinfo.next_scanline < cinfo.image_height) {
      const uint8_t* src = bgr + static_cast<std::size_t>(cinfo.next_scanline) * w * 3;
      // JCS_EXT_BGR consumes BGR directly; still copy per row for safety.
      std::memcpy(row.data(), src, row.size());
      JSAMPROW rp = row.data();
      jpeg_write_scanlines(&cinfo, &rp, 1);
    }
    jpeg_finish_compress(&cinfo);
    out.assign(mem, mem + mem_size);
    jpeg_destroy_compress(&cinfo);
    if (mem) free(mem);
    return !out.empty();
  }

  // ---- control socket ----
  std::string handle_cmd(const std::string& line) {
    if (line == "STATUS") {
      std::lock_guard<std::mutex> lk(mu);
      auto st = sess.stats();
      char buf[2048];
      std::snprintf(buf, sizeof(buf),
                    "{\"ok\":true,\"state\":\"%s\",\"frames\":%llu,\"decode_fail\":%llu,"
                    "\"infer_fail\":%llu,\"timeouts\":%llu,\"pool_drop\":%llu,"
                    "\"queue_drop\":%llu,\"session\":\"%s\","
                    "\"active_model\":\"%s\",\"rule_reason\":\"%s\","
                    "\"low_conf_frames\":%llu,\"low_conf_streak\":%llu,"
                    "\"last_max_conf\":%.3f,"
                    "\"reviews_written\":%llu,\"worker\":\"%s\"}",
                    state_str(state), (unsigned long long)frames_ok,
                    (unsigned long long)decode_fail, (unsigned long long)infer_fail,
                    (unsigned long long)timeouts,
                    (unsigned long long)st.frames_dropped_pool.load(),
                    (unsigned long long)st.frames_dropped_queue.load(), session_dir.c_str(),
                    active_model.c_str(), rule_reason.c_str(),
                    (unsigned long long)low_conf_frames,
                    (unsigned long long)low_conf_streak, last_max_conf,
                    (unsigned long long)reviews_written, worker_state.c_str());
      return std::string(buf);
    }
    if (line == "RULE") {
      std::lock_guard<std::mutex> lk(mu);
      char buf[1024];
      std::snprintf(buf, sizeof(buf),
                    "{\"ok\":true,\"active_model\":\"%s\",\"reason\":\"%s\","
                    "\"low_conf_thresh\":%.2f,\"low_conf_frames\":%llu,"
                    "\"low_conf_streak\":%llu,\"last_max_conf\":%.3f,"
                    "\"reviews_written\":%llu,\"review_dir\":\"%s\"}",
                    active_model.c_str(), rule_reason.c_str(), low_conf_thresh,
                    (unsigned long long)low_conf_frames,
                    (unsigned long long)low_conf_streak, last_max_conf,
                    (unsigned long long)reviews_written, review_dir.c_str());
      return std::string(buf);
    }
    if (line.rfind("WORKER ", 0) == 0) {
      // scheduler reports worker progress: WORKER <state> [detail<=120 chars]
      // e.g. WORKER owl-working | WORKER done:cup 0.91 mask ok
      std::string rest = line.substr(7);
      std::lock_guard<std::mutex> lk(mu);
      if (rest.size() > 160) rest.resize(160);
      worker_state = rest;
      worker_detail = rest;
      return "{\"ok\":true}";
    }
    if (line == "WORKER") {
      std::lock_guard<std::mutex> lk(mu);
      return "{\"ok\":true,\"worker\":\"" + worker_state + "\",\"reviews\":" +
             std::to_string(reviews_written) + "}";
    }
    if (line == "START") {
      std::string err;
      if (do_start(err)) {
        std::lock_guard<std::mutex> lk(mu);
        char buf[256];
        std::snprintf(buf, sizeof(buf), "{\"ok\":true,\"accepted\":true,\"session\":\"%s\"}",
                      session_dir.c_str());
        return std::string(buf);
      }
      return "{\"ok\":false,\"error\":\"" + err + "\"}";
    }
    if (line == "STOP") {
      std::string err;
      if (do_stop(err)) return "{\"ok\":true,\"completed\":true}";
      return "{\"ok\":false,\"error\":\"" + err + "\"}";
    }
    return "{\"ok\":false,\"error\":\"unknown cmd (STATUS|RULE|WORKER|START|STOP|QUIT)\"}";
  }

  void control_loop() {
    ::unlink(sock_path.c_str());
    listen_fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd < 0) {
      std::fprintf(stderr, "[edgevisiond] FAIL socket: %s\n", std::strerror(errno));
      return;
    }
    struct sockaddr_un addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    std::strncpy(addr.sun_path, sock_path.c_str(), sizeof(addr.sun_path) - 1);
    if (::bind(listen_fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0 ||
        ::listen(listen_fd, 4) != 0) {
      std::fprintf(stderr, "[edgevisiond] FAIL bind %s: %s\n", sock_path.c_str(),
                   std::strerror(errno));
      return;
    }
    // non-blocking accept with 1s poll so heartbeat + shutdown stay responsive
    int flags = ::fcntl(listen_fd, F_GETFL, 0);
    ::fcntl(listen_fd, F_SETFL, flags | O_NONBLOCK);
    std::printf("[edgevisiond] control listening on %s\n", sock_path.c_str());
    std::fflush(stdout);
    uint64_t tick = 0;
    while (!stop_control && !g_stop_process) {
      int cfd = ::accept(listen_fd, nullptr, nullptr);
      if (cfd < 0) {
        // idle: heartbeat every ~5s for journald
        if (++tick % 5 == 0) {
          std::lock_guard<std::mutex> lk(mu);
          std::printf("[edgevisiond] heartbeat state=%s frames=%llu session=%s\n",
                      state_str(state), (unsigned long long)frames_ok,
                      session_dir.empty() ? "-" : session_dir.c_str());
          std::fflush(stdout);
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
        continue;
      }
      char buf[256] = {0};
      ssize_t n = ::recv(cfd, buf, sizeof(buf) - 1, 0);
      std::string line(n > 0 ? buf : "");
      line.erase(line.find_last_not_of(" \r\n\t") + 1);
      std::string resp;
      if (line == "QUIT") {
        resp = "{\"ok\":true,\"bye\":true}";
        resp += "\n";
        ::send(cfd, resp.c_str(), resp.size(), 0);
        ::close(cfd);
        g_stop_process = 1;
        break;
      }
      resp = handle_cmd(line) + "\n";
      ::send(cfd, resp.c_str(), resp.size(), 0);
      ::close(cfd);
    }
  }

  std::thread http_thread;
  std::atomic<bool> stop_http{false};

  static const char* kIndexHtml;  // defined after struct

  void http_loop() {
    if (http_port <= 0) return;
    int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
      std::fprintf(stderr, "[edgevisiond] FAIL http socket: %s\n", std::strerror(errno));
      return;
    }
    int one = 1;
    ::setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<uint16_t>(http_port));
    if (::bind(fd, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) != 0 ||
        ::listen(fd, 4) != 0) {
      std::fprintf(stderr, "[edgevisiond] FAIL http bind :%d: %s\n", http_port,
                   std::strerror(errno));
      ::close(fd);
      return;
    }
    int flags = ::fcntl(fd, F_GETFL, 0);
    ::fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    std::printf("[edgevisiond] web UI on :%d\n", http_port);
    std::fflush(stdout);
    while (!stop_http && !g_stop_process) {
      int cfd = ::accept(fd, nullptr, nullptr);
      if (cfd < 0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        continue;
      }
      handle_http(cfd);
      ::close(cfd);
    }
    ::close(fd);
  }

  static void send_all(int cfd, const void* data, std::size_t n) {
    const char* p = static_cast<const char*>(data);
    while (n > 0) {
      ssize_t w = ::send(cfd, p, n, 0);
      if (w <= 0) break;
      p += w;
      n -= w;
    }
  }

  static void send_resp(int cfd, const char* ctype, const void* body, std::size_t n) {
    char head[256];
    int hl = std::snprintf(head, sizeof(head),
                           "HTTP/1.0 200 OK\r\nContent-Type: %s\r\n"
                           "Content-Length: %zu\r\nCache-Control: no-cache\r\n"
                           "Connection: close\r\n\r\n",
                           ctype, n);
    send_all(cfd, head, hl);
    if (n) send_all(cfd, body, n);
  }

  void handle_http(int cfd) {
    char req[1024] = {0};
    ::recv(cfd, req, sizeof(req) - 1, 0);
    std::string line(req);
    line = line.substr(0, line.find("\r\n"));
    auto snapshot = [&]() -> std::vector<uint8_t> {
      std::vector<uint8_t> bgr;
      int w = 0, h = 0;
      {
        std::lock_guard<std::mutex> lk(snap_mu);
        bgr = snap_bgr;
        w = snap_w;
        h = snap_h;
      }
      std::vector<uint8_t> jpg;
      if (!bgr.empty()) encode_jpeg(bgr.data(), w, h, 70, jpg);
      return jpg;
    };
    if (line.rfind("GET /snapshot.jpg", 0) == 0) {
      auto jpg = snapshot();
      if (jpg.empty()) {
        const char* nf = "no frame yet";
        char head[128];
        int hl = std::snprintf(head, sizeof(head),
                               "HTTP/1.0 503 Busy\r\nContent-Length: %zu\r\n\r\n",
                               std::strlen(nf));
        send_all(cfd, head, hl);
        send_all(cfd, nf, std::strlen(nf));
        return;
      }
      send_resp(cfd, "image/jpeg", jpg.data(), jpg.size());
    } else if (line.rfind("GET /api/status", 0) == 0) {
      auto st = handle_cmd("STATUS");
      send_resp(cfd, "application/json", st.data(), st.size());
    } else if (line.rfind("GET /api/rule", 0) == 0) {
      auto r = handle_cmd("RULE");
      send_resp(cfd, "application/json", r.data(), r.size());
    } else if (line.rfind("GET /api/worker", 0) == 0) {
      auto r = handle_cmd("WORKER");
      send_resp(cfd, "application/json", r.data(), r.size());
    } else if (line.rfind("GET /api/dets", 0) == 0) {
      std::lock_guard<std::mutex> lk(snap_mu);
      std::string j = "{\"frame\":" + std::to_string(snap_frame) + ",\"dets\":[";
      for (std::size_t i = 0; i < snap_dets.size(); ++i) {
        if (i) j += ",";
        char b[160];
        std::snprintf(b, sizeof(b), "{\"x\":%.0f,\"y\":%.0f,\"w\":%.0f,\"h\":%.0f,"
                                    "\"conf\":%.2f,\"cls\":%d}",
                      snap_dets[i].x, snap_dets[i].y, snap_dets[i].w, snap_dets[i].h,
                      snap_dets[i].conf, snap_dets[i].cls);
        j += b;
      }
      j += "]}";
      send_resp(cfd, "application/json", j.data(), j.size());
    } else if (line.rfind("GET /api/start", 0) == 0) {
      auto r = handle_cmd("START");
      send_resp(cfd, "application/json", r.data(), r.size());
    } else if (line.rfind("GET /api/stop", 0) == 0) {
      auto r = handle_cmd("STOP");
      send_resp(cfd, "application/json", r.data(), r.size());
    } else {
      send_resp(cfd, "text/html", kIndexHtml, std::strlen(kIndexHtml));
    }
  }

  int run() {
    std::signal(SIGTERM, on_signal);
    std::signal(SIGINT, on_signal);
    if (!init_cuda()) {
      std::fprintf(stderr, "[edgevisiond] FAIL %s\n", last_error.c_str());
      return 1;
    }
    if (!load_engine()) {
      std::fprintf(stderr, "[edgevisiond] FAIL %s\n", last_error.c_str());
      return 1;
    }
    std::printf("[edgevisiond] engine %s input %dx%d classes=%d anchors=%d sock=%s\n",
                engine_path.c_str(), target_w, target_h, num_classes, num_anchors,
                sock_path.c_str());
    std::fflush(stdout);
    control_thread = std::thread(&Daemon::control_loop, this);
    http_thread = std::thread(&Daemon::http_loop, this);
    while (!g_stop_process) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    // graceful: finish an open session before exiting
    bool need_stop = false;
    {
      std::lock_guard<std::mutex> lk(mu);
      need_stop = (state == DaemonState::RUNNING);
    }
    if (need_stop) {
      std::string err;
      do_stop(err);
      if (!err.empty()) std::fprintf(stderr, "[edgevisiond] stop-at-exit: %s\n", err.c_str());
    }
    stop_control = true;
    stop_http = true;
    if (control_thread.joinable()) control_thread.join();
    if (http_thread.joinable()) http_thread.join();
    if (listen_fd >= 0) {
      ::close(listen_fd);
      ::unlink(sock_path.c_str());
    }
    if (bgr_gpu) cudaFree(bgr_gpu);
    if (stream) cudaStreamDestroy(stream);
    std::printf("[edgevisiond] exit\n");
    return 0;
  }
};

}  // namespace

const char* Daemon::kIndexHtml =
    "<!doctype html><html><head><meta charset=utf-8>"
    "<meta name=viewport content='width=device-width,initial-scale=1'>"
    "<title>EdgeVisionRT</title>"
    "<style>body{font-family:monospace;background:#111;color:#0f0;margin:0;padding:12px}"
    "#bar{display:flex;gap:8px;align-items:center;margin-bottom:8px;flex-wrap:wrap}"
    "button{background:#222;color:#0f0;border:1px solid #0f0;padding:6px 14px;cursor:pointer}"
    "#wrap{position:relative;display:inline-block}"
    "#view{max-width:96vw;border:1px solid #0f0}"
    "#ov{position:absolute;left:0;top:0;pointer-events:none}"
    "#rule{background:#221;color:#fc0;border:1px solid #fc0;padding:4px 8px;margin-bottom:8px}"
    "#worker{background:#112;color:#8cf;border:1px solid #8cf;padding:4px 8px;margin-bottom:8px}"
    "#st{white-space:pre-wrap;font-size:12px;margin-top:8px}</style></head><body>"
    "<div id=bar><b>EdgeVisionRT [M8a]</b>"
    "<button onclick=\"api('/api/start')\">START</button>"
    "<button onclick=\"api('/api/stop')\">STOP</button>"
    "<span id=lat></span></div>"
    "<div id=rule>rule: loading…</div>"
    "<div id=worker>worker: idle</div>"
    "<div id=wrap><img id=view src='/snapshot.jpg'>"
    "<canvas id=ov></canvas></div><div id=st>loading…</div>"
    "<script>"
    "const img=document.getElementById('view'),ov=document.getElementById('ov'),"
    "st=document.getElementById('st'),lat=document.getElementById('lat'),"
    "rule=document.getElementById('rule'),wk=document.getElementById('worker');"
    "function api(p){fetch(p).then(r=>r.text()).then(t=>st.textContent=t);}"
    "function tick(){"
    "img.src='/snapshot.jpg?t='+Date.now();"
    "fetch('/api/dets').then(r=>r.json()).then(j=>{"
    "const r=img.getBoundingClientRect();ov.width=r.width;ov.height=r.height;"
    "ov.style.width=r.width+'px';ov.style.height=r.height+'px';"
    "const c=ov.getContext('2d');c.clearRect(0,0,ov.width,ov.height);"
    "const kx=ov.width/img.naturalWidth,ky=ov.height/img.naturalHeight;"
    "c.strokeStyle='#0f0';c.lineWidth=2;c.font='12px monospace';c.fillStyle='#0f0';"
    "(j.dets||[]).forEach(d=>{c.strokeRect(d.x*kx,d.y*ky,d.w*kx,d.h*ky);"
    "c.fillText((d.cls??'?')+':'+(+d.conf).toFixed(2),d.x*kx,(d.y*ky-4>10?d.y*ky-4:d.y*ky+14));});"
    "lat.textContent='frame '+j.frame+' dets '+(j.dets||[]).length;});"
    "fetch('/api/status').then(r=>r.json()).then(s=>{"
    "st.textContent=JSON.stringify(s);}).catch(()=>{});"
    "fetch('/api/rule').then(r=>r.json()).then(g=>{"
    "rule.textContent='model='+g.active_model+' | '+g.reason+"
    "' | low_conf '+g.low_conf_streak+' streak / '+g.low_conf_frames+' frames'+"
    "' | max_conf '+(+g.last_max_conf).toFixed(2)"
    "+' | reviews '+g.reviews_written;"
    "rule.style.borderColor=(g.active_model==='yolo')?'#0f0':'#f80';}).catch(()=>{});"
    "fetch('/api/worker').then(r=>r.json()).then(w=>{"
    "wk.textContent='worker='+w.worker+' | reviews '+w.reviews;}).catch(()=>{});}"
    "setInterval(tick,1000);tick();</script></body></html>";

int main(int argc, char** argv) {
  Daemon d;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--engine") == 0 && i + 1 < argc) d.engine_path = argv[++i];
    else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) d.device = argv[++i];
    else if (std::strcmp(argv[i], "--sock") == 0 && i + 1 < argc) d.sock_path = argv[++i];
    else if (std::strcmp(argv[i], "--sessions") == 0 && i + 1 < argc) d.sessions_dir = argv[++i];
    else if (std::strcmp(argv[i], "--conf") == 0 && i + 1 < argc)
      d.conf_thresh = std::atof(argv[++i]);
    else if (std::strcmp(argv[i], "--iou") == 0 && i + 1 < argc)
      d.iou_thresh = std::atof(argv[++i]);
    else if (std::strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
      d.warmup = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--low-conf") == 0 && i + 1 < argc)
      d.low_conf_thresh = std::atof(argv[++i]);
    else if (std::strcmp(argv[i], "--review-dir") == 0 && i + 1 < argc)
      d.review_dir = argv[++i];
    else if (std::strcmp(argv[i], "--review-keep") == 0 && i + 1 < argc)
      d.review_keep = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--http") == 0 && i + 1 < argc)
      d.http_port = std::atoi(argv[++i]);
  }
  std::printf("[edgevisiond] device=%s sessions=%s\n", d.device.c_str(),
              d.sessions_dir.c_str());
  return d.run();
}

#else
#include <cstdio>
int main() {
  std::printf("[edgevisiond] SKIP (requires Linux + CUDA)\n");
  return 0;
}
#endif
