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
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/un.h>
#include <unistd.h>

extern "C" {
#include <jpeglib.h>
}

#include <cuda_runtime.h>

#include "edgevision/cuda_preprocess.h"
#include "edgevision/mjpeg_decoder.h"
#include "edgevision/rtsp_server.h"
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

uint64_t mono_ns_now() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + ts.tv_nsec;
}

// ---- console helpers: all best-effort, never throw, never block ----
static std::string read_file_trim(const char* path, std::size_t max_n = 512) {
  std::ifstream f(path);
  if (!f) return "";
  std::string s;
  std::getline(f, s);
  if (s.size() > max_n) s.resize(max_n);
  while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == '\0'))
    s.pop_back();
  return s;
}

static double percentile_sorted(const std::vector<double>& sorted, double q) {
  if (sorted.empty()) return 0;
  std::size_t i = static_cast<std::size_t>(q * (sorted.size() - 1));
  if (i >= sorted.size()) i = sorted.size() - 1;
  return sorted[i];
}

// CPU%: delta of /proc/stat idle vs total, cached across calls.
// Thread-safe (own mutex): may be called from pipeline + control/HTTP threads.
// Never called under Daemon::mu (does file I/O).
struct CpuMeter {
  uint64_t last_idle = 0, last_total = 0;
  double last_pct = -1;
  std::mutex cmu;
  double sample() {
    std::lock_guard<std::mutex> lk(cmu);
    std::ifstream f("/proc/stat");
    if (!f) return -1;
    std::string tag;
    uint64_t u, n, s, idle, iow, irq, sir, st;
    f >> tag >> u >> n >> s >> idle >> iow >> irq >> sir >> st;
    if (tag != "cpu") return -1;
    uint64_t idle_all = idle + iow;
    uint64_t total = u + n + s + idle + iow + irq + sir + st;
    double out = -1;
    if (last_total && total > last_total) {
      uint64_t dt = total - last_total, di = idle_all - last_idle;
      out = dt ? (100.0 * (dt - di) / dt) : -1;
      last_pct = out;
    } else if (last_pct >= 0) {
      out = last_pct;  // same-tick re-read: return cached, never fake 0
    }
    last_idle = idle_all;
    last_total = total;
    return out;
  }
};

// Jetson GPU load: /sys/devices/gpu.0/load is per-mille (0..1000). -1 = unknown
// (Orin Nano has no NVENC but the GPU load node still exists; never fake 0).
static double read_gpu_load() {
  std::string s = read_file_trim("/sys/devices/gpu.0/load");
  if (s.empty()) return -1;
  try {
    return std::stod(s) / 10.0;
  } catch (...) {
    return -1;
  }
}

struct MemInfo {
  long total_mb = -1, avail_mb = -1;
};
static MemInfo read_mem() {
  MemInfo m;
  std::ifstream f("/proc/meminfo");
  std::string k;
  long v;
  std::string unit;
  long tot_kb = -1, av_kb = -1;
  while (f >> k >> v >> unit) {
    if (k == "MemTotal:") tot_kb = v;
    if (k == "MemAvailable:") av_kb = v;
  }
  if (tot_kb > 0) m.total_mb = tot_kb / 1024;
  if (av_kb >= 0) m.avail_mb = av_kb / 1024;
  return m;
}

// Hottest thermal zone in °C. -1000 = unknown (sensor missing, never fake a temp).
static double read_temp_c() {
  double best = -1000;
  for (int i = 0; i < 16; ++i) {
    char p[128];
    std::snprintf(p, sizeof(p), "/sys/class/thermal/thermal_zone%d/temp", i);
    std::string s = read_file_trim(p);
    if (s.empty()) continue;
    try {
      double c = std::stod(s) / 1000.0;
      if (c > best) best = c;
    } catch (...) {
    }
  }
  return best;
}

static std::string json_escape(const std::string& s) {
  std::string o;
  o.reserve(s.size() + 8);
  for (char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if ((unsigned char)c < 0x20) {
          char b[8];
          std::snprintf(b, sizeof(b), "\\u%04x", c);
          o += b;
        } else {
          o += c;
        }
    }
  }
  return o;
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
  uint64_t snap_src_id = 0;    // pipeline frame_no that produced this snapshot
  uint64_t snap_epoch = 0;     // stream epoch (reset on camera reconnect)
  uint64_t snap_ts_monotonic_ns = 0;  // capture time of source frame
  int snap_src_w = 0, snap_src_h = 0;  // source decoded size before 480px downscale

  // ---- console metrics (bounded, never unbounded) ----
  uint64_t boot_ns_monotonic = 0;  // process start (CLOCK_MONOTONIC)
  std::string version = "edgevisiond m11-console";
  std::string board_model;   // lazy-filled from /proc/device-tree/model
  std::string l4t_version;   // lazy-filled from /etc/nv_tegra_release
  // per-stage timings (ms), guarded by mu, capped at 600 samples
  std::vector<double> lat_decode, lat_infer, lat_post;
  // 1s-bucket trend for the last 120s: {t_wall_s, fps, infer_ms_p50, cpu, gpu, mem_mb}
  struct TrendPt { uint64_t t = 0; double fps = 0, infer = 0, cpu = 0, gpu = 0; long mem = 0; };
  std::deque<TrendPt> trend;  // guarded by mu, cap 120
  uint64_t trend_bucket = 0;  // wall seconds of open bucket
  uint64_t trend_frames = 0;  // frames in open bucket
  std::vector<double> trend_lat;  // infer samples in open bucket
  // bounded event log (cap 200), guarded by mu
  std::deque<std::string> events;  // JSON strings, newest at back
  uint64_t start_count = 0, stop_count = 0, recover_count = 0;
  std::string last_fault;  // last error text for the console fault banner
  CpuMeter cpu_meter;       // own lock, NEVER under mu (does file I/O)
  // pipeline-thread-only sys sample cache (no lock; flushed into trend under mu)
  uint64_t last_sys_sample_mono = 0;
  double plc_cpu = -1, plc_gpu = -1, plc_temp = -1000;
  long plc_mem_used = -1, plc_mem_total = -1, plc_mem_avail = -1;

  // mu must be held. Bounded: cap 200, drop oldest. Never throws.
  void push_event_locked(const char* type, const std::string& detail) {
    char b[320];
    std::snprintf(b, sizeof(b), "{\"t\":%llu,\"type\":\"%s\",\"detail\":\"%s\"}",
                  (unsigned long long)(wall_ns_now() / 1000000000ull), type,
                  json_escape(detail).c_str());
    events.emplace_back(b);
    while (events.size() > 200) events.pop_front();
  }
  // mu must be held. All latency vectors capped at 600 (oldest dropped).
  static void push_capped(std::vector<double>& v, double x) {
    v.push_back(x);
    if (v.size() > 600) v.erase(v.begin(), v.begin() + (v.size() - 600));
  }
  // mu must be held. 1s-bucket trend accumulator; caller passes infer ms +
  // pre-sampled system values (sampled OUTSIDE mu in run_one, ~1Hz cached).
  // Never does file I/O under lock.
  void trend_add_locked(uint64_t wall_s, double infer_ms, double cpu, double gpu,
                        long mem_used_mb) {
    if (trend_bucket == 0) trend_bucket = wall_s;
    if (wall_s != trend_bucket) {
      // close bucket: emit one TrendPt (fps = frames in bucket, infer p50)
      double p50 = 0;
      if (!trend_lat.empty()) {
        std::sort(trend_lat.begin(), trend_lat.end());
        p50 = percentile_sorted(trend_lat, 0.5);
      }
      TrendPt pt;
      pt.t = trend_bucket;
      pt.fps = static_cast<double>(trend_frames);
      pt.infer = p50;
      pt.cpu = cpu;
      pt.gpu = gpu;
      pt.mem = mem_used_mb;
      trend.push_back(pt);
      while (trend.size() > 120) trend.pop_front();
      trend_bucket = wall_s;
      trend_frames = 0;
      trend_lat.clear();
    }
    trend_frames++;
    trend_lat.push_back(infer_ms);
  }

  // ---- M10 RTSP live monitor: same annotated small frame -> H.264 ----
  edgevision::RtspServer rtsp;
  int rtsp_port = 8554;               // 0 = disable RTSP
  std::string rtsp_encoder = "x264";  // x264 (SW) or nvv4l2h264 (Jetson HW)

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
    // best-effort: keep newest `keep` subdirs, remove the rest.
    // No shell: explicit readdir + sort + remove (no injection, no fork).
    try {
      std::vector<std::string> names;
      for (const auto& e : std::filesystem::directory_iterator(dir)) {
        if (e.is_directory()) names.push_back(e.path().filename().string());
      }
      std::sort(names.begin(), names.end());
      if (static_cast<int>(names.size()) <= keep) return;
      for (std::size_t i = 0; i + static_cast<std::size_t>(keep) < names.size(); ++i) {
        std::error_code ec;
        std::filesystem::remove_all(std::filesystem::path(dir) / names[i], ec);
      }
    } catch (...) {
    }
  }

  // Returns review stamp on success, "" when skipped/failed.
  std::string maybe_dump_review(uint64_t frame_no, const std::vector<uint8_t>& bgr,
                                int w, int h, const std::vector<Det>& dets,
                                double max_conf, bool empty_streak,
                                uint64_t streak_after) {
    if (streak_after != 3) return "";  // edge trigger only
    std::string stamp = session_stamp() + "_f" + std::to_string(frame_no);
    std::string rdir = review_dir + "/" + stamp;
    {
      std::error_code ec;
      std::filesystem::create_directories(rdir, ec);
      if (ec) return "";
    }
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
  // Lock discipline: mu guards state/counters only. Blocking I/O (mkdir,
  // fopen, sess.start, join) NEVER runs under mu, so STATUS/HTTP stay live.
  bool do_start(std::string& err) {
    std::string new_dir;
    {
      std::lock_guard<std::mutex> lk(mu);
      if (state != DaemonState::IDLE) {
        err = std::string("already ") + state_str(state);
        return false;
      }
      state = DaemonState::STARTING;
      new_dir = sessions_dir + "/" + session_stamp();
      session_dir = new_dir;  // reserve name so 2nd START sees STARTING
    }
    // create sessions/<stamp> (outside lock: never hold mu across I/O,
    // no shell: std::filesystem only, no injection via session_stamp)
    {
      std::error_code ec;
      std::filesystem::create_directories(new_dir, ec);
      if (ec) {
        std::lock_guard<std::mutex> lk(mu);
        state = DaemonState::IDLE;
        session_dir.clear();
        err = "mkdir session dir failed: " + new_dir + " (" + ec.message() + ")";
        return false;
      }
    }
    std::string ev_tmp = new_dir + "/events.jsonl.tmp";
    FILE* fp = std::fopen(ev_tmp.c_str(), "w");
    if (!fp) {
      std::lock_guard<std::mutex> lk(mu);
      state = DaemonState::IDLE;
      session_dir.clear();
      err = "open events.jsonl.tmp failed: " + ev_tmp + " (" +
            std::strerror(errno) + ")";
      return false;
    }
    if (!sess.start(device, 1280, 720, 0x47504A4D /*MJPG*/, 6, 4, 4)) {
      std::fclose(fp);
      std::lock_guard<std::mutex> lk(mu);
      state = DaemonState::IDLE;
      err = "session start: " + sess.last_error();
      session_dir.clear();
      return false;
    }
    // accepted: driver really streaming now
    // Defensive: STARTING can only be left behind if a second control executor
    // is added later (today control is single-threaded). On abort, stop the
    // already-streaming session + close fp outside mu (no join/IO under lock).
    bool aborted = false;
    {
      std::lock_guard<std::mutex> lk(mu);
      if (state != DaemonState::STARTING) {
        err = std::string("aborted (state=") + state_str(state) + ")";
        session_dir.clear();
        aborted = true;
      }
    }
    if (aborted) {
      sess.stop();
      std::fclose(fp);
      return false;
    }
    {
      std::lock_guard<std::mutex> lk(mu);
      events_fp = fp;
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
      lat_decode.clear();
      lat_infer.clear();
      lat_post.clear();
      trend.clear();
      trend_bucket = 0;
      trend_frames = 0;
      trend_lat.clear();
      start_count++;
      push_event_locked("start", new_dir);
      stop_session = false;
      state = DaemonState::RUNNING;
    }
    pipeline_thread = std::thread(&Daemon::pipeline_loop, this);
    return true;
  }

  bool do_stop(std::string& err) {
    std::string dir;
    {
      std::lock_guard<std::mutex> lk(mu);
      if (state != DaemonState::RUNNING) {
        err = std::string("not running (state=") + state_str(state) + ")";
        return false;
      }
      state = DaemonState::STOPPING;
      dir = session_dir;
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
    if (!write_summary(dir, err)) {
      std::lock_guard<std::mutex> lk(mu);
      state = DaemonState::IDLE;
      last_fault = err;
      push_event_locked("fault", err);
      return false;
    }
    {
      std::lock_guard<std::mutex> lk(mu);
      state = DaemonState::IDLE;
      stop_count++;
      push_event_locked("stop", dir);
    }
    return true;
  }

  bool write_summary(const std::string& dir, std::string& err) {
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
    std::string tmp = dir + "/summary.json.tmp";
    std::string fin = dir + "/summary.json";
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
    std::string ev_tmp = dir + "/events.jsonl.tmp";
    std::string ev_fin = dir + "/events.jsonl";
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
    auto t_dec0 = std::chrono::steady_clock::now();
    std::vector<uint8_t> bgr;
    int dec_w = 0, dec_h = 0;
    std::string dec_err;
    if (!edgevision::mjpeg_decode_to_bgr(lease->data.data(), lease->data.size(), bgr, dec_w,
                                         dec_h, dec_err)) {
      std::lock_guard<std::mutex> lk(mu);
      decode_fail++;
      last_fault = "mjpeg decode failed";
      return;
    }
    auto t_dec1 = std::chrono::steady_clock::now();
    double lat_dec =
        std::chrono::duration_cast<std::chrono::microseconds>(t_dec1 - t_dec0).count() / 1000.0;
    uint64_t src_ts_mono = lease->ts_ns_monotonic;
    uint64_t src_epoch = lease->epoch;
    std::size_t need = static_cast<std::size_t>(dec_w) * dec_h * 3;
    cudaMemcpy(bgr_gpu, bgr.data(), need, cudaMemcpyHostToDevice);
    auto t_inf0 = std::chrono::steady_clock::now();
    edgevision::cuda_preprocess::preprocess(static_cast<const uint8_t*>(bgr_gpu), dec_w,
                                            dec_h,
                                            static_cast<float*>(engine.input_gpu()), target_w,
                                            target_h, stream);
    if (!engine.infer(stream)) {
      std::lock_guard<std::mutex> lk(mu);
      infer_fail++;
      last_fault = "trt infer failed";
      push_event_locked("fault", last_fault);
      return;
    }
    cudaStreamSynchronize(stream);
    auto t_inf1 = std::chrono::steady_clock::now();
    double lat_inf =
        std::chrono::duration_cast<std::chrono::microseconds>(t_inf1 - t_inf0).count() / 1000.0;
    cudaMemcpy(output.data(), engine.output_gpu(), output.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    auto t_post0 = std::chrono::steady_clock::now();
    auto dets = postprocess(output.data(), num_anchors, num_classes, conf_thresh, iou_thresh);
    auto t_post1 = std::chrono::steady_clock::now();
    double lat_post =
        std::chrono::duration_cast<std::chrono::microseconds>(t_post1 - t_post0).count() / 1000.0;
    double lat_ms = lat_dec + lat_inf + lat_post;
    update_snapshot(bgr, dec_w, dec_h, dets, frame_no, src_ts_mono, src_epoch);
    if (!record) return;
    // ---- sys sample OUTSIDE mu (~1Hz cache, pipeline-thread-local, no lock)
    // CpuMeter has its own lock; /proc + /sys reads never run under mu.
    uint64_t mono_now = mono_ns_now();
    if (last_sys_sample_mono == 0 || mono_now - last_sys_sample_mono >= 1000000000ull) {
      plc_cpu = cpu_meter.sample();
      plc_gpu = read_gpu_load();
      plc_temp = read_temp_c();
      MemInfo sm = read_mem();
      plc_mem_total = sm.total_mb;
      plc_mem_avail = sm.avail_mb;
      plc_mem_used =
          (sm.total_mb > 0 && sm.avail_mb >= 0) ? (sm.total_mb - sm.avail_mb) : -1;
      last_sys_sample_mono = mono_now;
    }
    // ---- M7 rule + M8a review dump (disk I/O OUTSIDE mu; mu is counters only)
    double mx = 0;
    for (const auto& d : dets) mx = std::max(mx, static_cast<double>(d.conf));
    bool low;
    uint64_t streak_after = 0;
    bool want_dump = false;
    uint64_t wall_ns = wall_ns_now();
    std::string cur_rule;
    {
      std::lock_guard<std::mutex> lk(mu);
      frames_ok++;
      dets_total += dets.size();
      push_capped(latencies, lat_ms);
      push_capped(lat_decode, lat_dec);
      push_capped(lat_infer, lat_inf);
      push_capped(lat_post, lat_post);
      trend_add_locked(wall_ns / 1000000000ull, lat_inf, plc_cpu, plc_gpu,
                       plc_mem_used);
      // ---- M7 rule hint (metadata only, no pipeline change) ----
      last_max_conf = mx;
      low = (dets.empty() || mx < low_conf_thresh);
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
      cur_rule = active_model;
      streak_after = low_conf_streak;
      want_dump = (record && low && low_conf_streak == 3);
      if (want_dump) {
        reviews_written++;  // count attempt; rolled back below on failure
      }
    }
    if (events_fp) {
      std::fprintf(events_fp, "{\"frame\":%llu,\"ts_wall_ns\":%llu,\"lat_ms\":%.2f,"
                              "\"dec_ms\":%.2f,\"inf_ms\":%.2f,\"post_ms\":%.2f,"
                              "\"dets\":%zu,\"max_conf\":%.3f,\"low_conf\":%d,"
                              "\"rule\":\"%s\"}\n",
                   (unsigned long long)frame_no, (unsigned long long)wall_ns, lat_ms,
                   lat_dec, lat_inf, lat_post, dets.size(), mx, low ? 1 : 0,
                   cur_rule.c_str());
    }
    if (want_dump) {
      // Heavy JPEG + mkdir + rename fully outside mu: STATUS/HTTP never block.
      std::string review_stamp =
          maybe_dump_review(frame_no, bgr, dec_w, dec_h, dets, mx, dets.empty(), 3);
      std::lock_guard<std::mutex> lk(mu);
      if (!review_stamp.empty()) {
        push_event_locked("review", review_stamp);
        if (events_fp) {
          std::fprintf(events_fp, "{\"frame\":%llu,\"ts_wall_ns\":%llu,"
                                  "\"escalated\":true,\"review\":\"%s\"}\n",
                       (unsigned long long)frame_no, (unsigned long long)wall_ns_now(),
                       review_stamp.c_str());
        }
      } else {
        // dump failed: keep streak/stats, just no review dir this time
        reviews_written--;
        last_fault = "review dump failed";
      }
    }
  }

  // ---- web snapshot: downscale to 480px wide, draw boxes, store ----
  void update_snapshot(const std::vector<uint8_t>& bgr, int w, int h,
                       const std::vector<Det>& dets, uint64_t frame_no,
                       uint64_t src_ts_mono_ns, uint64_t src_epoch) {
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
      // model coords are PIXELS in target_w x target_h space (NOT normalized).
      // Preprocess is a direct full-frame stretch (no letterbox), so scale by
      // sx/sy into the 480px snapshot. (Fix: was scaling by dst_w directly,
      // which collapsed boxes to the edge.)
      float cx = d.x * sx, cy = d.y * sy;
      float bw = d.w * sx, bh = d.h * sy;
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
    }
    std::lock_guard<std::mutex> lk(snap_mu);
    snap_bgr.swap(small);
    snap_w = dst_w;
    snap_h = dst_h;
    snap_frame++;
    snap_src_id = frame_no;
    snap_ts_monotonic_ns = src_ts_mono_ns;
    snap_epoch = src_epoch;
    snap_src_w = w;
    snap_src_h = h;
    snap_dets.swap(px);
    // M10: push annotated frame to RTSP (drop-oldest, never blocks).
    // Lazy-start on first real frame so width/height always match.
    if (rtsp_port > 0 && !rtsp.running())
      rtsp.start(rtsp_port, snap_w, snap_h, 30, rtsp_encoder);
    // Copy under snap_mu; rtsp.push does its own drop when viewers are slow.
    rtsp.push(snap_bgr, snap_w, snap_h);
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
  // Board info is filled once at startup (run()) and never does file I/O
  // under mu. No lazy fill here: STATUS/META/METRICS never touch /proc,/sys
  // while holding mu (samples are taken before locking, see handle_cmd).

  std::string handle_cmd(const std::string& line) {
    if (line == "STATUS") {
      // Sample sys files OUTSIDE mu (never hold mu across /proc,/sys I/O).
      MemInfo mi = read_mem();
      double temp = read_temp_c();
      double gpu = read_gpu_load();
      double cpu = cpu_meter.sample();
      // Snapshot counters under mu; format JSON after lock is released.
      DaemonState st_state;
      uint64_t c_frames_ok, c_decode_fail, c_infer_fail, c_timeouts;
      uint64_t c_low_conf_frames, c_low_conf_streak, c_reviews;
      double c_last_max_conf;
      uint64_t c_start_count, c_stop_count;
      std::string c_session, c_model, c_reason, c_worker, c_fault;
      uint64_t c_pool_drop, c_queue_drop;
      std::vector<double> sl;
      uint64_t now_mono;
      {
        std::lock_guard<std::mutex> lk(mu);
        auto sst = sess.stats();
        c_pool_drop = sst.frames_dropped_pool.load();
        c_queue_drop = sst.frames_dropped_queue.load();
        sl = latencies;
        st_state = state;
        c_frames_ok = frames_ok;
        c_decode_fail = decode_fail;
        c_infer_fail = infer_fail;
        c_timeouts = timeouts;
        c_session = session_dir;
        c_model = active_model;
        c_reason = rule_reason;
        c_low_conf_frames = low_conf_frames;
        c_low_conf_streak = low_conf_streak;
        c_last_max_conf = last_max_conf;
        c_reviews = reviews_written;
        c_worker = worker_state;
        c_start_count = start_count;
        c_stop_count = stop_count;
        c_fault = last_fault;
        now_mono = mono_ns_now();
      }
      std::sort(sl.begin(), sl.end());
      double avg = 0, p50 = 0, p99 = 0;
      if (!sl.empty()) {
        avg = std::accumulate(sl.begin(), sl.end(), 0.0) / sl.size();
        p50 = percentile_sorted(sl, 0.5);
        p99 = percentile_sorted(sl, 0.99);
      }
      uint64_t up_s = boot_ns_monotonic ? (now_mono - boot_ns_monotonic) / 1000000000ull : 0;
      char buf[4096];
      std::snprintf(buf, sizeof(buf),
                    "{\"ok\":true,\"state\":\"%s\",\"frames\":%llu,\"decode_fail\":%llu,"
                    "\"infer_fail\":%llu,\"timeouts\":%llu,\"pool_drop\":%llu,"
                    "\"queue_drop\":%llu,\"session\":\"%s\","
                    "\"active_model\":\"%s\",\"rule_reason\":\"%s\","
                    "\"low_conf_frames\":%llu,\"low_conf_streak\":%llu,"
                    "\"last_max_conf\":%.3f,"
                    "\"reviews_written\":%llu,\"worker\":\"%s\","
                    "\"lat_avg_ms\":%.2f,\"lat_p50_ms\":%.2f,\"lat_p99_ms\":%.2f,"
                    "\"uptime_s\":%llu,\"start_count\":%llu,\"stop_count\":%llu,"
                    "\"cpu_pct\":%.1f,\"gpu_pct\":%.1f,\"mem_total_mb\":%ld,"
                    "\"mem_avail_mb\":%ld,\"temp_c\":%.1f,"
                    "\"last_fault\":\"%s\"}",
                    state_str(st_state), (unsigned long long)c_frames_ok,
                    (unsigned long long)c_decode_fail, (unsigned long long)c_infer_fail,
                    (unsigned long long)c_timeouts,
                    (unsigned long long)c_pool_drop,
                    (unsigned long long)c_queue_drop, c_session.c_str(),
                    c_model.c_str(), c_reason.c_str(),
                    (unsigned long long)c_low_conf_frames,
                    (unsigned long long)c_low_conf_streak, c_last_max_conf,
                    (unsigned long long)c_reviews, c_worker.c_str(),
                    avg, p50, p99, (unsigned long long)up_s,
                    (unsigned long long)c_start_count, (unsigned long long)c_stop_count,
                    cpu, gpu,
                    mi.total_mb, mi.avail_mb, temp,
                    json_escape(c_fault).c_str());
      return std::string(buf);
    }
    if (line == "META") {
      // Snapshot under mu; format outside (never hold mu across formatting).
      std::string c_ver, c_board, c_l4t, c_eng, c_dev, c_sock;
      int c_tw = 0, c_th = 0, c_nc = 0, c_http = 0, c_rtsp = 0;
      uint64_t c_up = 0;
      {
        std::lock_guard<std::mutex> lk(mu);
        c_ver = version;
        c_board = board_model;
        c_l4t = l4t_version;
        c_eng = engine_path;
        c_tw = target_w;
        c_th = target_h;
        c_nc = num_classes;
        c_dev = device;
        c_sock = sock_path;
        c_http = http_port;
        c_rtsp = rtsp_port;
        uint64_t now_mono = mono_ns_now();
        c_up = boot_ns_monotonic ? (now_mono - boot_ns_monotonic) / 1000000000ull : 0;
      }
      std::string j = "{\"ok\":true,\"version\":\"" + json_escape(c_ver) + "\""
                      ",\"board_model\":\"" + json_escape(c_board) + "\""
                      ",\"l4t\":\"" + json_escape(c_l4t) + "\""
                      ",\"engine\":\"" + json_escape(c_eng) + "\""
                      ",\"input_wh\":[" + std::to_string(c_tw) + "," +
                      std::to_string(c_th) + "]"
                      ",\"num_classes\":" + std::to_string(c_nc) +
                      ",\"device\":\"" + json_escape(c_dev) + "\""
                      ",\"sock\":\"" + json_escape(c_sock) + "\""
                      ",\"http_port\":" + std::to_string(c_http) +
                      ",\"rtsp_port\":" + std::to_string(c_rtsp) +
                      ",\"uptime_s\":" + std::to_string(c_up) + "}";
      return j;
    }
    if (line == "METRICS") {
      // Sample sys files OUTSIDE mu; copy vectors/trend under mu; format after.
      MemInfo mmi = read_mem();
      double mtemp = read_temp_c();
      double mgpu = read_gpu_load();
      double mcpu = cpu_meter.sample();
      std::vector<double> c_dec, c_inf, c_post;
      std::deque<TrendPt> c_trend;
      uint64_t c_qpeak = 0, c_pdrop = 0, c_qdrop = 0;
      {
        std::lock_guard<std::mutex> lk(mu);
        auto sst = sess.stats();
        c_qpeak = sst.queue_peak_depth.load();
        c_pdrop = sst.frames_dropped_pool.load();
        c_qdrop = sst.frames_dropped_queue.load();
        c_dec = lat_decode;
        c_inf = lat_infer;
        c_post = lat_post;
        c_trend = trend;
      }
      auto pct = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        double a = 0, p5 = 0, p9 = 0;
        if (!v.empty()) {
          a = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
          p5 = percentile_sorted(v, 0.5);
          p9 = percentile_sorted(v, 0.99);
        }
        char b[128];
        std::snprintf(b, sizeof(b), "{\"avg\":%.2f,\"p50\":%.2f,\"p99\":%.2f,\"n\":%zu}",
                      a, p5, p9, v.size());
        return std::string(b);
      };
      MemInfo mi = mmi;
      double temp = mtemp;
      std::string j = "{\"ok\":true,\"decode\":" + pct(c_dec) + ",\"infer\":" + pct(c_inf) +
                      ",\"post\":" + pct(c_post);
      char b[256];
      std::snprintf(b, sizeof(b),
                    ",\"queue_peak\":%llu,\"pool_drop\":%llu,\"queue_drop\":%llu,"
                    "\"mem_total_mb\":%ld,\"mem_avail_mb\":%ld,"
                    "\"cpu_pct\":%.1f,\"gpu_pct\":%.1f,\"temp_c\":%.1f,\"trend_n\":%zu}",
                    (unsigned long long)c_qpeak,
                    (unsigned long long)c_pdrop,
                    (unsigned long long)c_qdrop, mi.total_mb,
                    mi.avail_mb, mcpu, mgpu, temp,
                    c_trend.size());
      j += b;
      // trend array (last 120 buckets)
      j += ",\"trend\":[";
      bool first = true;
      for (auto& t : c_trend) {
        if (!first) j += ",";
        first = false;
        char tb[192];
        std::snprintf(tb, sizeof(tb),
                      "{\"t\":%llu,\"fps\":%.1f,\"infer\":%.2f,\"cpu\":%.1f,\"gpu\":%.1f,"
                      "\"mem\":%ld}",
                      (unsigned long long)t.t, t.fps, t.infer, t.cpu, t.gpu, t.mem);
        j += tb;
      }
      j += "]}";
      return j;
    }
    if (line == "EVENTS") {
      std::deque<std::string> c_events;
      {
        std::lock_guard<std::mutex> lk(mu);
        c_events = events;
      }
      std::string j = "{\"ok\":true,\"events\":[";
      bool first = true;
      for (auto& e : c_events) {
        if (!first) j += ",";
        first = false;
        j += e;
      }
      j += "]}";
      return j;
    }
    if (line == "RULE") {
      std::string c_model, c_reason, c_rdir;
      double c_thresh = 0, c_max = 0;
      uint64_t c_lcf = 0, c_lcs = 0, c_rev = 0;
      {
        std::lock_guard<std::mutex> lk(mu);
        c_model = active_model;
        c_reason = rule_reason;
        c_thresh = low_conf_thresh;
        c_lcf = low_conf_frames;
        c_lcs = low_conf_streak;
        c_max = last_max_conf;
        c_rev = reviews_written;
        c_rdir = review_dir;
      }
      char buf[1024];
      std::snprintf(buf, sizeof(buf),
                    "{\"ok\":true,\"active_model\":\"%s\",\"reason\":\"%s\","
                    "\"low_conf_thresh\":%.2f,\"low_conf_frames\":%llu,"
                    "\"low_conf_streak\":%llu,\"last_max_conf\":%.3f,"
                    "\"reviews_written\":%llu,\"review_dir\":\"%s\"}",
                    c_model.c_str(), c_reason.c_str(), c_thresh,
                    (unsigned long long)c_lcf,
                    (unsigned long long)c_lcs, c_max,
                    (unsigned long long)c_rev, c_rdir.c_str());
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
      std::string c_worker;
      uint64_t c_rev = 0;
      {
        std::lock_guard<std::mutex> lk(mu);
        c_worker = worker_state;
        c_rev = reviews_written;
      }
      return "{\"ok\":true,\"worker\":\"" + c_worker + "\",\"reviews\":" +
             std::to_string(c_rev) + "}";
    }
    if (line == "START") {
      std::string err;
      if (do_start(err)) {
        std::string c_session;
        {
          std::lock_guard<std::mutex> lk(mu);
          c_session = session_dir;
        }
        char buf[256];
        std::snprintf(buf, sizeof(buf), "{\"ok\":true,\"accepted\":true,\"session\":\"%s\"}",
                      c_session.c_str());
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
        // idle: heartbeat every ~5s for journald (snapshot under mu, print outside)
        if (++tick % 5 == 0) {
          DaemonState hb_state;
          uint64_t hb_frames;
          std::string hb_session;
          {
            std::lock_guard<std::mutex> lk(mu);
            hb_state = state;
            hb_frames = frames_ok;
            hb_session = session_dir;
          }
          std::printf("[edgevisiond] heartbeat state=%s frames=%llu session=%s\n",
                      state_str(hb_state), (unsigned long long)hb_frames,
                      hb_session.empty() ? "-" : hb_session.c_str());
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
    char head[384];
    int hl = std::snprintf(head, sizeof(head),
                           "HTTP/1.0 200 OK\r\nContent-Type: %s\r\n"
                           "Content-Length: %zu\r\nCache-Control: no-cache\r\n"
                           "Access-Control-Allow-Origin: *\r\n"
                           "Access-Control-Allow-Methods: GET, OPTIONS\r\n"
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
    // CORS preflight for the file:// interview demo page (double-click HTML).
    if (line.rfind("OPTIONS", 0) == 0) {
      const char* pre =
          "HTTP/1.0 200 OK\r\nAccess-Control-Allow-Origin: *\r\n"
          "Access-Control-Allow-Methods: GET, OPTIONS\r\nContent-Length: 0\r\n"
          "Connection: close\r\n\r\n";
      send_all(cfd, pre, std::strlen(pre));
      return;
    }
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
        char head[256];
        int hl = std::snprintf(head, sizeof(head),
                               "HTTP/1.0 503 Busy\r\nContent-Type: text/plain\r\n"
                               "Content-Length: %zu\r\nCache-Control: no-cache\r\n"
                               "Access-Control-Allow-Origin: *\r\n"
                               "Access-Control-Allow-Methods: GET, OPTIONS\r\n"
                               "Connection: close\r\n\r\n",
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
      std::string j;
      {
        std::lock_guard<std::mutex> lk(snap_mu);
        j = "{\"frame\":" + std::to_string(snap_frame) +
            ",\"src_id\":" + std::to_string(snap_src_id) +
            ",\"epoch\":" + std::to_string(snap_epoch) +
            ",\"ts_mono_ns\":" + std::to_string(snap_ts_monotonic_ns) +
            ",\"src_wh\":[" + std::to_string(snap_src_w) + "," +
            std::to_string(snap_src_h) + "]"
            ",\"snap_wh\":[" + std::to_string(snap_w) + "," + std::to_string(snap_h) +
            "],\"dets\":[";
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
      }
      send_resp(cfd, "application/json", j.data(), j.size());
    } else if (line.rfind("GET /api/meta", 0) == 0) {
      auto r = handle_cmd("META");
      send_resp(cfd, "application/json", r.data(), r.size());
    } else if (line.rfind("GET /api/metrics", 0) == 0) {
      auto r = handle_cmd("METRICS");
      send_resp(cfd, "application/json", r.data(), r.size());
    } else if (line.rfind("GET /api/events", 0) == 0) {
      auto r = handle_cmd("EVENTS");
      send_resp(cfd, "application/json", r.data(), r.size());
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
    boot_ns_monotonic = mono_ns_now();
    // Board identity once at startup (no per-request file I/O under mu).
    board_model = read_file_trim("/proc/device-tree/model");
    {
      std::string l = read_file_trim("/etc/nv_tegra_release", 4096);
      auto nl = l.find('\n');
      l4t_version = (nl == std::string::npos) ? l : l.substr(0, nl);
    }
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
    // M10: RTSP live monitor (annotated 480px frames, drop-oldest).
    // Lazy-started on first snapshot frame (real width/height); port 0 disables.
    if (rtsp_port > 0)
      std::printf("[edgevisiond] rtsp armed on :%d/live (encoder=%s)\n",
                  rtsp_port, rtsp_encoder.c_str());
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
    "<!doctype html><html lang=zh><head><meta charset=utf-8>"
    "<meta name=viewport content='width=device-width,initial-scale=1'>"
    "<title>EdgeVisionRT Console</title>"
    "<style>"
    ":root{--bg:#0b0f14;--panel:#111823;--line:#1e2a3a;--txt:#d7e3f4;--dim:#8aa0b8;"
    "--acc:#2fbf71;--warn:#f5a623;--bad:#ff5d5d;--info:#4aa8ff}"
    "*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--txt);"
    "font:13px/1.5 -apple-system,'Segoe UI',Roboto,'Noto Sans SC',monospace}"
    "header{display:flex;gap:10px;align-items:center;padding:10px 14px;"
    "background:var(--panel);border-bottom:1px solid var(--line);position:sticky;top:0;z-index:5}"
    "header b{font-size:15px}header .ver{color:var(--dim);font-size:12px}"
    ".pill{padding:2px 10px;border-radius:20px;border:1px solid var(--line);font-size:12px}"
    ".pill.run{color:var(--acc);border-color:var(--acc)}"
    ".pill.idle{color:var(--dim)}.pill.stop{color:var(--warn);border-color:var(--warn)}"
    "button{background:#16202e;color:var(--txt);border:1px solid var(--line);border-radius:6px;"
    "padding:6px 14px;cursor:pointer}button:hover{border-color:var(--acc)}"
    "button.danger:hover{border-color:var(--bad)}"
    "main{display:grid;grid-template-columns:minmax(320px,1.35fr) minmax(300px,1fr);gap:12px;padding:12px}"
    ".card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 12px}"
    ".card h3{margin:0 0 8px;font-size:13px;color:var(--dim);font-weight:600}"
    "#wrap{position:relative;display:inline-block;max-width:100%}"
    "#view{max-width:100%;border:1px solid var(--line);border-radius:6px;display:block;background:#000}"
    "#ov{position:absolute;left:0;top:0;pointer-events:none}"
    ".kv{display:grid;grid-template-columns:auto 1fr;gap:2px 10px;font-size:12px}"
    ".kv dt{color:var(--dim)}.kv dd{margin:0;text-align:right;font-variant-numeric:tabular-nums}"
    ".bar{height:6px;background:#0a0e13;border-radius:4px;overflow:hidden;margin:3px 0 8px}"
    ".bar i{display:block;height:100%;background:var(--acc)}"
    "table{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}"
    "th,td{text-align:right;padding:3px 6px;border-bottom:1px solid var(--line)}"
    "th:first-child,td:first-child{text-align:left}"
    "th{color:var(--dim);font-weight:600}"
    "#events{max-height:220px;overflow:auto;font-size:12px}"
    "#events div{padding:3px 6px;border-left:3px solid var(--line);margin:4px 0;background:#0d131c}"
    "#events div.start{border-color:var(--acc)}#events div.review{border-color:var(--warn)}"
    "#events div.fault{border-color:var(--bad)}"
    "#fault{display:none;background:#2a1214;border:1px solid var(--bad);color:#ffb3b3;"
    "border-radius:8px;padding:8px 10px;margin:0 12px}"
    "canvas#trend{width:100%;height:90px;background:#0a0e13;border:1px solid var(--line);border-radius:6px}"
    "footer{color:var(--dim);font-size:11px;padding:8px 14px}"
    "a{color:var(--info)}@media(max-width:900px){main{grid-template-columns:1fr}}"
    "</style></head><body>"
    "<header><b>EdgeVisionRT</b><span class=ver id=ver>console</span>"
    "<span class='pill idle' id=state>IDLE</span>"
    "<span style='flex:1'></span>"
    "<button onclick=\"api('/api/start')\">START</button>"
    "<button class=danger onclick=\"api('/api/stop')\">STOP</button>"
    "<button onclick=\"tick()\">Refresh</button>"
    "<a id=rtsp style='font-size:12px' href='#'>rtsp</a></header>"
    "<div id=fault></div>"
    "<main>"
    "<section class=card><h3>Live (snapshot 480px + RTSP)</h3>"
    "<div id=wrap><img id=view src='/snapshot.jpg' alt='live'>"
    "<canvas id=ov></canvas></div>"
    "<div class=kv style='margin-top:8px'><dt>frame</dt><dd id=f_frame>--</dd>"
    "<dt>src frame / epoch</dt><dd id=f_src>--</dd>"
    "<dt>detections</dt><dd id=f_dets>--</dd>"
    "<dt>capture ts (mono)</dt><dd id=f_ts>--</dd></div></section>"
    "<section class=card><h3>Pipeline latency (capped 600 samples)</h3>"
    "<table><tr><th>stage</th><th>avg</th><th>p50</th><th>p99</th><th>n</th></tr>"
    "<tr><td>decode</td><td id=m_dec_a>--</td><td id=m_dec_p>--</td><td id=m_dec_9>--</td><td id=m_dec_n>--</td></tr>"
    "<tr><td>infer (gpu)</td><td id=m_inf_a>--</td><td id=m_inf_p>--</td><td id=m_inf_9>--</td><td id=m_inf_n>--</td></tr>"
    "<tr><td>post (nms)</td><td id=m_post_a>--</td><td id=m_post_p>--</td><td id=m_post_9>--</td><td id=m_post_n>--</td></tr></table>"
    "<h3 style='margin-top:10px'>Trend (last 120s, 1s buckets)</h3><canvas id=trend></canvas>"
    "<div class=kv><dt>queue peak</dt><dd id=m_q>--</dd>"
    "<dt>drops pool/queue</dt><dd id=m_drop>--</dd></div></section>"
    "<section class=card><h3>Board / service</h3>"
    "<div class=kv><dt>board</dt><dd id=b_board>--</dd><dt>L4T</dt><dd id=b_l4t>--</dd>"
    "<dt>engine</dt><dd id=b_eng>--</dd><dt>input</dt><dd id=b_in>--</dd>"
    "<dt>uptime</dt><dd id=b_up>--</dd><dt>starts/stops</dt><dd id=b_ss>--</dd></div>"
    "<h3 style='margin-top:10px'>Resources</h3>"
    "<div class=kv><dt>CPU</dt><dd id=r_cpu>--</dd></div><div class=bar><i id=r_cpu_b></i></div>"
    "<div class=kv><dt>GPU</dt><dd id=r_gpu>--</dd></div><div class=bar><i id=r_gpu_b></i></div>"
    "<div class=kv><dt>MEM</dt><dd id=r_mem>--</dd><dt>TEMP</dt><dd id=r_temp>--</dd>"
    "<dt>session</dt><dd id=r_sess>--</dd></div></section>"
    "<section class=card><h3>Scheduler rule + worker</h3>"
    "<div class=kv><dt>rule model</dt><dd id=w_model>--</dd><dt>reason</dt><dd id=w_reason>--</dd>"
    "<dt>low-conf</dt><dd id=w_low>--</dd><dt>max conf</dt><dd id=w_max>--</dd>"
    "<dt>reviews</dt><dd id=w_rev>--</dd><dt>worker</dt><dd id=w_state>--</dd></div>"
    "<h3 style='margin-top:10px'>Events (cap 200, newest last)</h3><div id=events></div></section>"
    "</main>"
    "<footer>Data: <code>/snapshot.jpg</code> + <code>/api/{status,rule,worker,dets,meta,metrics,events}</code>"
    " | Latency clocks: decode/infer/post via steady_clock, GPU time includes enqueue+sync"
    " | RTSP: <code id=rtsp2></code></footer>"
    "<script>"
    "const $=id=>document.getElementById(id);"
    "const img=$('view'),ov=$('ov');"
    "function api(p){fetch(p).then(r=>r.text()).then(()=>tick()).catch(()=>{});}"
    "function fmt(v,d=1){return (v==null||v<0)?'--':(+v).toFixed(d);}"
    "function drawBoxes(j){"
    "const r=img.getBoundingClientRect();ov.width=r.width;ov.height=r.height;"
    "ov.style.width=r.width+'px';ov.style.height=r.height+'px';"
    "const c=ov.getContext('2d');c.clearRect(0,0,ov.width,ov.height);"
    "if(!img.naturalWidth)return;"
    "const kx=ov.width/img.naturalWidth,ky=ov.height/img.naturalHeight;"
    "c.lineWidth=2;c.font='12px monospace';"
    "(j.dets||[]).forEach(d=>{c.strokeStyle=d.conf>=0.5?'#2fbf71':'#f5a623';"
    "c.strokeRect(d.x*kx,d.y*ky,d.w*kx,d.h*ky);c.fillStyle=c.strokeStyle;"
    "c.fillText((d.cls??'?')+':'+(+d.conf).toFixed(2),d.x*kx,(d.y*ky-4>10?d.y*ky-4:d.y*ky+14));});}"
    "function drawTrend(tr){const cv=$('trend'),c=cv.getContext('2d');"
    "cv.width=cv.clientWidth||600;cv.height=90;c.clearRect(0,0,cv.width,cv.height);"
    "if(!tr||!tr.length)return;const W=cv.width,H=cv.height;"
    "let mx=1;tr.forEach(t=>{mx=Math.max(mx,t.infer||0)});"
    "c.strokeStyle='#2fbf71';c.beginPath();"
    "tr.forEach((t,i)=>{const x=i/(tr.length-1||1)*W,y=H-4-(t.infer/mx)*(H-10);"
    "i?c.lineTo(x,y):c.moveTo(x,y)});c.stroke();"
    "c.fillStyle='#8aa0b8';c.fillText('infer p50 ms (max '+mx.toFixed(1)+')',6,12);}"
    "async function tick(){"
    "img.src='/snapshot.jpg?t='+Date.now();"
    "try{const j=await(await fetch('/api/dets')).json();drawBoxes(j);"
    "$('f_frame').textContent=j.frame;$('f_src').textContent=j.src_id+' / epoch '+j.epoch;"
    "$('f_dets').textContent=(j.dets||[]).length;"
    "$('f_ts').textContent=(j.ts_mono_ns/1e9).toFixed(2)+'s';}catch(e){}"
    "try{const s=await(await fetch('/api/status')).json();"
    "const st=$('state');st.textContent=s.state;"
    "st.className='pill '+(s.state==='RUNNING'?'run':s.state==='IDLE'?'idle':'stop');"
    "$('r_sess').textContent=s.session||'-';"
    "$('w_low').textContent=s.low_conf_streak+' streak / '+s.low_conf_frames+' frames';"
    "$('w_max').textContent=fmt(s.last_max_conf,2);$('w_rev').textContent=s.reviews_written;"
    "$('r_cpu').textContent=fmt(s.cpu_pct)+'%';$('r_cpu_b').style.width=Math.max(0,s.cpu_pct)+'%';"
    "$('r_gpu').textContent=fmt(s.gpu_pct)+'%';$('r_gpu_b').style.width=Math.max(0,s.gpu_pct)+'%';"
    "$('r_mem').textContent=(s.mem_avail_mb<0?'--':(s.mem_total_mb-s.mem_avail_mb)+'/'+s.mem_total_mb+' MB');"
    "$('r_temp').textContent=fmt(s.temp_c)+' C';"
    "$('b_ss').textContent=s.start_count+' / '+s.stop_count;"
    "$('b_up').textContent=s.uptime_s+'s';"
    "const f=$('fault');if(s.last_fault){f.style.display='block';f.textContent='FAULT: '+s.last_fault;}"
    "else f.style.display='none';}catch(e){}"
    "try{const g=await(await fetch('/api/rule')).json();"
    "$('w_model').textContent=g.active_model;$('w_reason').textContent=g.reason;}catch(e){}"
    "try{const w=await(await fetch('/api/worker')).json();"
    "$('w_state').textContent=w.worker;}catch(e){}"
    "try{const m=await(await fetch('/api/meta')).json();"
    "$('ver').textContent=m.version;$('b_board').textContent=m.board_model||'--';"
    "$('b_l4t').textContent=(m.l4t||'--').slice(0,40);$('b_eng').textContent=m.engine;"
    "$('b_in').textContent=m.input_wh;$('rtsp2').textContent='rtsp://<board-ip>:'+m.rtsp_port+'/live';"
    "$('rtsp').href='#';}catch(e){}"
    "try{const k=await(await fetch('/api/metrics')).json();"
    "$('m_dec_a').textContent=fmt(k.decode.avg,2);$('m_dec_p').textContent=fmt(k.decode.p50,2);"
    "$('m_dec_9').textContent=fmt(k.decode.p99,2);$('m_dec_n').textContent=k.decode.n;"
    "$('m_inf_a').textContent=fmt(k.infer.avg,2);$('m_inf_p').textContent=fmt(k.infer.p50,2);"
    "$('m_inf_9').textContent=fmt(k.infer.p99,2);$('m_inf_n').textContent=k.infer.n;"
    "$('m_post_a').textContent=fmt(k.post.avg,2);$('m_post_p').textContent=fmt(k.post.p50,2);"
    "$('m_post_9').textContent=fmt(k.post.p99,2);$('m_post_n').textContent=k.post.n;"
    "$('m_q').textContent=k.queue_peak;$('m_drop').textContent=k.pool_drop+' / '+k.queue_drop;"
    "drawTrend(k.trend);}catch(e){}"
    "try{const v=await(await fetch('/api/events')).json();const box=$('events');box.innerHTML='';"
    "(v.events||[]).slice(-40).forEach(e=>{const d=document.createElement('div');"
    "d.className=e.type||'';d.textContent=new Date(e.t*1000).toLocaleTimeString()+' ['+e.type+'] '+e.detail;"
    "box.appendChild(d);});box.scrollTop=box.scrollHeight;}catch(e){}"
    "}"
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
    else if (std::strcmp(argv[i], "--rtsp") == 0 && i + 1 < argc)
      d.rtsp_port = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--rtsp-encoder") == 0 && i + 1 < argc)
      d.rtsp_encoder = argv[++i];
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
