// capture_probe: bring-up tool. Enumerate cameras, or stream N frames and report stats.
// Usage:
//   capture_probe --list                                   enumerate /dev/video* (V4L2 build only)
//   capture_probe --stream N [--device /dev/video0]        capture N frames MJPG 1280x720 by default
//   capture_probe --stream N --width 1280 --height 720 --fourcc MJPG [--out /tmp/x.mjpg]
// On hosts without V4L2 this binary prints HARDWARE_REQUIRED and exits 2.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#ifdef __linux__
#include <linux/videodev2.h>

#include "edgevision/frame.h"
#include "edgevision/v4l2_capture.h"

namespace {

uint32_t parse_fourcc(const char* s) {
  if (!s || std::strlen(s) < 4) return v4l2_fourcc('M', 'J', 'P', 'G');
  return v4l2_fourcc(s[0], s[1], s[2], s[3]);
}

void usage() {
  std::printf(
      "usage:\n"
      "  capture_probe --list\n"
      "  capture_probe --stream N [--device /dev/video0] [--width 1280 --height 720]\n"
      "                 [--fourcc MJPG] [--out /tmp/out.mjpg]\n");
}

int cmd_list() {
  auto cams = edgevision::V4L2Capture::enumerate();
  std::printf("[capture_probe] found %zu video-capture nodes\n", cams.size());
  for (const auto& c : cams) {
    std::printf("  %s | %s | %s\n", c.device.c_str(), c.name.c_str(), c.bus.c_str());
  }
  if (cams.empty()) {
    std::printf("[capture_probe] HINT: no /dev/video* with VIDEO_CAPTURE; "
                "check camera plug + uvcvideo + lsusb\n");
    return 1;
  }
  return 0;
}

int cmd_stream(int n, const std::string& dev, uint32_t w, uint32_t h, uint32_t fourcc,
               const std::string& out_path) {
  edgevision::V4L2Capture cap;
  if (!cap.open(dev)) {
    std::printf("[capture_probe] FAIL open %s (errno path: check permissions / busy)\n",
                dev.c_str());
    return 1;
  }
  if (!cap.start(w, h, fourcc)) {
    std::printf("[capture_probe] FAIL start %ux%u fourcc=%.4s on %s\n", w, h,
                reinterpret_cast<const char*>(&fourcc), dev.c_str());
    return 1;
  }

  FILE* out = nullptr;
  if (!out_path.empty()) {
    out = std::fopen(out_path.c_str(), "wb");
    if (!out) std::printf("[capture_probe] WARN cannot open out %s, continue w/o save\n",
                          out_path.c_str());
  }

  char cc[5] = {char(fourcc & 0xFF), char((fourcc >> 8) & 0xFF),
                char((fourcc >> 16) & 0xFF), char((fourcc >> 24) & 0xFF), 0};
  std::printf("[capture_probe] stream %d frames from %s %ux%u %s\n", n, dev.c_str(), w, h,
              cc);

  edgevision::Frame f;
  int ok = 0, timeouts = 0;
  uint64_t first_ts = 0, last_ts = 0, total_bytes = 0;
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < n; ++i) {
    f.valid = false;
    if (!cap.read(f, 0)) {
      ++timeouts;
      std::printf("[capture_probe] frame %d TIMEOUT (total timeouts=%d)\n", i, timeouts);
      continue;
    }
    if (first_ts == 0) first_ts = f.ts_ns_monotonic;
    last_ts = f.ts_ns_monotonic;
    total_bytes += f.data.size();
    ++ok;
    // Per-frame line: index, bytes, monotonic timestamp. Enough for fps/drop analysis.
    std::printf("[capture_probe] frame %d bytes=%zu ts_ns=%llu\n", i, f.data.size(),
                (unsigned long long)f.ts_ns_monotonic);
    if (out) std::fwrite(f.data.data(), 1, f.data.size(), out);
  }
  auto t1 = std::chrono::steady_clock::now();
  double wall_s =
      std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / 1000.0;
  if (out) std::fclose(out);

  double fps_wall = wall_s > 0 ? ok / wall_s : 0;
  double fps_ts = 0;
  if (ok >= 2 && last_ts > first_ts) fps_ts = (ok - 1) * 1e9 / double(last_ts - first_ts);
  std::printf("[capture_probe] SUMMARY ok=%d/%d timeouts=%d bytes=%llu avg_bytes=%.0f\n",
              ok, n, timeouts, (unsigned long long)total_bytes,
              ok ? double(total_bytes) / ok : 0);
  std::printf("[capture_probe] SUMMARY wall_s=%.2f fps_wall=%.2f fps_ts=%.2f\n", wall_s,
              fps_wall, fps_ts);
  std::printf("[capture_probe] EXPECTED ~30fps for MJPG 720p on USB2; "
              "fps_wall << 25 or timeouts > 0 means investigate (bandwidth/power/dmesg)\n");

  cap.stop();
  cap.close();
  return (ok == n) ? 0 : 2;
}

}  // namespace
#endif

int main(int argc, char** argv) {
#ifdef __linux__
  bool list = false;
  int stream_n = 0;
  std::string device = "/dev/video0";
  uint32_t width = 1280, height = 720;
  uint32_t fourcc = 0;
  std::string fourcc_str = "MJPG";
  std::string out_path;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--list") == 0)
      list = true;
    else if (std::strcmp(argv[i], "--stream") == 0 && i + 1 < argc)
      stream_n = std::atoi(argv[++i]);
    else if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc)
      device = argv[++i];
    else if (std::strcmp(argv[i], "--width") == 0 && i + 1 < argc)
      width = static_cast<uint32_t>(std::atoi(argv[++i]));
    else if (std::strcmp(argv[i], "--height") == 0 && i + 1 < argc)
      height = static_cast<uint32_t>(std::atoi(argv[++i]));
    else if (std::strcmp(argv[i], "--fourcc") == 0 && i + 1 < argc)
      fourcc_str = argv[++i];
    else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc)
      out_path = argv[++i];
    else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      usage();
      return 0;
    }
  }
  fourcc = parse_fourcc(fourcc_str.c_str());

  if (list) return cmd_list();
  if (stream_n > 0) return cmd_stream(stream_n, device, width, height, fourcc, out_path);
  usage();
  return 1;
#else
  (void)argc;
  (void)argv;
  std::printf("HARDWARE_REQUIRED: capture needs Linux + V4L2 camera.\n");
  return 2;
#endif
}
