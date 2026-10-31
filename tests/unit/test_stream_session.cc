#ifdef __linux__
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include "edgevision/stream_session.h"

int main(int argc, char** argv) {
  const char* dev = "/dev/video0";
  int n = 60;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--device") == 0 && i + 1 < argc) dev = argv[++i];
    if (std::strcmp(argv[i], "--frames") == 0 && i + 1 < argc) n = std::atoi(argv[++i]);
  }

  edgevision::StreamSession sess;

  if (!sess.start(dev, 1280, 720, 0x47504A4D, 6, 4, 4)) {
    std::printf("[test_stream_session] FAIL start: %s\n", sess.last_error().c_str());
    return 1;
  }

  auto stats0 = sess.stats();
  std::printf("[test_stream_session] negotiated %ux%u fourcc=%.4s driver_fps=%.2f\n",
              stats0.negotiated_w, stats0.negotiated_h,
              reinterpret_cast<const char*>(&stats0.negotiated_fourcc),
              stats0.driver_fps);

  int ok = 0;
  int fail_timeout = 0;
  uint64_t first_id = 0;
  uint64_t last_id = 0;
  uint64_t total_bytes = 0;

  for (int i = 0; i < n; ++i) {
    edgevision::FrameLease lease;
    if (!sess.consume(lease, 2000)) {
      fail_timeout++;
      std::printf("[test_stream_session] frame %d TIMEOUT\n", i);
      continue;
    }
    if (!lease.valid()) {
      std::printf("[test_stream_session] frame %d STALE\n", i);
      continue;
    }
    if (first_id == 0) first_id = lease->frame_id;
    last_id = lease->frame_id;
    total_bytes += lease->data.size();
    ok++;
    std::printf("[test_stream_session] frame %d id=%llu bytes=%zu ts=%llu\n",
                i, (unsigned long long)lease->frame_id, lease->data.size(),
                (unsigned long long)lease->ts_ns_monotonic);
  }

  sess.stop();
  auto stats = sess.stats();

  std::printf("[test_stream_session] SUMMARY ok=%d/%d timeout=%d total_bytes=%llu\n",
              ok, n, fail_timeout, (unsigned long long)total_bytes);
  std::printf("[test_stream_session] SUMMARY captured=%llu dropped_pool=%llu "
              "dropped_queue=%llu errors=%llu queue_peak=%llu\n",
              (unsigned long long)stats.frames_captured.load(),
              (unsigned long long)stats.frames_dropped_pool.load(),
              (unsigned long long)stats.frames_dropped_queue.load(),
              (unsigned long long)stats.frames_error.load(),
              (unsigned long long)stats.queue_peak_depth.load());

  int result = (ok == n && fail_timeout == 0) ? 0 : 2;
  std::printf("[test_stream_session] %s\n", result == 0 ? "PASS" : "FAIL");
  return result;
}
#else
int main() {
  std::printf("[test_stream_session] SKIP (V4L2 requires Linux)\n");
  return 0;
}
#endif