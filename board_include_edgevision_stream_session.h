#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

#include "edgevision/bounded_queue.h"
#include "edgevision/frame.h"
#include "edgevision/frame_lease.h"
#include "edgevision/v4l2_capture.h"

namespace edgevision {

struct StreamStats {
  std::atomic<uint64_t> frames_captured{0};
  std::atomic<uint64_t> frames_dropped_pool{0};
  std::atomic<uint64_t> frames_dropped_queue{0};
  std::atomic<uint64_t> frames_error{0};
  std::atomic<uint64_t> queue_peak_depth{0};
  double driver_fps = 0;
  uint32_t negotiated_w = 0;
  uint32_t negotiated_h = 0;
  uint32_t negotiated_fourcc = 0;

  StreamStats() = default;
  StreamStats(const StreamStats& o) {
    frames_captured.store(o.frames_captured.load());
    frames_dropped_pool.store(o.frames_dropped_pool.load());
    frames_dropped_queue.store(o.frames_dropped_queue.load());
    frames_error.store(o.frames_error.load());
    queue_peak_depth.store(o.queue_peak_depth.load());
    driver_fps = o.driver_fps;
    negotiated_w = o.negotiated_w;
    negotiated_h = o.negotiated_h;
    negotiated_fourcc = o.negotiated_fourcc;
  }
  StreamStats& operator=(const StreamStats& o) {
    if (this != &o) {
      frames_captured.store(o.frames_captured.load());
      frames_dropped_pool.store(o.frames_dropped_pool.load());
      frames_dropped_queue.store(o.frames_dropped_queue.load());
      frames_error.store(o.frames_error.load());
      queue_peak_depth.store(o.queue_peak_depth.load());
      driver_fps = o.driver_fps;
      negotiated_w = o.negotiated_w;
      negotiated_h = o.negotiated_h;
      negotiated_fourcc = o.negotiated_fourcc;
    }
    return *this;
  }
};

class StreamSession {
 public:
  StreamSession() = default;
  ~StreamSession() { stop(); }

  StreamSession(const StreamSession&) = delete;
  StreamSession& operator=(const StreamSession&) = delete;

  bool start(const std::string& device, uint32_t width, uint32_t height,
             uint32_t fourcc, uint32_t pool_slots = 6,
             uint32_t queue_capacity = 4, uint32_t buffer_count = 4);

  bool consume(FrameLease& lease, int timeout_ms);

  void stop();

  StreamStats stats() const { return stats_; }
  const std::string& last_error() const { return last_error_; }

 private:
  void capture_loop();

  V4L2Capture cap_;
  std::unique_ptr<FramePool> pool_;
  std::unique_ptr<BoundedQueue<int>> queue_;
  std::thread worker_;
  std::atomic<bool> running_{false};
  StreamStats stats_;
  std::string last_error_;
};

}  // namespace edgevision