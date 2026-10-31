#include "edgevision/stream_session.h"
#include <chrono>
#include <thread>

namespace edgevision {

bool StreamSession::start(const std::string& device, uint32_t width,
                           uint32_t height, uint32_t fourcc,
                           uint32_t pool_slots, uint32_t queue_capacity,
                           uint32_t buffer_count) {
  if (running_.load()) {
    last_error_ = "StreamSession: already running";
    return false;
  }

  cap_.set_buffer_count(buffer_count);
  if (!cap_.open(device)) {
    last_error_ = "StreamSession: open failed: " + cap_.last_error();
    return false;
  }

  if (!cap_.start(width, height, fourcc)) {
    last_error_ = "StreamSession: start failed: " + cap_.last_error();
    cap_.close();
    return false;
  }

  stats_ = StreamStats{};
  stats_.negotiated_w = cap_.negotiated_width();
  stats_.negotiated_h = cap_.negotiated_height();
  stats_.negotiated_fourcc = cap_.negotiated_fourcc();
  stats_.driver_fps = cap_.actual_fps();

  pool_.reset(new FramePool(pool_slots,
                             static_cast<std::size_t>(width) * height * 3));
  queue_.reset(new BoundedQueue<int>(queue_capacity));

  running_.store(true);
  worker_ = std::thread(&StreamSession::capture_loop, this);
  return true;
}

void StreamSession::capture_loop() {
  while (running_.load()) {
    int slot = pool_->acquire_index();
    if (slot < 0) {
      stats_.frames_dropped_pool.fetch_add(1, std::memory_order_relaxed);
      std::this_thread::sleep_for(std::chrono::microseconds(500));
      continue;
    }

    Frame& f = pool_->slot_ref(slot);
    if (!cap_.read(f, slot)) {
      pool_->release(slot);
      stats_.frames_error.fetch_add(1, std::memory_order_relaxed);
      continue;
    }

    f.valid = true;
    stats_.frames_captured.fetch_add(1, std::memory_order_relaxed);

    if (!queue_->try_push(slot)) {
      pool_->release(slot);
      stats_.frames_dropped_queue.fetch_add(1, std::memory_order_relaxed);
    }

    uint64_t qsz = queue_->size();
    uint64_t prev = stats_.queue_peak_depth.load(std::memory_order_relaxed);
    while (qsz > prev &&
           !stats_.queue_peak_depth.compare_exchange_weak(prev, qsz,
                                                           std::memory_order_relaxed)) {
    }
  }

  cap_.stop();
  cap_.close();
}

bool StreamSession::consume(FrameLease& lease, int timeout_ms) {
  int slot = -1;
  if (!queue_->wait_pop(slot, timeout_ms)) return false;
  lease = FrameLease(pool_.get(), slot);
  return lease.valid();
}

void StreamSession::stop() {
  if (!running_.load()) return;
  running_.store(false);
  queue_->shutdown();
  if (worker_.joinable()) worker_.join();
}

}  // namespace edgevision