#pragma once
// BoundedQueue<T>: SPSC/MPSC-safe bounded FIFO with drop-newest-on-full.
//
// Why bounded (never unbounded): on an 8GB edge box, an unbounded queue
// turns a transient consumer stall (e.g. inference jitter) into OOM.
// Bounded + explicit drop counter makes backpressure VISIBLE in metrics.
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <queue>

namespace edgevision {

template <typename T>
class BoundedQueue {
 public:
  explicit BoundedQueue(std::size_t capacity) : capacity_(capacity) {}
  ~BoundedQueue() = default;
  BoundedQueue(const BoundedQueue&) = delete;
  BoundedQueue& operator=(const BoundedQueue&) = delete;

  // Push; returns false + bumps dropped_ if full (never blocks producer).
  bool try_push(T item) {
    std::lock_guard<std::mutex> lk(mu_);
    if (q_.size() >= capacity_) {
      dropped_++;
      return false;
    }
    q_.push(std::move(item));
    cv_.notify_one();
    return true;
  }

  // Blocking pop with timeout (ms). Returns false on timeout/shutdown.
  bool wait_pop(T& out, int timeout_ms) {
    std::unique_lock<std::mutex> lk(mu_);
    if (!cv_.wait_for(lk, std::chrono::milliseconds(timeout_ms),
                       [&] { return !q_.empty() || shutdown_; })) {
      return false;  // timeout
    }
    if (q_.empty()) return false;  // shutdown with empty queue
    out = std::move(q_.front());
    q_.pop();
    return true;
  }

  void shutdown() {
    std::lock_guard<std::mutex> lk(mu_);
    shutdown_ = true;
    cv_.notify_all();
  }

  std::size_t size() const {
    std::lock_guard<std::mutex> lk(mu_);
    return q_.size();
  }
  uint64_t dropped() const {
    std::lock_guard<std::mutex> lk(mu_);
    return dropped_;
  }

 private:
  const std::size_t capacity_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::queue<T> q_;
  uint64_t dropped_ = 0;
  bool shutdown_ = false;
};

}  // namespace edgevision
