#pragma once
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace edgevision {

struct Frame {
  std::vector<uint8_t> data;
  uint64_t ts_ns_monotonic = 0;
  uint64_t ts_ns_wall = 0;
  uint64_t frame_id = 0;
  uint64_t epoch = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t pixfmt_fourcc = 0;  // V4L2 fourcc, e.g. 'MJPG'
  bool valid = false;
};

class FramePool {
 public:
  FramePool(std::size_t slots, std::size_t capacity);
  ~FramePool() = default;
  FramePool(const FramePool&) = delete;
  FramePool& operator=(const FramePool&) = delete;

  int acquire_index();
  Frame& slot_ref(int slot);
  int acquire(Frame& out);
  void release(int slot);
  uint64_t epoch_of(int slot) const;
  std::size_t busy() const;
  std::size_t slots() const { return frames_.size(); }

 private:
  std::vector<Frame> frames_;
  std::vector<bool> busy_;
  std::vector<uint64_t> epochs_;
  mutable std::mutex mu_;
  uint64_t next_id_ = 0;
};

}  // namespace edgevision
