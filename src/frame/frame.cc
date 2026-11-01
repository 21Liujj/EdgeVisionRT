#include "edgevision/frame.h"

namespace edgevision {

FramePool::FramePool(std::size_t slots, std::size_t bytes_per_slot)
    : frames_(slots), busy_(slots, false), epochs_(slots, 0) {
  for (auto& f : frames_) {
    f.data.resize(bytes_per_slot);
    f.data.clear();  // capacity reserved, size 0 until filled
    f.valid = false;
  }
}

int FramePool::acquire_index() {
  std::lock_guard<std::mutex> lk(mu_);
  for (std::size_t i = 0; i < frames_.size(); ++i) {
    if (!busy_[i]) {
      busy_[i] = true;
      epochs_[i]++;  // generation bump: old leases with epoch-1 are stale
      Frame& slot = frames_[i];
      slot.frame_id = next_id_++;
      slot.epoch = epochs_[i];
      slot.valid = false;  // caller must fill data + set valid=true
      slot.data.clear();   // keep capacity, reset size
      return static_cast<int>(i);
    }
  }
  return -1;  // pool exhausted: caller must drop/skip, never block capture
}

Frame& FramePool::slot_ref(int slot) {
  // Storage is stable after ctor (no resize), so returning a reference is
  // safe. Bounds-checked; caller validates generation via epoch_of/valid().
  return frames_.at(static_cast<std::size_t>(slot));
}

int FramePool::acquire(Frame& out) {
  std::lock_guard<std::mutex> lk(mu_);
  for (std::size_t i = 0; i < frames_.size(); ++i) {
    if (!busy_[i]) {
      busy_[i] = true;
      epochs_[i]++;  // generation bump: old leases with epoch-1 are stale
      Frame& slot = frames_[i];
      slot.frame_id = next_id_++;
      slot.epoch = epochs_[i];
      slot.valid = false;  // caller must fill data + set valid=true
      slot.data.clear();   // keep capacity, reset size
      out = slot;          // copy metadata; shares no ownership of slot itself
      // NOTE: out.data is a copy of the vector header + bytes here is wasteful
      // for large frames; the pool hands out slot INDEX in real path.
      // This simplified acquire is for control-plane/small-frame use.
      // High-throughput path uses acquire_index()/slot_ref() (added in M1).
      out.frame_id = slot.frame_id;
      out.epoch = slot.epoch;
      return static_cast<int>(i);
    }
  }
  return -1;  // pool exhausted: caller must drop/skip, never block capture
}

void FramePool::release(int slot) {
  if (slot < 0) return;
  std::lock_guard<std::mutex> lk(mu_);
  if (static_cast<std::size_t>(slot) < busy_.size()) {
    busy_[slot] = false;
    frames_[slot].valid = false;
  }
}

uint64_t FramePool::epoch_of(int slot) const {
  std::lock_guard<std::mutex> lk(mu_);
  if (slot < 0 || static_cast<std::size_t>(slot) >= epochs_.size()) return 0;
  return epochs_[slot];
}

std::size_t FramePool::busy() const {
  std::lock_guard<std::mutex> lk(mu_);
  std::size_t n = 0;
  for (bool b : busy_) n += b ? 1 : 0;
  return n;
}

}  // namespace edgevision
