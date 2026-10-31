#pragma once
#include <utility>
#include "edgevision/frame.h"

namespace edgevision {

class FrameLease {
 public:
  FrameLease() = default;

  FrameLease(FramePool* pool, int slot)
      : pool_(pool), slot_(slot), epoch_(pool ? pool->epoch_of(slot) : 0) {}

  ~FrameLease() { release(); }

  FrameLease(FrameLease&& other) noexcept
      : pool_(other.pool_), slot_(other.slot_), epoch_(other.epoch_) {
    other.pool_ = nullptr;
    other.slot_ = -1;
    other.epoch_ = 0;
  }

  FrameLease& operator=(FrameLease&& other) noexcept {
    if (this != &other) {
      release();
      pool_ = other.pool_;
      slot_ = other.slot_;
      epoch_ = other.epoch_;
      other.pool_ = nullptr;
      other.slot_ = -1;
      other.epoch_ = 0;
    }
    return *this;
  }

  FrameLease(const FrameLease&) = delete;
  FrameLease& operator=(const FrameLease&) = delete;

  bool valid() const {
    return pool_ != nullptr && slot_ >= 0 &&
           pool_->epoch_of(slot_) == epoch_ &&
           pool_->slot_ref(slot_).valid;
  }

  Frame* operator->() { return pool_ ? &pool_->slot_ref(slot_) : nullptr; }
  Frame& operator*() { return pool_->slot_ref(slot_); }
  const Frame* operator->() const { return pool_ ? &pool_->slot_ref(slot_) : nullptr; }
  const Frame& operator*() const { return pool_->slot_ref(slot_); }

  void release() {
    if (pool_ && slot_ >= 0) {
      pool_->release(slot_);
      pool_ = nullptr;
      slot_ = -1;
      epoch_ = 0;
    }
  }

  int slot() const { return slot_; }

 private:
  FramePool* pool_ = nullptr;
  int slot_ = -1;
  uint64_t epoch_ = 0;
};

}  // namespace edgevision