#pragma once
// V4L2Capture: Linux-only MMAP streaming capture. Single owner of camera fd.
// Contract: open -> set_format -> setup_mmap(REQBUFS) -> stream_on ->
//   DQBUF/QBUF loop -> stream_off -> close. No VideoCapture, no hidden threads.
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "edgevision/camera.h"

namespace edgevision {
class V4L2Capture : public Camera {
 public:
  V4L2Capture() = default;
  ~V4L2Capture() override { close(); }
  V4L2Capture(const V4L2Capture&) = delete;
  V4L2Capture& operator=(const V4L2Capture&) = delete;

  bool open(const std::string& device) override;
  bool start(uint32_t w, uint32_t h, uint32_t fourcc) override;
  bool read(Frame& frame, int slot) override;
  void stop() override;
  void close() override;
  bool is_open() const override { return fd_ >= 0; }

  static std::vector<CameraInfo> enumerate();

  // Kernel streaming buffer count (REQBUFS), applied on next start().
  // Too few buffers stall on consumer jitter; too many add pipeline latency.
  void set_buffer_count(uint32_t n) { buffers_ = n ? n : 4; }

  // Why the last open()/start()/read() failed. Empty when it succeeded.
  const std::string& last_error() const { return last_error_; }

  // Format actually granted by the driver. S_FMT may adjust the request, and
  // every downstream byte/size assumption depends on these, not on the request.
  uint32_t negotiated_width() const { return width_; }
  uint32_t negotiated_height() const { return height_; }
  uint32_t negotiated_fourcc() const { return pixfmt_; }

  // Frame interval reported by the driver (VIDIOC_G_PARM). 0 = not reported.
  // Advertised caps routinely overstate this; treat it as the pacing ceiling.
  double actual_fps() const { return actual_fps_; }

 private:
  int fd_ = -1;
  std::string device_;
  struct MmapBuf {
    void* addr = nullptr;
    std::size_t len = 0;
  };
  std::vector<MmapBuf> bufs_;
  uint32_t width_ = 0, height_ = 0, pixfmt_ = 0;
  uint32_t buffers_ = 4;
  double actual_fps_ = 0;
  uint64_t frame_seq_ = 0;
  bool streaming_ = false;
  std::string last_error_;
};
}  // namespace edgevision
