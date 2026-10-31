// V4L2Capture: MMAP streaming. Full ioctl chain, Linux-only.
// (Compiled only when EDGEVISION_ENABLE_V4L2=ON, i.e. Linux builds.)
#include "edgevision/v4l2_capture.h"

#include <fcntl.h>
#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>

namespace edgevision {
namespace {

int xioctl(int fd, unsigned long req, void* arg) {
  int r;
  do {
    r = ioctl(fd, req, arg);
  } while (r == -1 && errno == EINTR);  // restart if signal interrupted
  return r;
}

uint64_t now_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return uint64_t(ts.tv_sec) * 1000000000ull + uint64_t(ts.tv_nsec);
}

}  // namespace

std::vector<CameraInfo> V4L2Capture::enumerate() {
  std::vector<CameraInfo> out;
  for (int i = 0; i < 16; ++i) {
    char dev[32];
    snprintf(dev, sizeof(dev), "/dev/video%d", i);
    int fd = ::open(dev, O_RDWR | O_NONBLOCK);
    if (fd < 0) continue;
    struct v4l2_capability cap;
    memset(&cap, 0, sizeof(cap));
    if (xioctl(fd, VIDIOC_QUERYCAP, &cap) == 0 &&
        (cap.device_caps & V4L2_CAP_VIDEO_CAPTURE)) {
      CameraInfo info;
      info.device = dev;
      info.name = reinterpret_cast<char*>(cap.card);
      info.bus = reinterpret_cast<char*>(cap.bus_info);
      // VID:PID parse from bus_info ("usb-xxxx:..." not always present;
      // authoritative VID:PID comes from lsusb in probe_camera.py).
      out.push_back(std::move(info));
    }
    ::close(fd);
  }
  return out;
}

bool V4L2Capture::open(const std::string& device) {
  close();
  fd_ = ::open(device.c_str(), O_RDWR | O_NONBLOCK);
  if (fd_ < 0) return false;
  device_ = device;
  return true;
}

bool V4L2Capture::start(uint32_t w, uint32_t h, uint32_t fourcc) {
  if (fd_ < 0) return false;
  struct v4l2_format fmt;
  memset(&fmt, 0, sizeof(fmt));
  fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  fmt.fmt.pix.width = w;
  fmt.fmt.pix.height = h;
  fmt.fmt.pix.pixelformat = fourcc;
  fmt.fmt.pix.field = V4L2_FIELD_ANY;
  if (xioctl(fd_, VIDIOC_S_FMT, &fmt) < 0) return false;
  width_ = fmt.fmt.pix.width;
  height_ = fmt.fmt.pix.height;
  pixfmt_ = fmt.fmt.pix.pixelformat;

  // REQBUFS: ask driver for kernel streaming buffers (here 4).
  struct v4l2_requestbuffers req;
  memset(&req, 0, sizeof(req));
  req.count = 4;
  req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  req.memory = V4L2_MEMORY_MMAP;
  if (xioctl(fd_, VIDIOC_REQBUFS, &req) < 0 || req.count < 2) return false;

  bufs_.resize(req.count);
  for (uint32_t i = 0; i < req.count; ++i) {
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;
    if (xioctl(fd_, VIDIOC_QUERYBUF, &buf) < 0) return false;
    bufs_[i].len = buf.length;
    bufs_[i].addr =
        mmap(nullptr, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, buf.m.offset);
    if (bufs_[i].addr == MAP_FAILED) return false;
  }
  // Queue all buffers, then STREAMON.
  for (uint32_t i = 0; i < req.count; ++i) {
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;
    if (xioctl(fd_, VIDIOC_QBUF, &buf) < 0) return false;
  }
  enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  if (xioctl(fd_, VIDIOC_STREAMON, &type) < 0) return false;
  streaming_ = true;
  return true;
}

bool V4L2Capture::read(Frame& frame, int slot) {
  (void)slot;
  if (!streaming_) return false;
  // DQBUF with select() wait (2s cap): block until a frame is ready or timeout.
  fd_set fds;
  FD_ZERO(&fds);
  FD_SET(fd_, &fds);
  struct timeval tv = {2, 0};
  int r = select(fd_ + 1, &fds, nullptr, nullptr, &tv);
  if (r <= 0) return false;  // timeout or signal: caller decides (retry/abort)

  struct v4l2_buffer buf;
  memset(&buf, 0, sizeof(buf));
  buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  buf.memory = V4L2_MEMORY_MMAP;
  if (xioctl(fd_, VIDIOC_DQBUF, &buf) < 0) return false;
  if (buf.index >= bufs_.size()) return false;

  // Copy kernel buffer payload into the pool slot's vector.
  // (M1: zero-copy via refcounted slot is a later optimization; copy first,
  //  measure, then optimize — never prematurely zero-copy with UVC quirks.)
  frame.width = width_;
  frame.height = height_;
  frame.pixfmt_fourcc = pixfmt_;
  frame.ts_ns_monotonic = now_ns();
  frame.data.assign(static_cast<uint8_t*>(bufs_[buf.index].addr),
                    static_cast<uint8_t*>(bufs_[buf.index].addr) + buf.bytesused);
  frame.valid = true;

  // Re-queue the kernel buffer immediately to keep the pipeline full.
  if (xioctl(fd_, VIDIOC_QBUF, &buf) < 0) return false;
  return true;
}

void V4L2Capture::stop() {
  if (fd_ >= 0 && streaming_) {
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(fd_, VIDIOC_STREAMOFF, &type);
    streaming_ = false;
  }
  for (auto& b : bufs_) {
    if (b.addr && b.addr != MAP_FAILED) munmap(b.addr, b.len);
  }
  bufs_.clear();
}

void V4L2Capture::close() {
  stop();
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
}

}  // namespace edgevision
