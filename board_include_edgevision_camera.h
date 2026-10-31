#pragma once
// Camera abstraction: edgevisiond owns the single camera fd.
// Implementations: V4L2 (Linux, real path), Mock/Replay (explicit test path).
#include <cstdint>
#include <string>
#include <vector>

namespace edgevision {

struct CameraFormat {
  uint32_t fourcc = 0;  // V4L2 fourcc, e.g. 'MJPG'
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t fps_num = 1;
  uint32_t fps_den = 30;
  uint64_t est_bandwidth_bps = 0;  // width*height*fps*bpp estimate
};

struct CameraInfo {
  std::string device;      // e.g. /dev/video0
  std::string name;        // card string from VIDIOC_QUERYCAP
  std::string bus;         // bus_info (usb-xxx)
  uint16_t vid = 0;
  uint16_t pid = 0;
  std::vector<CameraFormat> formats;
};

class Camera {
 public:
  virtual ~Camera() = default;
  virtual bool open(const std::string& device) = 0;
  virtual bool start(uint32_t w, uint32_t h, uint32_t fourcc) = 0;
  virtual bool read(class Frame& frame, int slot) = 0;  // fill pool slot
  virtual void stop() = 0;
  virtual void close() = 0;
  virtual bool is_open() const = 0;
};

inline std::string fourcc_to_str(uint32_t f) {
  char s[5] = {char(f & 0xFF), char((f >> 8) & 0xFF), char((f >> 16) & 0xFF),
               char((f >> 24) & 0xFF), 0};
  return std::string(s);
}

}  // namespace edgevision
