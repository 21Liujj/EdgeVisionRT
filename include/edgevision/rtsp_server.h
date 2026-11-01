#pragma once
// M10: RTSP live monitor — annotated (boxed) frames -> H.264 -> rtsp://:port/live
//
// Design:
// - Feeds off update_snapshot's already-downscaled+boxed 480px BGR (no extra
//   copy on the inference path; push is a memcpy into appsrc under a drop-
//   oldest policy so a slow viewer NEVER blocks detection).
// - Software x264enc default (ultrafast/zerolatency). Jetson HW NVENC
//   (nvv4l2h264enc) is selected via encoder string when the BSP provides it —
//   one-line change, no code change.
// - GStreamer gst-rtsp-server embedded; no external mediamtx needed.
// - Disabled with port 0. All symbols no-op when EDGEVISION_ENABLE_RTSP=OFF.
#include <cstdint>
#include <string>
#include <vector>

namespace edgevision {

class RtspServer {
 public:
  RtspServer() = default;
  ~RtspServer() { stop(); }

  RtspServer(const RtspServer&) = delete;
  RtspServer& operator=(const RtspServer&) = delete;

  // port 0 = disabled. encoder: "x264" (default) or "nvv4l2h264" (Jetson HW).
  // width/height must match pushed frames (snapshot 480px path).
  bool start(int port, int width, int height, int fps,
             const std::string& encoder = "x264");
  void stop();
  // Thread-safe push of one BGR frame (already boxed). Drops when the
  // viewer is slow; never blocks the caller.
  void push(const std::vector<uint8_t>& bgr, int w, int h);
  bool running() const { return running_; }
  std::string url() const;

 private:
  bool running_ = false;
  int port_ = 0;
  int width_ = 0;
  int height_ = 0;
  // opaque GStreamer state (GstElement*/GstRTSPServer*), pimpl via void*
  void* loop_ = nullptr;
  void* server_ = nullptr;
  void* appsrc_ = nullptr;
  void* bus_watch_ = nullptr;
  void* thread_ = nullptr;
};

}  // namespace edgevision
