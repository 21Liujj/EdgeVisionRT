// M10: RTSP live monitor implementation.
//
// Pipeline (built once in start()):
//   appsrc(name=evsrc, BGR, w×h@fps) -> videoconvert -> videoscale ->
//   x264enc(speed-preset=ultrafast,tune=zerolatency,bitrate=800,key-int-max=fps)
//     -> rtph264pay(name=pay0,pt=96)  [served by gst-rtsp-server mount /live]
//
// HW path (encoder=="nvv4l2h264", Jetson BSP with encoder plugin):
//   appsrc -> videoconvert -> nvvideoconvert -> nvv4l2h264enc(bitrate,preset-level=1)
//     -> h264parse -> rtph264pay
//
// Backpressure: appsrc max-buffers=2 + drop=true => slow viewer drops,
// detection thread never blocks (push returns immediately).
#ifdef EDGEVISION_ENABLE_RTSP_IMPL
#include "edgevision/rtsp_server.h"

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/rtsp-server/rtsp-server.h>

#include <chrono>
#include <cstring>
#include <mutex>
#include <thread>

namespace edgevision {
namespace {

std::mutex g_mu;  // guards appsrc push (GStreamer appsrc is thread-safe, belt & suspenders)

gboolean bus_msg(GstBus*, GstMessage* msg, gpointer) {
  if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
    GError* err = nullptr;
    gchar* dbg = nullptr;
    gst_message_parse_error(msg, &err, &dbg);
    g_printerr("[rtsp] ERROR: %s (%s)\n", err ? err->message : "?",
               dbg ? dbg : "");
    if (err) g_error_free(err);
    if (dbg) g_free(dbg);
  }
  return TRUE;
}

}  // namespace

bool RtspServer::start(int port, int width, int height, int fps,
                       const std::string& encoder) {
  if (port <= 0 || width <= 0 || height <= 0 || fps <= 0) return false;
  if (running_) return true;
  port_ = port;
  width_ = width;
  height_ = height;

  gst_init(nullptr, nullptr);
  GMainLoop* loop = g_main_loop_new(nullptr, FALSE);
  GstRTSPServer* server = gst_rtsp_server_new();
  g_object_set(server, "service", std::to_string(port).c_str(), nullptr);
  GstRTSPMountPoints* mounts = gst_rtsp_server_get_mount_points(server);
  GstRTSPMediaFactory* factory = gst_rtsp_media_factory_new();
  gst_rtsp_media_factory_set_shared(factory, TRUE);

  // Encoder branch: HW NVENC when requested AND available, else x264 SW.
  std::string enc;
  if (encoder == "nvv4l2h264" &&
      gst_registry_find_feature(gst_registry_get(), "nvv4l2h264enc", GST_TYPE_ELEMENT_FACTORY) != nullptr) {
    g_print("[rtsp] using Jetson HW encoder nvv4l2h264enc\n");
    enc = "nvvideoconvert ! nvv4l2h264enc bitrate=800000 preset-level=1 insert-sps-pps=true ! h264parse ! ";
  } else {
    if (encoder == "nvv4l2h264")
      g_print("[rtsp] nvv4l2h264enc not found, falling back to x264enc\n");
    enc = "x264enc speed-preset=ultrafast tune=zerolatency bitrate=800 "
          "key-int-max=" + std::to_string(fps) +
          " ! video/x-h264,profile=baseline ! ";
  }

  std::string launch =
      "( appsrc name=evsrc is-live=true format=time do-timestamp=true "
      "max-buffers=2 drop=true "
      "caps=video/x-raw,format=BGR,width=" + std::to_string(width) +
      ",height=" + std::to_string(height) +
      ",framerate=" + std::to_string(fps) +
      "/1 ! videoconvert ! videoscale ! video/x-raw,width=" +
      std::to_string(width) + ",height=" + std::to_string(height) +
      " ! " + enc + "rtph264pay name=pay0 pt=96 )";
  gst_rtsp_media_factory_set_launch(factory, launch.c_str());
  gst_rtsp_mount_points_add_factory(mounts, "/live", factory);
  g_object_unref(mounts);

  // appsrc handle: grabbed when media is prepared (first client or preroll).
  // Simpler robust approach: keep our own pipeline ref via "media-configure".
  struct Ctx {
    GstElement* appsrc = nullptr;
  };
  Ctx* ctx = new Ctx();
  g_signal_connect(factory, "media-configure",
                   G_CALLBACK(+[](GstRTSPMediaFactory*, GstRTSPMedia* media, gpointer ud) {
                     Ctx* c = static_cast<Ctx*>(ud);
                     GstElement* el = gst_rtsp_media_get_element(media);
                     c->appsrc = gst_bin_get_by_name(GST_BIN(el), "evsrc");
                     gst_object_unref(el);
                   }),
                   ctx);

  guint id = gst_rtsp_server_attach(server, nullptr);
  if (id == 0) {
    delete ctx;
    g_object_unref(server);
    g_main_loop_unref(loop);
    return false;
  }
  // Bus watch for encoder errors (pipeline lives inside the media).
  GstBus* bus = nullptr;  // media bus observed via factory signals; keep simple

  // Run loop on its own thread; push() feeds appsrc when clients connect.
  std::thread* th = new std::thread([loop]() { g_main_loop_run(loop); });

  loop_ = loop;
  server_ = server;
  appsrc_ = ctx;  // actually Ctx*; resolved below on push
  thread_ = th;
  running_ = true;
  g_print("[rtsp] live on rtsp://0.0.0.0:%d/live (%dx%d@%dfps)\n", port, width,
          height, fps);
  (void)bus;
  (void)bus_watch_;
  return true;
}

void RtspServer::stop() {
  if (!running_) return;
  running_ = false;
  if (loop_) g_main_loop_quit(static_cast<GMainLoop*>(loop_));
  if (thread_) {
    static_cast<std::thread*>(thread_)->join();
    delete static_cast<std::thread*>(thread_);
    thread_ = nullptr;
  }
  // server/loop unref; Ctx freed with them (appsrc owned by pipeline)
  if (server_) {
    g_object_unref(static_cast<GstRTSPServer*>(server_));
    server_ = nullptr;
  }
  if (loop_) {
    g_main_loop_unref(static_cast<GMainLoop*>(loop_));
    loop_ = nullptr;
  }
  if (appsrc_) {
    // Ctx wrapper (appsrc element itself owned by pipeline)
    // leak-safe: small, freed here
    struct Ctx {
      void* appsrc;
    };
    delete static_cast<Ctx*>(appsrc_);
    appsrc_ = nullptr;
  }
}

void RtspServer::push(const std::vector<uint8_t>& bgr, int w, int h) {
  if (!running_ || w != width_ || h != height_) return;
  struct Ctx {
    GstElement* appsrc;
  };
  Ctx* ctx = static_cast<Ctx*>(appsrc_);
  if (!ctx || !ctx->appsrc) return;  // no client yet / media not configured
  std::lock_guard<std::mutex> lk(g_mu);
  GstBuffer* buf = gst_buffer_new_allocate(nullptr, bgr.size(), nullptr);
  GstMapInfo map;
  if (gst_buffer_map(buf, &map, GST_MAP_WRITE)) {
    std::memcpy(map.data, bgr.data(), bgr.size());
    gst_buffer_unmap(buf, &map);
    GstFlowReturn ret = gst_app_src_push_buffer(
        GST_APP_SRC(ctx->appsrc), buf);
    (void)ret;  // drop=true => OK to ignore
  } else {
    gst_buffer_unref(buf);
  }
}

std::string RtspServer::url() const {
  if (!running_) return "";
  return "rtsp://0.0.0.0:" + std::to_string(port_) + "/live";
}

}  // namespace edgevision
#else
// RTSP disabled at build time: header-only no-ops (link-safe).
#include "edgevision/rtsp_server.h"
namespace edgevision {
bool RtspServer::start(int, int, int, int, const std::string&) { return false; }
void RtspServer::stop() {}
void RtspServer::push(const std::vector<uint8_t>&, int, int) {}
std::string RtspServer::url() const { return ""; }
}  // namespace edgevision
#endif
