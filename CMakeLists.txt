cmake_minimum_required(VERSION 3.22)
project(edgevisionrt VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(EDGEVISION_ENABLE_V4L2 "Build V4L2 capture backend (Linux only)" ON)
option(EDGEVISION_ENABLE_TRT "Build TensorRT inference backend (Linux+CUDA only)" ON)
option(EDGEVISION_BUILD_TESTS "Build unit tests" ON)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(EDGEVISION_ENABLE_V4L2 OFF)
  set(EDGEVISION_ENABLE_TRT OFF)
endif()

# ---- core ----
add_library(edgevision_core STATIC src/frame/frame.cc)
target_include_directories(edgevision_core PUBLIC include)

# ---- camera (V4L2) ----
if(EDGEVISION_ENABLE_V4L2)
  add_library(edgevision_camera STATIC
    src/camera/v4l2_capture.cc
    src/pipeline/stream_session.cc
    src/backends/mjpeg_decoder.cc
  )
  target_include_directories(edgevision_camera PUBLIC include)
  target_link_libraries(edgevision_camera PUBLIC edgevision_core jpeg pthread)
endif()

# ---- RTSP live monitor (GStreamer appsrc -> H.264 -> rtsp) ----
option(EDGEVISION_ENABLE_RTSP "Build RTSP live monitor (GStreamer)" ON)
if(EDGEVISION_ENABLE_RTSP AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
  find_package(PkgConfig QUIET)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-rtsp-server-1.0
                      gstreamer-app-1.0)
  endif()
  if(GST_FOUND)
    add_library(edgevision_stream STATIC src/stream/rtsp_server.cc)
    target_include_directories(edgevision_stream PUBLIC include ${GST_INCLUDE_DIRS})
    target_compile_definitions(edgevision_stream PRIVATE EDGEVISION_ENABLE_RTSP_IMPL)
    target_link_libraries(edgevision_stream PUBLIC ${GST_LIBRARIES})
    message(STATUS "RTSP monitor: ON (${GST_LIBRARIES})")
  else()
    message(STATUS "RTSP monitor: OFF (GStreamer not found)")
  endif()
endif()

# ---- TensorRT backend ----
if(EDGEVISION_ENABLE_TRT)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_ARCHITECTURES "87")
  find_library(NVINFER_LIB nvinfer PATHS /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64)
  find_library(NVINFER_PLUGIN_LIB nvinfer_plugin PATHS /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64)
  find_library(CUDA_LIB cuda PATHS /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64 /usr/local/cuda/lib64/stubs)
  find_library(CUDART_LIB cudart PATHS /usr/lib/aarch64-linux-gnu /usr/local/cuda/lib64)
  find_path(CUDA_INCLUDE_DIR cuda_runtime.h PATHS /usr/local/cuda/include /usr/local/cuda-12/include /usr/local/cuda/targets/aarch64-linux/include)

  add_library(edgevision_trt STATIC src/backends/trt_engine.cc)
  target_include_directories(edgevision_trt PUBLIC include ${CUDA_INCLUDE_DIR})
  target_link_libraries(edgevision_trt PUBLIC
    ${NVINFER_LIB} ${NVINFER_PLUGIN_LIB}
    ${CUDA_LIB} ${CUDART_LIB}
  )

  add_library(edgevision_cuda STATIC src/backends/cuda_preprocess.cu)
  target_include_directories(edgevision_cuda PUBLIC include ${CUDA_INCLUDE_DIR})
  target_link_libraries(edgevision_cuda PUBLIC ${CUDART_LIB})
endif()

# ---- apps ----
add_executable(capture_probe apps/capture_probe/main.cc)
target_link_libraries(capture_probe PRIVATE edgevision_core
  $<$<BOOL:${EDGEVISION_ENABLE_V4L2}>:edgevision_camera>)

if(EDGEVISION_ENABLE_V4L2 AND EDGEVISION_ENABLE_TRT)
  add_executable(infer_bench apps/infer_bench/main.cc)
  target_link_libraries(infer_bench PRIVATE edgevision_core edgevision_camera edgevision_trt edgevision_cuda)

  add_executable(edgevisiond apps/edgevisiond/main.cc)
  target_link_libraries(edgevisiond PRIVATE edgevision_core edgevision_camera edgevision_trt edgevision_cuda
    $<$<TARGET_EXISTS:edgevision_stream>:edgevision_stream>)

  # file_infer: replay images through the SAME preprocess+infer+NMS (no V4L2).
  # Output goes to replay_*/ (explicitly NOT sessions/): replay never counts
  # as on-device acceptance.
  add_executable(file_infer apps/file_infer/main.cc)
  target_link_libraries(file_infer PRIVATE edgevision_core edgevision_camera edgevision_trt edgevision_cuda)
endif()

# ---- tests ----
if(EDGEVISION_BUILD_TESTS)
  enable_testing()

  add_executable(test_bounded_queue tests/unit/test_bounded_queue.cc)
  target_link_libraries(test_bounded_queue PRIVATE edgevision_core)
  add_test(NAME bounded_queue COMMAND test_bounded_queue)

  add_executable(test_frame tests/unit/test_frame.cc)
  target_link_libraries(test_frame PRIVATE edgevision_core)
  add_test(NAME frame COMMAND test_frame)

  add_executable(test_frame_lease tests/unit/test_frame_lease.cc)
  target_link_libraries(test_frame_lease PRIVATE edgevision_core)
  add_test(NAME frame_lease COMMAND test_frame_lease)

  if(EDGEVISION_ENABLE_V4L2)
    add_executable(test_stream_session tests/unit/test_stream_session.cc)
    target_link_libraries(test_stream_session PRIVATE edgevision_core edgevision_camera)
    add_test(NAME stream_session COMMAND test_stream_session)
  endif()
endif()
