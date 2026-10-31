# M2: YOLOv8-n FP16 推理管线 · 技术解析

**Date**: 2026-10-28  
**Target**: Jetson Orin Nano Super (aarch64, L4T R36.4.3, JetPack 6.2)  
**Status**: ✅ 完成

---

## 1. 架构概览

M2 阶段在 M1（V4L2 + 帧池 + 生产者-消费者管道）之上，堆叠了推理层：

```
Camera → StreamSession ──→ infer_bench
               │                │
               ▼                ▼
          [MJPEG帧]    mjpeg_decode_to_bgr()
                              │
                              ▼
                      resize_bilinear(640×640)
                              │
                              ▼
                      bgr_to_chw_float([0,1])
                              │
                              ▼
                     cudaMemcpy → GPU
                              │
                              ▼
                     TrtEngine::infer() 7.05ms
                              │
                              ▼
                     cudaMemcpy → CPU
                              │
                              ▼
                     postprocess() → dets[]
```

---

## 2. 新增模块

### 2.1 TrtEngine (RAII TensorRT 封装)

**文件**: `include/edgevision/trt_engine.h` + `src/backends/trt_engine.cc`

```cpp
class TrtEngine {
  bool load(const std::string& engine_path);  // 反序列化引擎文件
  bool prepare_io();                          // 分配 GPU I/O 内存
  bool infer();                               // 同步推理 (enqueueV3 + sync)
  void* input_gpu() const;                    // GPU 输入指针
  void* output_gpu() const;                   // GPU 输出指针
  const Dims& input_dims() const;             // e.g. [1, 3, 640, 640]
  const Dims& output_dims() const;            // e.g. [1, 84, 1, 8400]
};
```

设计决策：
- **无拷贝**：`input_gpu()` / `output_gpu()` 直接返回 GPU 指针，由调用方 `cudaMemcpy` 自行管理传输
- **RAII 析构**：自动释放 CUDA 内存、context、engine、runtime
- **禁止拷贝**：`= delete` 拷贝构造/赋值，防止 double-free
- **单流同步推理**：M2 阶段使用默认流 `enqueueV3` + `cudaStreamSynchronize`；M3 将升级为非默认流

### 2.2 MJPEG 解码器 (libjpeg-turbo)

**文件**: `include/edgevision/mjpeg_decoder.h` + `src/backends/mjpeg_decoder.cc`

```cpp
bool mjpeg_decode_to_bgr(const uint8_t* jpeg_data, std::size_t jpeg_size,
                         std::vector<uint8_t>& bgr_out, int& width, int& height,
                         std::string& error);
```

设计决策：
- **零外部依赖**：仅用系统 `libjpeg-turbo 2.1.5`，不依赖 OpenCV
- **输出 BGR888**：与 OpenCV 一致，方便后续对比验证
- **异常安全**：使用 `setjmp/longjmp` 捕获 libjpeg 致命错误，不会崩溃
- **线程安全**：无状态函数，可多线程并发调用

### 2.3 infer_bench 应用

**文件**: `apps/infer_bench/main.cc`

```bash
./build/board/infer_bench --engine models/yolov8n_fp16.engine \
                          --device /dev/video0 --frames 50 --warmup 10
```

处理管线的每个环节：
1. `StreamSession::consume()` 获取一帧 MJPEG 数据
2. `mjpeg_decode_to_bgr()` JPEG → BGR uint8
3. `resize_bilinear()` BGR 1280×720 → 640×640
4. `bgr_to_chw_float()` HWC uint8 → CHW float32 [0,1] 归一化
5. `cudaMemcpy` CPU → GPU
6. `TrtEngine::infer()` 同步推理
7. `cudaMemcpy` GPU → CPU
8. `postprocess()` 置信度过滤 + NMS

### 2.4 模型导出与构建

**ONNX 导出** (`/tmp/m2_export_onnx.py`):
```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=640, opset=12, simplify=True)
```

**TensorRT FP16 引擎** (trtexec CLI):
```bash
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_fp16.engine --fp16
```

| 指标 | 值 |
|------|-----|
| ONNX 文件大小 | 12.3 MB |
| FP16 引擎大小 | 9.1 MB |
| 构建时间 | ~10 分钟 |
| 构建 GPU 内存 | 6477 MiB / 7619 MiB |

---

## 3. 完整测试结果

### 3.1 单元测试 (4/4 PASS)

```
Test #1: bounded_queue  ....   Passed    0.01 sec
Test #2: frame  ............   Passed    0.00 sec
Test #3: frame_lease  ......   Passed    0.00 sec
Test #4: stream_session  ...   Passed    3.07 sec
```

M1 原有测试在新增 M2 代码后全部通过，无回归。

### 3.2 trtexec 独立推理

```
GPU Latency:      7.05 ms   (pure compute)
Host Latency:     7.82 ms   (compute + enqueue + D2H)
H2D Latency:      0.45 ms
D2H Latency:      0.32 ms
Throughput:     141.4 QPS  (batch=1, ideal)
```

### 3.3 infer_bench 端到端

```
Engine:         yolov8n_fp16.engine [1x3x640x640] → [1x84x8400]
Camera:         /dev/video0, 1280x720 MJPEG, 24.9fps (实测)
Frames:         50 帧测试 + 10 帧暖机
Result:         50/50 成功, 0 解码失败, 0 推理失败

延迟统计:
  avg = 13.78ms   (完整管线，含解码+resize+GPU拷贝+推理)
  p50 = 13.76ms
  p95 = 13.98ms
  p99 = 14.06ms
  min = 13.69ms
  max = 14.06ms

检测统计:
  总检测数 = 116, 每帧平均 = 2.3 个目标
```

### 3.4 延迟分解

| 阶段 | 耗时 | 说明 |
|------|------|------|
| MJPEG 解码 | ~1-2ms | libjpeg-turbo on ARM |
| bilinear resize 1280→640 | ~2-3ms | CPU scalar |
| HWC→CHW + normalize | ~2-3ms | CPU scalar |
| cudaMemcpy H2D | ~0.45ms | 640×640×3×4B |
| GPU 推理 | 7.05ms | TRT FP16 |
| cudaMemcpy D2H | ~0.32ms | 84×8400×4B |
| NMS 后处理 | ~0.3ms | CPU |

总额 ~13.78ms，GPU 推理占约 51%。

---

## 4. 已知约束

| 项目 | 约束 | 解决方案 |
|------|------|----------|
| TRT 默认流 | 使用 default stream 有 sync 开销 | M3 升级为非默认流 |
| CPU 预处理 | resize + normalize 在 CPU 上运行 | M3 用 CUDA kernel |
| 静态 batch=1 | 引擎不支持动态 batch | M3 考虑动态 shape |
| numpy 兼容 | numpy>=2 与 ml_dtypes 冲突 | 降级到 numpy 1.26.4 |
| 编译依赖 | 需要 `/usr/local/cuda/include` 路径 | CMake 已自动 find_path |

---

## 5. 编译系统

**CMakeLists.txt** 更新内容：
- 新增 `edgevision_trt` 静态库（链接 nvinfer / nvinfer_plugin / cuda / cudart）
- 新增 `edgevision_camera` 包含 `mjpeg_decoder.cc`
- 新增 `infer_bench` 可执行程序
- 自动 `find_path(CUDA_INCLUDE_DIR cuda_runtime.h)` 查找 CUDA header
- 移除 `CUDA` 语言（无 .cu 文件，仅链接 cuda runtime）

**构建命令**:
```bash
cmake --preset board-native && cmake --build build/board -j 4
# 输出: 15/15 targets, 全部成功
```

---

## 6. M2 → M3 路线

M3 阶段将把 CPU 预处理迁移到 CUDA：
1. CUDA kernel: bilinear resize BGR→RGB planar float
2. CUDA kernel: normalize (/255.0) 融合到 resize
3. 非默认 CUDA stream 避免同步开销
4. 预期端到端延迟降至 ~10ms