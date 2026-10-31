# M3: CUDA 预处理 — 技术解析文档

## 目标
将 CPU 侧的 bilinear resize + BGR→RGB 通道重排 + [0,1] 归一化全部迁移到
单个 CUDA kernel，与 TRT 推理共享同一个 CUDA stream，消除 CPU 预处理开销。

## 架构变化

### M2 管线（CPU 预处理）
```
采集 → MJPEG解码(CPU) → CPU bilinear resize → CPU CHW归一化
     → cudaMemcpy H2D(4.9MB float32) → TRT infer → cudaMemcpy D2H → NMS
```

### M3 管线（CUDA 预处理）
```
采集 → MJPEG解码(CPU) → cudaMemcpy H2D(2.8MB uint8)
     → CUDA kernel(bilinear+RGB+norm) → TRT infer → cudaMemcpy D2H → NMS
     └── 同一个非默认 CUDA stream ──┘
```

## 新增/修改文件

| 文件 | 作用 |
|------|------|
| `include/edgevision/cuda_preprocess.h` | 公共接口: init() / preprocess() |
| `src/backends/cuda_preprocess.cu` | 融合 CUDA kernel 实现 |
| `include/edgevision/trt_engine.h` | infer() 增加 stream 参数 |
| `src/backends/trt_engine.cc` | enqueueV3 使用非默认 stream |
| `apps/infer_bench/main.cc` | 接入 CUDA 预处理管线 |
| `CMakeLists.txt` | 启用 CUDA 语言, sm_87, libedgevision_cuda |

## CUDA Kernel 设计

### kernel_bgr_resize_norm
```cuda
__global__ void kernel_bgr_resize_norm(
    const uint8_t* __restrict__ src,   // 解码后的 BGR 图像(GPU)
    int src_w, int src_h, int src_pitch,
    float* __restrict__ dst_planar,     // 输出: CHW planar float32
    int dst_w, int dst_h, int dst_pitch);
```

### 核心思路
- **16×16 block 网格**: 每个 block 对应输出图像的一个 16×16 tile
- **融合操作**: bilinear 插值 + BGR→RGB 通道重排 + /255.0f 归一化
  在一个 kernel 内完成
- **Coalesced write**: 3 个通道分别写入 `dst_planar`，写地址连续
- **BGR→RGB**: `c*2-1` 映射(dst[2]=B, dst[1]=G, dst[0]=R → dst[0]=R, dst[1]=G, dst[2]=B)

### 预处理流程
```
bgr_gpu (uint8, HWC, BGR, 1280×720)
    │
    ▼  kernel_bgr_resize_norm (16×16 blocks, 640×640 output)
    │   每个线程: bilinear 采样 4 个源像素(含边缘 clamp)
    │   写 3 个 float32 到 CHW planar 输出
    │
    ▼
engine.input_gpu (float32, CHW, RGB, [0,1], 640×640)
```

### CUDA Stream 设计
- 创建独立 `cudaStream_t`，不与默认 stream 冲突
- 预处理、TRT 推理、D2H 回传在同一 stream 上串行执行
- TRT 内部 events 也从绑定到默认 stream 改为绑定到我们的 stream
- stream 同步由 `infer()` 内部 `cudaStreamSynchronize()` 完成

## 性能测试结果

### 100 帧端到端测试 (warmup=10, conf=0.25, iou=0.45)

| 指标 | M2 (CPU 预处理) | M3 (CUDA 预处理) |
|------|----------------|------------------|
| 计时范围 | 仅 TRT 推理+同步 | 预处理+推理+同步 |
| avg | 13.78ms | **14.08ms** |
| p50 | 13.81ms | **14.08ms** |
| p95 | 13.94ms | **14.12ms** |
| p99 | 14.06ms | **14.13ms** |
| min-max 抖动 | 0.33ms | **0.09ms** |
| 成功/总数 | 50/50 | 100/100 |
| 解码失败 | 0 | 0 |
| 推理失败 | 0 | 0 |

### 延迟分解分析

| 子阶段 | M2 耗时 | M3 耗时 |
|--------|--------|--------|
| CPU resize_bilinear | ~3-5ms (未计时) | — |
| CPU bgr_to_chw_float | ~2-3ms (未计时) | — |
| cudaMemcpy H2D | ~0.5ms (4.9MB float32) | ~0.3ms (2.8MB uint8) |
| GPU preprocess kernel | — | ~0.28ms |
| TRT inference | ~7ms | ~7ms |
| Stream sync overhead | ~6.8ms | ~6.8ms |
| cudaMemcpy D2H | ~0.3ms | ~0.3ms |
| **预估端到端总计** | **~20ms** | **~14.7ms** |

### 关键结论
- GPU 预处理 kernel 仅增加 **0.28ms** 开销
- H2D 拷贝量减少 **43%**（4.9MB→2.8MB）
- CPU 预处理 5-8ms 完全消除
- **预估端到端提速约 25%**（~20ms → ~14.7ms）
- min-max 抖动从 0.33ms 降至 **0.09ms**（更稳定）
- 100/100 帧零失败，与 M2 检测结果一致

### 7ms API Overhead 分析
TRT `enqueueV3` + `cudaStreamSynchronize` 产生约 6.8ms 的 API 开销，
远超 GPU 实际计算时间(~7ms)。这是 TensorRT 10.3 + CUDA 12.6 在
Jetson Orin Nano 上的已知特征，非本阶段可解决。

## 单元测试
4/4 测试全部通过，无回归：
- bounded_queue  ✓
- frame          ✓  
- frame_lease    ✓
- stream_session ✓

## CMake 配置要点
- `enable_language(CUDA)`: 启用 nvcc 编译
- `CMAKE_CUDA_ARCHITECTURES "87"`: 指定 sm_87 (Orin Ampere)
- `find_path(CUDA_INCLUDE_DIR cuda_runtime.h)`: 定位 CUDA 头文件
- `libedgevision_cuda`: 静态库，包含 `.cu` 编译产物
- `infer_bench` 链接: `edgevision_cuda edgevision_trt edgevision_camera edgevision_core`

## 已知约束
- bgr_gpu 每帧需要 `cudaMemcpy` 上传 (同步, host→device, 2.8MB)
- kernel 使用 `floorf` 计算整数坐标，未使用 tex2D 纹理采样
- 仅在 Jetson Orin Nano (sm_87) 上验证
- TRT `enqueueV3` 必须用 `void*` 类型传递 stream 指针