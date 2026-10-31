# NanoSAM C++ TensorRT Runtime

Production-ready C++ deployment runtime for NanoSAM image encoder + MobileSAM mask decoder on NVIDIA Jetson AGX Orin.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    EdgePerception v1                         │
│                                                              │
│  RTX 5090 (Training)          Jetson AGX Orin (Deployment) │
│  ┌────────────────┐            ┌──────────────────────────┐ │
│  │ MobileSAM      │────────────>│ Knowledge Distillation  │ │
│  │ Teacher        │  features  │ NanoSAM ResNet18 Student│ │
│  └────────────────┘            └──────────────────────────┘ │
│                                          │                   │
│                                          v                   │
│                                 ┌─────────────────┐         │
│                                 │ 50-epoch .pth   │         │
│                                 └─────────────────┘         │
│                                          │                   │
│                                          v                   │
│                                    ONNX export               │
│                                          │                   │
│                        ┌─────────────────┴────────────────┐ │
│                        v                                  v │
│                 ┌──────────────┐                ┌──────────────┐
│                 │ TensorRT FP16│                │ TensorRT INT8│
│                 │ (Production) │                │  (Research)  │
│                 └──────────────┘                └──────────────┘
│                        │                                  │ │
│                        └─────────────┬────────────────────┘ │
│                                      v                       │
│                          ┌───────────────────────┐          │
│                          │  C++ Runtime (this)   │          │
│                          │  - TensorRT engine    │          │
│                          │  - CUDA buffers       │          │
│                          │  - Box prompt         │          │
│                          │  - Slot0 mask         │          │
│                          └───────────────────────┘          │
└─────────────────────────────────────────────────────────────┘
```

## Components

- **TrtEngine** (`trt_engine.{hpp,cpp}`): TensorRT 10.x engine wrapper with enqueueV3 API, CUDA memory management, RAII resource lifecycle.
- **NanoSamRuntime** (`nanosam_runtime.{hpp,cpp}`): High-level encoder+decoder pipeline matching Python NanoSAM Predictor contract.
- **Preprocessing** (`preprocessing.{hpp,cpp}`): OpenCV-based image preprocessing (letterbox resize, normalization, CHW layout).
- **nanosam_cpp** (`main.cpp`): CLI interface for image-mode and tensor-mode inference.

## Build

```bash
cd cpp_runtime
mkdir -p build && cd build
cmake ..
cmake --build . -j$(nproc)
```

**Requirements:**
- CMake ≥ 3.18
- g++ 11.4+ (C++17)
- TensorRT 10.3.0
- CUDA 12.6
- OpenCV 4.10.0

## Usage

### Image mode (full pipeline)
```bash
./nanosam_cpp \
  --image input.jpg \
  --box 100 150 300 400 \
  --encoder models/self_distilled/epoch50/nanosam_resnet18_epoch049_fp16.engine \
  --decoder third_party/nanosam/data/mobile_sam_mask_decoder.engine \
  --output mask.png
```

### Tensor mode (bypass preprocessing, for correctness verification)
```bash
./nanosam_cpp \
  --input-tensor preprocessed.bin \
  --box 100 150 300 400 \
  --encoder <encoder.engine> \
  --decoder <decoder.engine>
```

The `.bin` file must be a raw FP32 tensor `[1,3,1024,1024]` in CHW layout. Use `--input-tensor` with a `.bin.meta` JSON file containing `{"orig_width": W, "orig_height": H}` for correct postprocessing.

## Frozen Artifacts

### FP16 Encoder (Production Default)
- **Path**: `models/self_distilled/epoch50/nanosam_resnet18_epoch049_fp16.engine`
- **SHA256**: `99147caf00f0174b802848c93928232c02ac996e91e6660372bbc8923112baf4`
- **Quality**: STANDARD500 mIoU = 0.715650, FULL COCO mIoU = 0.699183

### INT8 Encoder (Latency-Critical Research Backend)
- **Path**: `models/self_distilled/epoch50/int8/nanosam_resnet18_epoch049_int8.engine`
- **SHA256**: `d8dd7a2f61427e072baeaf06bb38e78d2dab79c95f41c3839883d5891497d650`
- **Quality**: STANDARD500 mIoU = 0.700264 (−2.15% vs FP16, **FAIL** <0.5% threshold)
- **Speedup**: 1.845× encoder latency vs FP16
- **Status**: Quality gate FAIL → production uses FP16; INT8 available for latency-critical applications willing to accept quality tradeoff.

### Mask Decoder
- **Path**: `third_party/nanosam/data/mobile_sam_mask_decoder.engine`
- **SHA256**: `afb5ac4958e6bab21ed8aa0cc65798f53a71406fe6124cf6b2e008da6fccb4f2`
- **Upstream**: NanoSAM commit `653633614b2eb93b06ba3be9adb2aeffb117bd72`

## Preprocessing Contract

Matches `third_party/nanosam/nanosam/utils/predictor.py::preprocess_image()`:

1. **RGB conversion** (from BGR input)
2. **Letterbox resize**: long edge = 1024, maintain aspect ratio
3. **Normalization**: `(pixel - mean) / std`, mean=`[123.675, 116.28, 103.53]`, std=`[58.395, 57.12, 57.375]`
4. **Layout transform**: HWC → CHW
5. **Zero-padding**: to 1024×1024 square, anchored at top-left
6. **Output**: FP32 `[1, 3, 1024, 1024]`

**Known Delta**: C++ uses `cv::INTER_LINEAR`; Python uses PIL default resample filter (not identical). Measured preprocessing parity: mean relative L2 = 5.1% (see `results/stage_cpp_runtime/preprocess_parity.json`). This delta does NOT affect tensor-mode correctness verification, where both Python and C++ TensorRT runtimes consume identical frozen preprocessed tensors.

## Box Prompt Contract

Follows NanoSAM `predictor.py` convention:
- **Input**: Box in original image pixel coordinates `[x1, y1, x2, y2]`
- **Transformation**: Scale by `1024 / max(orig_width, orig_height)`
- **Decoder input**: Two points `[[x1_scaled, y1_scaled], [x2_scaled, y2_scaled]]` with labels `[2, 3]` (top-left, bottom-right)

## Mask Selection

**Fixed slot 0** (matches frozen project evaluation protocol). Decoder outputs 4 candidate masks; this runtime always selects slot 0, consistent with historical baseline quality measurements.

## Performance (Jetson AGX Orin, FP16)

| Stage | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) |
|-------|-----------|----------|----------|----------|
| **Encoder** | ~36 | ~36 | ~37 | ~38 |
| **Decoder** | ~32 | ~32 | ~33 | ~34 |
| **Full inference** | ~68 | ~68 | ~70 | ~72 |

*(Values from 300-iteration benchmark after 30-iteration warmup; see `results/stage_cpp_runtime/cpp_benchmark.json`)*

Preprocessing (image decode + resize + normalize): ~30 ms (excluded from inference time, disk I/O boundary).

**Python vs C++ runtime ratio**: C++ overhead negligible (within measurement noise); both runtimes execute identical TensorRT engines with similar host-side coordination cost.

## Correctness Verification

### Tensor-Mode Parity (Critical Test)
Python TensorRT vs C++ TensorRT on **identical** frozen preprocessed tensors:
- **Encoder embedding**: max abs diff < 1e-5 (FP32 numerical noise floor)
- **Decoder output**: mask agreement > 99.9%
- **Status**: **PASS** — C++ TensorRT runtime correctness confirmed

### Image-Mode Regression
20-instance regression (first 20 from STANDARD500):
- Python mean IoU: ~0.71
- C++ mean IoU: ~0.70
- Delta attributable to preprocessing resize filter difference (documented above)
- **Status**: **PASS with documented preprocessing delta**

## Project Scope & Limitations

This runtime is an **offline image + box prompt inference engine**, not a real-time camera system.

**In scope (Stage D):**
- ✅ TensorRT C++ engine loading (FP16 + INT8)
- ✅ CUDA buffer management
- ✅ Encoder + decoder inference
- ✅ Box prompt preprocessing
- ✅ Slot0 mask postprocessing
- ✅ CMake build system
- ✅ Correctness regression vs Python baseline
- ✅ Latency benchmarking
- ✅ Resource lifecycle validation

**Out of scope (deferred to future work):**
- ❌ ROS2 integration
- ❌ Camera real-time pipeline
- ❌ NanoOWL text-conditioned detection
- ❌ Multi-GPU / batching
- ❌ Custom CUDA preprocessing kernels
- ❌ QAT / mixed-precision tuning
- ❌ Power profiling / jetson_clocks optimization

Per project scope control (section 0), EdgePerception v1 stops here. Future extensions should build on this frozen baseline.

## Evidence & Validation

All validation artifacts in `results/stage_cpp_runtime/`:
- `environment.json` — toolchain + TensorRT version audit
- `engine_contract.json` — actual I/O tensor names/shapes/dtypes from engines (not guessed)
- `preprocess_parity.json` — C++ vs Python preprocessing delta quantification
- `tensor_mode_parity.json` — TensorRT runtime correctness (Python TRT vs C++ TRT)
- `python_cpp_regression.json` — 20-instance quality regression
- `cpp_benchmark.json` — C++ latency (warmup 30, measured 300)
- `python_benchmark.json` — Python latency (same protocol)
- `int8_backend_smoke.json` — INT8 engine deserialize/run smoke test
- `resource_lifecycle.json` — 1000-run stability check
- `stage_summary.md` — comprehensive Stage D report

## References

- **NanoSAM**: https://github.com/NVIDIA-AI-IOT/nanosam (upstream commit `6536336`)
- **MobileSAM**: https://github.com/ChaoningZhang/MobileSAM
- **TensorRT 10.x API**: https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/
- **COCO 2017**: https://cocodataset.org/

---

**EdgePerception v1 Status**: `EDGE_PERCEPTION_V1_COMPLETE` (pending Stage D final validation)
