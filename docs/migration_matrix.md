# migration_matrix — old repos → EdgeVisionRT

Labels: REUSE-IDEA (principle only, rewrite) / REWORK (adapt to realtime stream) / NEW (no predecessor).

## edge-perception → EdgeVisionRT

| Old path | New home | Label | Note |
|---|---|---|---|
| `cpp_runtime/include/trt_engine.hpp`, `src/trt_engine.cpp` | `backends/trt/` | REUSE-IDEA | RAII engine wrapper, explicit-batch caution; rewrite for YOLOv8-n FP16 stream input |
| `cpp_runtime/include/preprocessing.hpp`, `csrc/cuda_preprocess/cuda_preprocess_kernel.cu` | `cuda/` | REWORK | offline-image kernel → FrameLease-input fused kernel; keep CPU/CUDA parity habit |
| `edge_perception/runtime/cuda_preprocess.py`, `nanosam_optimized.py` | `backends/` workers (M7) | REWORK | snapshot-mode workers only, behind freeze-frame protocol |
| `scripts/bench_cuda_preprocess.py`, `benchmark_video_runtime.py`, `profile_edge_perception.py` | `bench/` | REUSE-IDEA | bench harness shape; metrics redefined for stream (p50/p99, drops) |
| `data/eval_manifests/`, `docs/*parity*` | `tests/data/` + `docs/` parity reports | REUSE-IDEA | manifest + parity-evidence habit, new datasets (live sessions) |
| `scripts/run_edge_perception_video.py` (file input) | `apps/edgevisiond` | REWORK | file source → V4L2 FrameLease source |
| — (no V4L2/pool/protocol/BSP/uvc/long-run) | `src/camera/`, `src/frame/`, `docs/protocol_v3.md`, `kernel/`, `bsp/`, `deploy/` | NEW | the entire lower half: this project's core value |

## edgeqwen-jetson → EdgeVisionRT

| Old path | New home | Label | Note |
|---|---|---|---|
| `scripts/*.sh`, `stage*_driver.sh`, `scripts/reproduce_benchmarks.sh` | `bench/`, `deploy/` | REUSE-IDEA | driver-script reproducibility habit; new targets |
| `docs/final/*` (CLOSURE/FINAL/METRICS/INTERVIEW_BRIEF) | `docs/` stage reports | REUSE-IDEA | evidence-pack discipline |
| TensorRT-Edge-LLM (third-party) | `backends/vlm_worker/` (M7, optional) | REWORK | frozen-frame VQA only, frame_id-bound answers |
| — (no realtime frame → question link) | protocol v3 `snapshot/vqa` ops | NEW | freeze-frame + frame_id binding is new design |

## Non-goals (do NOT migrate)

- GMSL multi-cam topology, PICO WebSocket control, OSS upload path (stay in reference repo).
- Any weight/engine binary: only hashes + manifests cross the boundary.
