# EdgeVisionRT — 3-Stage Cascaded Perception on Jetson Orin Nano

> Single Jetson Orin Nano Super 8GB + single USB camera: **4ms persistent detection → open-vocabulary补检 → VLM semantic confirmation**.
> Three edge-AI projects fused into one repo, evaluated on full COCO val2017, shipped as one Docker container.

[中文版](./README.md) | **English**

## In One Sentence

On a Jetson with 8GB unified memory, YOLOv8-n (4.3ms) stands guard, NanoOWL (open-vocabulary) recovers misses, and Qwen2-VL (2B) confirms semantics — three stages in series, only one model alive at a time, no VRAM blowup.

## Architecture

```
USB camera → edgevisiond (C++, persistent YOLOv8-n FP16, 4.3ms/frame)
   │ low-conf / miss (streak>=3)
   ▼ pending_review/<stamp>/ (crop.jpg + meta.json, atomic write)
agent.py → nanoowl_worker (throwaway subprocess, exits when done)
         → qwen_worker → qwen_ask (throwaway subprocess, exits when done)
```

```text
edgevisionrt/                  ← single repo
├── apps/edgevisiond/          C++ daemon: persistent YOLOv8-n FP16 + review trigger
├── control_service/           Python dispatch: agent / scheduler / worker_dispatch
│   ├── paths.py               Unified path resolution (config + env + repo fallback)
│   └── workers/               nanoowl_worker / qwen_worker (temp subprocesses)
├── vendors/                   Fused donor sources (code only, no engines)
│   ├── nanoowl/               run_nanoowl.py / eval_nanoowl_coco.py / NanoSAM C++ sources
│   └── edgeqwen/              runtime.cpp / memory_monitor / qwen_ask.cpp
├── models/MANIFEST.md         Large TensorRT engine manifest (not in git, on-board)
├── third_party/               nanoowl python pkg + torch2trt-0.5.0-py3.10.egg
├── config/device_config.json  Single source of config
├── eval/                      COCO dual-run harness (run_yolo_coco.sh / run_owl_coco.sh / compare.py)
└── deploy/                    Dockerfile + entrypoint.sh + run_board.sh (single container)
```

**Design principle**: code goes into the repo (`vendors/`), large engines are manifest-managed; one container, one CUDA context, serial worker subprocesses that exit when done.

## Measured Numbers (on-board, Jetson Orin Nano Super, MAXN_SUPER)

Full COCO val2017, 5000 images:

| Model | AP | AP50 | APsmall | Latency/img |
|---|---|---|---|---|
| YOLOv8-n FP16 (persistent) | 0.216 | 0.301 | 0.070 | 4.34ms |
| NanoOWL open-vocab (recovery) | 0.259 | 0.408 | 0.095 | ~28ms amortized |

Live cup test (real cup in front of the lens): YOLO missed → OWL empty (hard case) → Qwen correctly answered "Yes, there is a cup. It is on the right side of the image."

## Quick Start (on-board)

```bash
# 1. Native run
cmake --preset board-native && cmake --build build/board
./build/board/edgevisiond --engine models/yolov8n_fp16.engine \
  --device /dev/video-edgevision --sock /tmp/edgevisiond.sock \
  --sessions ./sessions --http 8080
python3 control_service/agent.py --sock /tmp/edgevisiond.sock

# 2. Docker single container (recommended)
docker build -f deploy/Dockerfile -t edgevisionrt:board .
bash deploy/run_board.sh   # CDI GPU + engine/session mounts, daemon+agent in one container

# 3. COCO dual-run eval
bash eval/run_yolo_coco.sh   # YOLO full set
bash eval/run_owl_coco.sh    # NanoOWL full set (needs pycocotools)
```

## Docs

- Progress (single source of truth): `docs/progress.md`
- Fusion report: `docs/m8_fusion_report.md`
- Milestone analyses: `docs/m1~m7_tech_analysis.md`

## Engineering Rules

- `sent != accepted != completed` for every start/stop/record op.
- No VideoCapture shortcuts, no fixed sleeps for device state, no fake inference results.
- Mock/Replay must be explicit and never counts as on-device acceptance.
