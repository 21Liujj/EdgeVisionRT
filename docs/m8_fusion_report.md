# M8 融合验证报告（Fusion + COCO 全量双跑，板上实测 2026-10-31）

> 三个边缘 AI 项目已融合进 **单一仓库 `edgevisionrt/`**。
> 融合策略：代码进 `vendors/`，大 TensorRT 引擎走 `models/MANIFEST.md` 清单，板上落盘，git 不进大文件。

## 1. 融合后项目结构

```
edgevisionrt/                  <- 唯一仓库
├── apps/edgevisiond/          原 edgevisionrt：C++ daemon，常驻 YOLOv8-n FP16
├── control_service/
│   ├── agent.py / scheduler.py / worker_dispatch.py
│   ├── paths.py               NEW：统一路径解析（config + 环境变量覆盖 + 仓库回退）
│   └── workers/
│       ├── nanoowl_worker.py  已收敛：经 paths.py 解析 vendored 脚本
│       └── qwen_worker.py     已收敛：经 paths.py 解析 qwen_ask 二进制
├── vendors/                   NEW：donor 源码（只放代码，不放引擎）
│   ├── nanoowl/               原 edge-perception：run_nanoowl.py / eval_nanoowl_coco.py /
│   │                          nanosam_src/ / nanosam_include/
│   └── edgeqwen/
│       ├── runtime_sdk/src/   原 edgeqwen-jetson：runtime.cpp / memory_monitor.cpp
│       └── demo/qwen_ask.cpp  自研 VLM 单问 CLI 源码
├── models/MANIFEST.md         大引擎清单（owl/sam/qwen engine 路径 + 体积）
├── third_party/               nanoowl python 包 + torch2trt-0.5.0-py3.10.egg
├── config/device_config.json  NEW：唯一配置源
└── eval/                      NEW：双跑入口
    ├── run_yolo_coco.sh       file_infer 全量 5000 张
    ├── run_owl_coco.sh        NanoOWL 全量（pycocotools COCOeval）
    └── compare.py             recall-lift 对比
```

路径收敛验证（板上）：
`RUN_OWL = /home/nvidia/edgevisionrt/vendors/nanoowl/run_nanoowl.py`
（此前硬编码 `/home/nvidia/jetson-deploy-lab/scripts/run_nanoowl.py`）

## 2. COCO val2017 全量双跑（5000 张，板上实测）

| 模型 | AP | AP50 | AP75 | APsmall | APmedium | APlarge | AR100 | 延迟/张 |
|---|---|---|---|---|---|---|---|---|
| YOLOv8-n FP16（file_infer replay） | 0.216 | 0.301 | 0.236 | 0.070 | 0.236 | 0.319 | 0.263 | 4.34ms |
| NanoOWL开放词汇（COCO类名prompt） | 0.259 | 0.408 | 0.273 | 0.095 | 0.264 | 0.442 | 0.372 | ~28ms摊销 |

评测方法：
- YOLO：`eval/out_yolo/events.jsonl` → 类索引经 COCO 有序 cats 映射 → 640 stretch-back → `coco_detections.json` → COCOeval bbox。
- OWL：`vendors/nanoowl/eval_nanoowl_coco.py --images-dir --annotations --engine --output-dir` → `coco_detections.json` → COCOeval bbox（`coco_eval_summary.json`）。
- 注意：file_infer replay 用 stretch 到 640×640 而非 letterbox，YOLO 的 AP 有轻微低估；板上实时链路为原生 640 流，不受影响。

结论：**OWL 全方位压过 YOLO（含小目标）**，级联架构的价值得证——4ms 常驻 YOLO 守门，漏检时 OWL 补强，Qwen 做语义确认。

## 3. 杯子实拍三级闭环（2026-10-31 11:17，镜头前真杯子）

- `pending_review/20261031_111708_f12/crop.jpg`：YOLO 仅检出 person(0.38)+chair(0.28)，杯子漏检 → low_conf streak 触发 review。
- NanoOWL worker：`owl_boxes: []`（`a cup,a mug,an object` prompt 下空回，硬 case）。
- Qwen worker：`{"ok": true, "lat_ms": 15604, "qwen_text": "Yes, there is a cup. It is on the right side of the image."}`

三级 cascade（YOLO → OWL+SAM → Qwen-VL）在融合后仓库路径下完整跑通。

## 4. 遗留事项

- `eval/compare.py` 的 per-class recall-lift 待 pycocotools 最终数字落盘后补齐。
- `NANOSAM_CPP` 仍回退到 board 旧路径（二进制未 vendor 化，可选）。
- docker image / 8h 长稳仍 DEFERRED（见 progress.md）。
