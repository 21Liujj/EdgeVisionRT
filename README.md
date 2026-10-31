# EdgeVisionRT — Jetson Orin Nano 端侧三级级联感知系统

> 单 Jetson Orin Nano Super 8GB + 单 USB 摄像头，完成 **4ms 常驻检测 → 开放词汇补检 → VLM 语义确认** 的三级级联。
> 融合三个边缘 AI 项目为单一仓库，COCO val2017 全量实测，Docker 单容器交付。
>
> [English](./README_EN.md) | **中文**

## 一句话

在 8GB 统一内存的 Jetson 上，让 YOLOv8-n（4.3ms）常驻守门、NanoOWL（开放词汇）补漏检、Qwen2-VL（2B）做语义确认——三级串行、一次只活一个模型，不爆显存。

## 架构

```
USB 摄像头 → edgevisiond (C++, YOLOv8-n FP16 常驻, 4.3ms/帧)
   │ 低置信/漏检 (streak>=3)
   ▼ pending_review/<stamp>/ (crop.jpg + meta.json, 原子写入)
agent.py → nanoowl_worker (子进程, 用完即退)
         → qwen_worker → qwen_ask (子进程, 用完即退)
```

```text
edgevisionrt/                  ← 唯一仓库
├── apps/edgevisiond/          C++ daemon：常驻 YOLOv8-n FP16 + review 触发
├── control_service/           Python 调度：agent / scheduler / worker_dispatch
│   ├── paths.py               统一路径解析（config + 环境变量 + 仓库回退）
│   └── workers/               nanoowl_worker / qwen_worker（临时子进程）
├── vendors/                   融合进来的 donor 源码（只放代码，不放引擎）
│   ├── nanoowl/               run_nanoowl.py / eval_nanoowl_coco.py / nanosam C++ 源码
│   └── edgeqwen/              runtime.cpp / memory_monitor / qwen_ask.cpp
├── models/MANIFEST.md         大 TensorRT 引擎清单（不进 git，板上落盘）
├── third_party/               nanoowl python 包 + torch2trt-0.5.0-py3.10.egg
├── config/device_config.json  唯一配置源
├── eval/                      COCO 双跑入口（run_yolo_coco.sh / run_owl_coco.sh / compare.py）
└── deploy/                    Dockerfile + entrypoint.sh + run_board.sh（单容器）
```

**设计原则**：代码进仓库（`vendors/`），大引擎走 MANIFEST 清单管理；单容器单 CUDA 上下文，worker 串行子进程，用完即退。

## 实测数据（板上，Jetson Orin Nano Super，MAXN_SUPER）

COCO val2017 全量 5000 张：

| 模型 | AP | AP50 | APsmall | 延迟/张 |
|---|---|---|---|---|
| YOLOv8-n FP16（常驻） | 0.216 | 0.301 | 0.070 | 4.34ms |
| NanoOWL 开放词汇（补检） | 0.259 | 0.408 | 0.095 | ~28ms 摊销 |

杯子实拍三级闭环（镜头前真杯子）：YOLO 漏检 → OWL 空回（硬 case）→ Qwen 正确回答 "Yes, there is a cup. It is on the right side of the image."

## 快速开始（板上）

```bash
# 1. 原生运行
cmake --preset board-native && cmake --build build/board
./build/board/edgevisiond --engine models/yolov8n_fp16.engine \
  --device /dev/video-edgevision --sock /tmp/edgevisiond.sock \
  --sessions ./sessions --http 8080
python3 control_service/agent.py --sock /tmp/edgevisiond.sock

# 2. Docker 单容器（推荐）
docker build -f deploy/Dockerfile -t edgevisionrt:board .
bash deploy/run_board.sh   # CDI GPU + 引擎/会话挂载，daemon+agent 同容器

# 3. COCO 双跑评测
bash eval/run_yolo_coco.sh   # YOLO 全量
bash eval/run_owl_coco.sh    # NanoOWL 全量（需 pycocotools）
```

## 文档

- 进度唯一真相源：`docs/progress.md`
- 融合验证报告：`docs/m8_fusion_report.md`
- 各里程碑技术分析：`docs/m1~m7_tech_analysis.md`

## 工程约束

- `sent != accepted != completed`：每个 start/stop/record 操作三态分离
- 禁止 VideoCapture 走捷径、禁止固定 sleep 等设备、禁止伪造推理结果
- Mock/Replay 必须显式声明，不计入板端验收
