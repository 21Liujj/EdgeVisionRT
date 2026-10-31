# models/MANIFEST.md — 大文件引擎清单（不进 git，板上部署校验）

> 原则：仓库只放**源码+清单**，`.engine`（GB 级）永不进 git。
> 板上部署脚本按此清单校验存在性 + sha256，缺失即报错。

| 模型 | 板上路径 | 大小 | 用途 | 来源 |
|---|---|---|---|---|
| `yolov8n_fp16.engine` | `edgevisionrt/models/` | ~12MB | YOLO 常驻推理 | 本仓库 CI 导出（见 `scripts/m2_build_model.py`） |
| `owlvit_image_encoder_patch32.engine` | `~/jetson-deploy-lab/third_party/nanoowl/data/` | ~184MB | OWL-ViT 图像编码器 FP16 | EdgePerception Day 1 导出 |
| `resnet18_image_encoder.engine` | `~/jetson-deploy-lab/third_party/nanosam/data/` | ~35MB | NanoSAM encoder FP16 | EdgePerception Day 3 导出 |
| `mobile_sam_mask_decoder.engine` | `~/jetson-deploy-lab/third_party/nanosam/data/` | ~19MB | NanoSAM decoder FP16 | EdgePerception Day 3 导出 |
| `llm.engine` | `~/edgeqwen/engines/llm/` | ~3.4GB | Qwen3-VL LLM 部分 | EdgeQwen Stage 9–12 构建 |
| `visual.engine` | `~/edgeqwen/engines/visual/` | ~824MB | Qwen3-VL visual 部分 | EdgeQwen Stage 9–12 构建 |

`resnet18_image_encoder_fp32.engine` 为精度对照保留，运行时不用。

校验命令（板上）：

```bash
cd ~/edgevisionrt && python3 scripts/check_models.py  # TODO(M8d): 按本表实现
```
