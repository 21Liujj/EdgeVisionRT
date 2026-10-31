# third_party 说明（不进 git 的外部依赖，板上原位引用）

| 依赖 | 板上路径 | 用途 | 获取 |
|---|---|---|---|
| `nanoowl` Python 包 | `~/jetson-deploy-lab/third_party/nanoowl` | `OwlPredictor`（`run_nanoowl.py` import） | 随 EdgePerception 源码，pip 不可装 |
| `torch2trt-0.5.0-py3.10.egg` | `~/jetson-deploy-lab/third_party/torch2trt/dist/` | OWL engine 的 `TRTModule` 加载器 | 预编 egg，worker 用 PYTHONPATH 直接引用（`pip install egg` 不支持） |
| `libNvInfer_edgellm_plugin.so` | `~/edgeqwen/TensorRT-Edge-LLM/build/` | Qwen TensorRT plugin | 随 EdgeQwen 构建，`EDGELLM_PLUGIN_PATH` 注入 |

约束：`control_service/workers/*.py` 不得硬编码上述绝对路径，
统一经 `config/device_config.json` 的 `vendors.*` 字段读取。
