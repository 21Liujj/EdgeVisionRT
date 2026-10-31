# vendors/edgeqwen — Qwen2/3-VL 端侧运行时（vendored）

来源：`~/edgeqwen/TensorRT-Edge-LLM/examples/runtime_sdk`（板上 EdgeQwen 项目）
版本基线：`d788ac5` + 本地 `qwen_ask.cpp`（未提交，需同步回上游）

## 内容

| 文件 | 来源 | 说明 |
|---|---|---|
| `runtime_sdk/include/edgeqwen/runtime.h` | 同路径 | C++ 对外接口（Runtime/generate/warmup） |
| `runtime_sdk/src/` | 同路径 | `runtime.cpp` + `memory_monitor.*` + `profile_report.h` |
| `runtime_sdk/demo/runtime_demo.cpp` | 同路径 | 上游写死路径冒烟 demo（保留作参考） |
| `runtime_sdk/demo/stability_check.cpp` | 同路径 | 稳定性检查 |
| `runtime_sdk/demo/qwen_ask.cpp` | **本仓库新增** | 真 CLI：`--image/--prompt/--max-tokens`，上游 `CMakeLists.txt` 已加 target |
| `runtime_sdk/CMakeLists.txt` | 同路径 | 含 `qwen_ask` target（`M examples/...`，见板上 git status） |

## 不在这里的东西（见 models/MANIFEST.md）

- `llm.engine`（3.4GB）+ `visual.engine`（824MB）：板上 `~/edgeqwen/engines/`
- `libNvInfer_edgellm_plugin.so`：`TensorRT-Edge-LLM/build/`
- `build/`（595M）：上游构建产物，`qwen_ask` 二进制在其 `examples/runtime_sdk/` 下

## 调用方

`control_service/workers/qwen_worker.py` 调 `qwen_ask` 二进制。
`EDGELLM_PLUGIN_PATH` 由 worker 注入。
