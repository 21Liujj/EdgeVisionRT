# vendors/nanoowl — 开放词汇检测 + 提示分割（vendored）

来源：`~/jetson-deploy-lab`（板上 EdgePerception 项目）
版本基线：2026-08-13（Day 1–4，OWL 基准 score 0.3419 已复现）

## 内容

| 文件 | 来源 | 说明 |
|---|---|---|
| `run_nanoowl.py` | `scripts/run_nanoowl.py` | OWL-ViT CLI，`--image/--prompts/--threshold/--engine/--output-dir` |
| `nanoowl_infer.py` | `scripts/nanoowl_infer.py` | 推理封装 |
| `eval_nanoowl_coco.py` | `scripts/eval_nanoowl_coco.py` | COCO 全量评测 |
| `make_coco_subset.py` | `scripts/make_coco_subset.py` | 子集切分 |
| `make_nanoowl_coco_manifests.py` | `scripts/make_nanoowl_coco_manifests.py` | manifest 生成 |
| `run_nanoowl_regression.py` | `scripts/run_nanoowl_regression.py` | Day 2 回归 runner |
| `benchmark_nanoowl.py` | `scripts/benchmark_nanoowl.py` | 延迟 bench |
| `nanosam_src/*.cpp` | `cpp_runtime/src/` | NanoSAM C++ 源码（encoder/decoder TRT 封装） |
| `nanosam_include/*.hpp` | `cpp_runtime/include/` | 对应头文件 |
| `nanosam_cmake.txt` | `cpp_runtime/CMakeLists.txt` | 构建参考 |
| `nanosam_readme.md` | `cpp_runtime/README.md` | 原始说明 |

## 不在这里的东西（见 models/MANIFEST.md）

- 4 个 `.engine`（539MB）：板上 `third_party/nanoowl/data/`、`third_party/nanosam/data/`
- `third_party/nanoowl` Python 包 + `torch2trt` egg：板上原位，worker 用 PYTHONPATH 引用
- COCO 数据：`~/jetson-deploy-lab/data/coco`

## 调用方

`control_service/workers/nanoowl_worker.py` 经 `config/device_config.json`
`vendors.nanoowl_*` 路径调用。
