"""M9 fusion: unified path resolution for workers.

Priority: env EDGEVISION_ROOT > config project_root > repo auto-detect.
Vendors resolve to vendored code first (vendors/...), fall back to
board legacy dirs (/home/nvidia/jetson-deploy-lab, /home/nvidia/edgeqwen)
so old deployments keep working during migration.
"""
from __future__ import annotations

import json
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CFG = os.path.join(_REPO, "config", "device_config.json")

_config: dict = {}
try:
    with open(_CFG) as f:
        _config = json.load(f)
except Exception:
    _config = {}

ROOT = os.environ.get("EDGEVISION_ROOT") or _config.get("project_root") or _REPO
V = _config.get("vendors", {})
NANO = _config.get("nanoowl", {})
QWEN = _config.get("qwen", {})

JDL = os.environ.get("EDGEVISION_JDL") or V.get("jetson_deploy_lab") or "/home/nvidia/jetson-deploy-lab"
EQQ = os.environ.get("EDGEVISION_EQQ") or V.get("edgeqwen") or "/home/nvidia/edgeqwen"
EQB = V.get("edgeqwen_build") or os.path.join(EQQ, "TensorRT-Edge-LLM", "build")


def _vendored(*parts: str) -> str:
    return os.path.join(ROOT, *parts)


def _first(*paths: str) -> str:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return paths[0]


# NanoOWL (vendored script preferred, legacy board path fallback)
RUN_OWL = _first(_vendored(NANO.get("run_script", "vendors/nanoowl/run_nanoowl.py")),
                 os.path.join(JDL, "scripts", "run_nanoowl.py"))
NANOSAM_CPP = os.environ.get("EDGEVISION_NANOSAM") or _first(
    _vendored(NANO.get("nanosam_bin", "vendors/nanoowl/nanosam_cpp")),
    os.path.join(JDL, "cpp_runtime", "build", "nanosam_cpp"))
OWL_ENGINE = _first(os.path.join(ROOT, NANO.get("owl_engine", "")),
                    os.path.join(JDL, "third_party", "nanoowl", "data",
                                 "owlvit_image_encoder_patch32.engine"))
SAM_ENCODER = _first(os.path.join(ROOT, NANO.get("sam_encoder", "")),
                     os.path.join(JDL, "third_party", "nanosam", "data",
                                  "resnet18_image_encoder.engine"))
SAM_DECODER = _first(os.path.join(ROOT, NANO.get("sam_decoder", "")),
                     os.path.join(JDL, "third_party", "nanosam", "data",
                                  "mobile_sam_mask_decoder.engine"))
NANOOWL_PKG = _first(os.path.join(ROOT, NANO.get("python_pkg", "")),
                     os.path.join(JDL, "third_party", "nanoowl"))
T2T_EGG = _first(os.path.join(ROOT, NANO.get("torch2trt_egg", "")),
                 os.path.join(JDL, "third_party", "torch2trt", "dist",
                              "torch2trt-0.5.0-py3.10.egg"))

# Qwen (vendored binary preferred, upstream build fallback)
QWEN_ASK = os.environ.get("EDGEVISION_QWEN_ASK") or _first(
    _vendored(QWEN.get("ask_bin", "vendors/edgeqwen/qwen_ask")),
    os.path.join(EQB, "examples", "runtime_sdk", "qwen_ask"))
QWEN_PLUGIN = _first(os.path.join(ROOT, QWEN.get("plugin", "")),
                      os.path.join(EQB, "libNvInfer_edgellm_plugin.so"))
