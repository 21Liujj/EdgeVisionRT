"""M8b NanoOWL worker: pending_review/crop.jpg -> OWL detect -> SAM mask -> result.json.

Calls board-side scripts (no model code copied here):
  OWL : scripts/run_nanoowl.py --image crop --prompts ... --output-dir <review>/
  SAM : cpp_runtime/build/nanosam_cpp --image crop --box ... --encoder ... --decoder ...

Single-review CLI (agent calls once per review dir). Exits 0 on attempt
(result.json always written, ok=false when models fail).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("worker.nanoowl")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import (NANOOWL_PKG, NANOSAM_CPP, OWL_ENGINE, RUN_OWL, SAM_DECODER,
                   SAM_ENCODER, T2T_EGG)

OWL_MODEL = "google/owlvit-base-patch32"

FIXED_PROMPTS = "an object,a thing,unknown item"

# COCO 80-class names (index = cls id). Used to build precise OWL prompts
# from the YOLO box that triggered this review.
COCO_NAMES = ["person","bicycle","car","motorcycle","airplane","bus","train","truck",
 "boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird",
 "cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
 "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
 "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
 "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
 "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
 "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
 "remote","keyboard","cell phone","microwave","oven","toaster","sink",
 "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
 "toothbrush"]


def run(cmd: list[str], timeout: int) -> tuple[int, str]:
    try:
        env = dict(os.environ)
        # NanoOWL python pkg vendored (third_party/nanoowl not pip-installable);
        # torch2trt is a prebuilt .egg (pip install egg unsupported), add directly.
        env["PYTHONPATH"] = os.pathsep.join(
            [p for p in (NANOOWL_PKG, T2T_EGG, env.get("PYTHONPATH")) if p])
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault("TRANSFORMERS_OFFLINE", "1")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return r.returncode, (r.stdout + r.stderr)[-2000:]
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", required=True, help="pending_review/<stamp> dir")
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--owl-threshold", type=float, default=0.10)
    ap.add_argument("--timeout-owl", type=int, default=180)
    ap.add_argument("--timeout-sam", type=int, default=120)
    args = ap.parse_args()

    review = args.review
    meta_p = os.path.join(review, "meta.json")
    crop_p = os.path.join(review, "crop.jpg")
    try:
        meta = json.load(open(meta_p))
    except Exception as exc:
        log.error("bad meta: %s", exc)
        return 2
    crop = os.path.join(review, meta.get("crop", "crop.jpg"))
    if not os.path.isfile(crop):
        crop = crop_p
    # precise prompt: YOLO trigger class name first, generic fallback after.
    # e.g. cls=56 (chair) -> "a chair,an object,a thing"
    prompts = args.prompts
    if not prompts:
        try:
            yb0 = (meta.get("yolo_boxes") or [None])[0] or {}
            cls = int(yb0.get("cls", -1))
            cname = COCO_NAMES[cls] if 0 <= cls < len(COCO_NAMES) else ""
        except Exception:
            cname = ""
        prompts = f"a {cname},{FIXED_PROMPTS}" if cname else FIXED_PROMPTS
    t0 = time.perf_counter()
    peak = 0.0

    # 1) OWL detect
    owl_out = os.path.join(review, "owl_out")
    rc, out = run(["python3", RUN_OWL, "--image", crop, "--prompts", prompts,
                   "--threshold", str(args.owl_threshold),
                   "--engine", OWL_ENGINE, "--model", OWL_MODEL,
                   "--output-dir", owl_out], args.timeout_owl)
    owl_dets: list[dict] = []
    try:
        det_p = os.path.join(owl_out, "detections.json")
        if os.path.isfile(det_p):
            owl_dets = json.load(open(det_p)).get("detections", [])
    except Exception:
        owl_dets = []
    owl_ms = (time.perf_counter() - t0) * 1000.0

    # 2) SAM mask for top box (YOLO box preferred when OWL empty)
    # NOTE: meta yolo_boxes are NORMALIZED cx,cy,w,h of the 640 frame.
    # Map to crop pixels: full-frame stretch => same normalized rect in crop,
    # so box = (cx-w/2)*W ... computed in crop pixel space.
    mask_path = ""
    sam_ms = 0.0
    box = None
    if owl_dets:
        b = sorted(owl_dets, key=lambda d: d.get("score", 0), reverse=True)[0]["box"]
        box = [float(v) for v in b]  # xyxy (crop pixels, from OWL on crop)
    else:
        yb = (meta.get("yolo_boxes") or [None])[0]
        if yb:
            try:
                from PIL import Image
                W, H = Image.open(crop).size
                cx, cy, bw, bh = yb["x"], yb["y"], yb["w"], yb["h"]
                box = [(cx - bw / 2) * W, (cy - bh / 2) * H,
                       (cx + bw / 2) * W, (cy + bh / 2) * H]
                # clamp into crop
                box = [max(0, box[0]), max(0, box[1]), min(W - 1, box[2]), min(H - 1, box[3])]
                if box[2] <= box[0] or box[3] <= box[1]:
                    box = None
            except Exception:
                box = None
    if box:
        t1 = time.perf_counter()
        mask_path = os.path.join(review, "mask.png")
        rc2, _ = run([NANOSAM_CPP, "--image", crop,
                      "--box", *[str(int(v)) for v in box],
                      "--encoder", SAM_ENCODER, "--decoder", SAM_DECODER,
                      "--output", mask_path], args.timeout_sam)
        sam_ms = (time.perf_counter() - t1) * 1000.0
        if rc2 != 0 or not os.path.isfile(mask_path):
            mask_path = ""

    # tegrastats peak (best-effort, may be absent)
    result = {
        "worker": "nanoowl",
        "ok": bool(owl_dets or mask_path),
        "lat_ms": round(owl_ms + sam_ms, 2),
        "owl_ms": round(owl_ms, 2),
        "sam_ms": round(sam_ms, 2),
        "peak_mem_mb": peak,
        "owl_boxes": owl_dets[:5],
        "mask_path": os.path.basename(mask_path) if mask_path else "",
        "prompts": prompts,
        "error": "" if (owl_dets or mask_path) else f"owl_rc={rc} no detections",
    }
    tmp = os.path.join(review, "result.json.tmp")
    fin = os.path.join(review, "result.json")
    json.dump(result, open(tmp, "w"), indent=1)
    os.rename(tmp, fin)
    log.info("done ok=%s lat_ms=%.1f boxes=%d", result["ok"], result["lat_ms"],
             len(result["owl_boxes"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
