# -*- coding: utf-8 -*-
"""Stage 2A - reusable NanoOWL open-vocabulary detection CLI.

Wraps the fixed NanoOWL OwlPredictor API and the Day 1 FP16 TensorRT engine.
Reuses the model logic from Day 1 baseline (no reimplementation); adds a clean
CLI + timing record (not a formal benchmark).

Image + text prompt -> NanoOWL (OWL-ViT) + TensorRT -> box / label / score.

Outputs (in --output-dir):
  detections.json  - prompts, threshold, label, score, box per detection
  result.jpg       - visualization with boxes + labels
  timing.json      - model_load_ms / inference_ms / total_ms (no stats)
"""
import argparse
import json
import os
import time

import numpy as np
import PIL.Image
from PIL import Image as _Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402  (after env vars)
from nanoowl.owl_predictor import OwlPredictor  # noqa: E402
from nanoowl.owl_drawing import draw_owl_output  # noqa: E402


# Defaults pinned from the Day 1 baseline.
DEFAULT_MODEL = "google/owlvit-base-patch32"
DEFAULT_IMAGE = "/home/nvidia/jetson-deploy-lab/third_party/nanoowl/assets/owl_glove_small.jpg"
DEFAULT_ENGINE = "/home/nvidia/jetson-deploy-lab/third_party/nanoowl/data/owlvit_image_encoder_patch32.engine"
DEFAULT_OUTPUT_DIR = "/home/nvidia/jetson-deploy-lab/results/nanoowl_cli"


def parse_args():
    ap = argparse.ArgumentParser(description="NanoOWL open-vocabulary detection CLI")
    ap.add_argument("--image", default=DEFAULT_IMAGE, help="path to input image")
    ap.add_argument("--prompts", default="an owl,a glove,a finger",
                    help="comma-separated text prompts, e.g. 'an owl,a glove'")
    ap.add_argument("--threshold", type=float, default=0.10, help="detection score threshold")
    ap.add_argument("--engine", default=DEFAULT_ENGINE, help="TensorRT image encoder engine path")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="directory for outputs")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="OWL-ViT model id (offline HF cache)")
    ap.add_argument("--pad-square", action="store_true", help="pad input image to square")
    return ap.parse_args()


def load_predictor(model, engine):
    """Load OwlPredictor once (timed), for batch reuse via run_once()."""
    t0 = time.perf_counter()
    predictor = OwlPredictor(model, image_encoder_engine=engine)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    model_load_ms = (time.perf_counter() - t0) * 1000.0
    return predictor, model_load_ms


def run_once(predictor, image_path, prompts_text, threshold, output_dir,
             pad_square=False, model=None, engine=None):
    """Reuse entry: run detection + save detections.json/result.jpg/timing.json.

    `predictor` is assumed already loaded (load NOT timed here, to allow batch
    reuse); model_load_ms is reported as 0.0 when reusing a predictor.
    Returns (detections_doc, timing_doc).
    """
    os.makedirs(output_dir, exist_ok=True)
    text = [t.strip() for t in prompts_text.split(",") if t.strip()]
    image = PIL.Image.open(image_path).convert("RGB")

    t1 = time.perf_counter()
    text_encodings = predictor.encode_text(text)
    output = predictor.predict(
        image=image, text=text, text_encodings=text_encodings,
        threshold=threshold, pad_square=pad_square,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    inference_ms = (time.perf_counter() - t1) * 1000.0

    labels = output.labels.detach().cpu().tolist()
    scores = output.scores.detach().cpu().tolist()
    boxes = [[float(v) for v in b] for b in output.boxes.detach().cpu().tolist()]
    detections = []
    for lab, sc, bx in zip(labels, scores, boxes):
        li = int(lab)
        detections.append({
            "label_index": li,
            "label": text[li] if 0 <= li < len(text) else str(li),
            "score": float(sc),
            "box": bx,  # xyxy
        })
    detections.sort(key=lambda d: d["score"], reverse=True)

    detections_doc = {
        "model": model,
        "engine": engine,
        "image": image_path,
        "prompts": text,
        "threshold": threshold,
        "num_detections": len(detections),
        "detections": detections,
    }
    with open(os.path.join(output_dir, "detections.json"), "w") as f:
        json.dump(detections_doc, f, indent=2)

    # Visualization: writable copy (OpenCV 4.x rejects read-only PIL->numpy arrays).
    vis_arr = np.array(image).copy()
    _ = draw_owl_output(vis_arr, output, text=text, draw_text=True)
    _Image.fromarray(vis_arr).save(os.path.join(output_dir, "result.jpg"))

    timing_doc = {
        "model_load_ms": 0.0,
        "inference_ms": inference_ms,
        "total_ms": inference_ms,
    }
    with open(os.path.join(output_dir, "timing.json"), "w") as f:
        json.dump(timing_doc, f, indent=2)
    return detections_doc, timing_doc


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    t_start = time.perf_counter()
    predictor, model_load_ms = load_predictor(args.model, args.engine)

    detections_doc, timing_doc = run_once(
        predictor, args.image, args.prompts, args.threshold,
        args.output_dir, pad_square=args.pad_square,
        model=args.model, engine=args.engine,
    )
    # restore load timing for the single-shot CLI view
    timing_doc["model_load_ms"] = model_load_ms
    timing_doc["total_ms"] = (time.perf_counter() - t_start) * 1000.0
    with open(os.path.join(args.output_dir, "timing.json"), "w") as f:
        json.dump(timing_doc, f, indent=2)

    print(json.dumps({"detections": detections_doc, "timing": timing_doc}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
