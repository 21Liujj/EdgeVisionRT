# -*- coding: utf-8 -*-
"""Stage 2C - NanoOWL steady-state performance benchmark on Jetson.

Measures stable-state inference latency of the NanoOWL TensorRT baseline:
model load (once) -> warmup -> >=300 timed inferences -> latency stats.
Reuses scripts/run_nanoowl.py load_predictor() (no model reimplementation,
no per-iter reload). Starts tegrastats during the timed window.

No optimization, no engine rebuild, no power-mode/jetson_clocks changes.
"""
import argparse
import csv
import json
import os
import statistics
import subprocess
import time

import numpy as np
import PIL.Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import run_nanoowl  # reuse load_predictor + defaults


DEFAULT_IMAGE = "/home/nvidia/jetson-deploy-lab/third_party/nanoowl/assets/owl_glove_small.jpg"
DEFAULT_PROMPTS = "an owl,a glove,a finger"
DEFAULT_THRESHOLD = 0.10
DEFAULT_OUT = "/home/nvidia/jetson-deploy-lab/results/nanoowl_benchmark"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--prompts", default=DEFAULT_PROMPTS)
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--engine", default=run_nanoowl.DEFAULT_ENGINE)
    ap.add_argument("--model", default=run_nanoowl.DEFAULT_MODEL)
    ap.add_argument("--output-dir", default=DEFAULT_OUT)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--tegrastats", default="/usr/bin/tegrastats")
    return ap.parse_args()


def percentile(vals, q):
    return float(statistics.quantiles(vals, n=100, method="inclusive")[int(q) - 1])


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    text = [t.strip() for t in args.prompts.split(",") if t.strip()]

    # 1) load model/engine ONCE
    predictor, model_load_ms = run_nanoowl.load_predictor(args.model, args.engine)
    image = PIL.Image.open(args.image).convert("RGB")
    text_encodings = predictor.encode_text(text)

    base_dir = os.path.join(args.output_dir)

    def one_inference():
        out = predictor.predict(
            image=image, text=text, text_encodings=text_encodings,
            threshold=args.threshold, pad_square=False,
        )
        return out

    # 2) warmup (not timed)
    for _ in range(args.warmup):
        one_inference()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # 3) start tegrastats over the timed window
    ts_log = os.path.join(args.output_dir, "tegrastats.log")
    ts_proc = subprocess.Popen(
        [args.tegrastats, "--interval", "500"],
        stdout=open(ts_log, "w"), stderr=subprocess.STDOUT,
    )

    # 4) timed steady-state inferences
    latencies = []
    owl_seen_every = True
    nan_detected = False
    infer_errors = 0
    try:
        for i in range(args.iters):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = one_inference()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) * 1000.0
            latencies.append(dt)

            # per-iter validity
            boxes = out.boxes.detach()
            scores = out.scores.detach()
            if not bool(torch.isfinite(boxes).all() and torch.isfinite(scores).all()):
                nan_detected = True
            labels = out.labels.detach().cpu().tolist()
            owl_here = any(text[int(l)] == "an owl" for l in labels if 0 <= int(l) < len(text))
            if not owl_here:
                owl_seen_every = False
    except Exception as e:
        infer_errors += 1
        raise
    finally:
        # 10) stop tegrastats (this stage only)
        if ts_proc.poll() is None:
            ts_proc.terminate()
            try:
                ts_proc.wait(timeout=10)
            except Exception:
                ts_proc.kill()

    # 5) latency stats
    mean = statistics.mean(latencies)
    lat_min = min(latencies)
    lat_max = max(latencies)
    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)
    p99 = percentile(latencies, 99)
    fps = 1000.0 / mean if mean > 0 else 0.0

    # write latencies.csv
    with open(os.path.join(args.output_dir, "latencies.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iter", "latency_ms"])
        for i, v in enumerate(latencies):
            w.writerow([i, v])

    bench = {
        "stage": "2C",
        "input_image": args.image,
        "prompts": text,
        "threshold": args.threshold,
        "engine": args.engine,
        "model": args.model,
        "model_load_ms": model_load_ms,
        "warmup_iters": args.warmup,
        "benchmark_iters": args.iters,
        "latency_ms": {
            "mean": mean, "min": lat_min, "max": lat_max,
            "p50": p50, "p95": p95, "p99": p99,
        },
        "estimated_fps": fps,
        "validity": {
            "inference_errors": infer_errors,
            "nan_detected": nan_detected,
            "owl_detection_present_every_iter": owl_seen_every,
        },
    }
    with open(os.path.join(args.output_dir, "benchmark.json"), "w") as f:
        json.dump(bench, f, indent=2)
    print(json.dumps(bench, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
