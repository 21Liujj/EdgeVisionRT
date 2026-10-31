# -*- coding: utf-8 -*-
"""Stage 2B - NanoOWL fixed regression suite + batch correctness runner.

Reuses scripts/run_nanoowl.py inference logic (load_predictor / run_once) — no
NanoOWL model core is copied or reimplemented. Runs every case in a fixed
manifest, saves per-case detections.json/result.jpg/timing.json/status, and
writes a summary.json aggregate.

A case with zero detections is NOT a failure; only model-load / inference
exception / invalid output is marked execution FAIL. The Day 1 owl case MUST
still detect "an owl", else the whole stage FAILs.
"""
import argparse
import json
import os
import sys
import traceback

# Make sibling script importable when run as a file.
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import run_nanoowl  # reuse inference logic

ASSETS = "/home/nvidia/jetson-deploy-lab/third_party/nanoowl/assets"

# Fixed regression manifest. Cases use distinct real images available locally.
# Only *raw* asset photos are used (not the *_out.jpg visualization outputs).
DEFAULT_MANIFEST = [
    {
        "case_id": "owl_glove",
        "image": os.path.join(ASSETS, "owl_glove_small.jpg"),
        "prompts": "an owl,a glove,a finger",
        "threshold": 0.10,
        "expect_label": "an owl",  # Day 1 baseline anchor
    },
    {
        "case_id": "owl_glove_full",
        "image": os.path.join(ASSETS, "owl_glove.jpg"),
        "prompts": "an owl,a glove",
        "threshold": 0.10,
    },
    {
        "case_id": "frog",
        "image": os.path.join(ASSETS, "frog.jpg"),
        "prompts": "a frog,a leaf,a hand",
        "threshold": 0.10,
    },
    {
        "case_id": "classroom",
        "image": os.path.join(ASSETS, "class.jpg"),
        "prompts": "a person,a backpack,a chair",
        "threshold": 0.10,
    },
    {
        "case_id": "camera_scene",
        "image": os.path.join(ASSETS, "camera.jpg"),
        "prompts": "a camera,a hand,a person",
        "threshold": 0.10,
    },
]


def parse_args():
    ap = argparse.ArgumentParser(description="NanoOWL fixed regression runner")
    ap.add_argument("--engine", default=run_nanoowl.DEFAULT_ENGINE)
    ap.add_argument("--model", default=run_nanoowl.DEFAULT_MODEL)
    ap.add_argument("--output-root", default="/home/nvidia/jetson-deploy-lab/results/nanoowl_regression")
    ap.add_argument("--manifest", default=None, help="optional path to a manifest JSON file")
    return ap.parse_args()


def main():
    args = parse_args()
    manifest = DEFAULT_MANIFEST
    if args.manifest:
        with open(args.manifest) as f:
            manifest = json.load(f)

    os.makedirs(args.output_root, exist_ok=True)

    # 1) load the model once, reuse for all cases
    case_results = []
    try:
        predictor, model_load_ms = run_nanoowl.load_predictor(args.model, args.engine)
    except Exception as e:
        tb = traceback.format_exc()
        summary = {
            "stage": "2B",
            "status": "FAIL",
            "fail_reason": "model_load_exception",
            "error": str(e),
            "traceback": tb,
            "total_cases": len(manifest),
            "pass_cases": 0,
            "failed_cases": len(manifest),
            "cases": [],
        }
        with open(os.path.join(args.output_root, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))
        return 1

    distinct_images = {c["image"] for c in manifest}

    for c in manifest:
        case_id = c["case_id"]
        case_dir = os.path.join(args.output_root, case_id)
        os.makedirs(case_dir, exist_ok=True)
        rec = {"case_id": case_id, "image": c["image"], "prompts": c["prompts"].split(","),
               "threshold": c["threshold"], "detection_count": 0,
               "labels": [], "scores": [], "execution": "PASS", "status_message": ""}
        try:
            detections_doc, timing_doc = run_nanoowl.run_once(
                predictor, c["image"], c["prompts"], c["threshold"], case_dir,
                pad_square=False, model=args.model, engine=args.engine,
            )
            timing_doc["model_load_ms"] = model_load_ms  # shared load, recorded once
            with open(os.path.join(case_dir, "timing.json"), "w") as f:
                json.dump(timing_doc, f, indent=2)
            with open(os.path.join(case_dir, "status.json"), "w") as f:
                json.dump({"execution": "PASS", "case_id": case_id}, f, indent=2)

            rec["detection_count"] = detections_doc["num_detections"]
            rec["labels"] = [d["label"] for d in detections_doc["detections"]]
            rec["scores"] = [d["score"] for d in detections_doc["detections"]]
            # correctness anchor (optional)
            if c.get("expect_label"):
                rec["expect_label"] = c["expect_label"]
                rec["expect_label_found"] = c["expect_label"] in rec["labels"]
        except Exception as e:
            rec["execution"] = "FAIL"
            rec["status_message"] = str(e)
            with open(os.path.join(case_dir, "status.json"), "w") as f:
                json.dump({"execution": "FAIL", "case_id": case_id, "error": str(e)}, f, indent=2)
            try:
                traceback.print_exc(file=open(os.path.join(case_dir, "error.log"), "w"))
            except Exception:
                pass
        case_results.append(rec)

    pass_cases = sum(1 for r in case_results if r["execution"] == "PASS")
    failed_cases = sum(1 for r in case_results if r["execution"] == "FAIL")

    # Stage gates: model-load ok + Day 1 owl anchor still detects "an owl".
    owl_ok = False
    for r in case_results:
        if r.get("expect_label") == "an owl" and r.get("expect_label_found"):
            owl_ok = True

    stage_status = "PASS" if (failed_cases == 0 and owl_ok) else "FAIL"

    summary = {
        "stage": "2B",
        "status": stage_status,
        "owl_baseline_pass": owl_ok,
        "model_load_ms": model_load_ms,
        "total_cases": len(case_results),
        "pass_cases": pass_cases,
        "failed_cases": failed_cases,
        "distinct_images": len(distinct_images),
        "cases": case_results,
        "note": "Zero-detection cases are NOT execution failures; only model-load/inference exceptions or invalid output are FAIL.",
    }
    with open(os.path.join(args.output_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0 if stage_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
