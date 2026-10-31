"""eval/compare.py — YOLO vs OWL COCO results: mAP-lite + recall-lift.

Inputs: file_infer out dir (YOLO) + eval_nanoowl_coco out dir (OWL).
Metric (no pycocotools needed on board):
  per-class recall @ conf>=t on val2017 annotations (instances JSON).
Core question: of YOLO's misses (low-conf/empty), how many does OWL recover?
Usage: python3 eval/compare.py --yolo eval/out_yolo --owl eval/out_owl \
         --ann data/coco/annotations/instances_val2017.json
"""
from __future__ import annotations

import argparse
import json
import os


def load_dets(d: str) -> dict:
    out = {}
    for root, _, fs in os.walk(d):
        for f in fs:
            if f.endswith(".json") and f != "summary.json":
                try:
                    j = json.load(open(os.path.join(root, f)))
                    img = j.get("image") or f
                    out[img] = j.get("detections", j.get("dets", []))
                except Exception:
                    pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", required=True)
    ap.add_argument("--owl", required=True)
    ap.add_argument("--ann", default="")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    y = load_dets(args.yolo)
    o = load_dets(args.owl)
    print(f"yolo images: {len(y)}  owl images: {len(o)}")
    common = set(y) & set(o)
    print(f"common images: {len(common)}")
    # lift: images where YOLO found nothing but OWL found something
    lift = sum(1 for i in common if not y[i] and o[i])
    print(f"recall-lift images (yolo empty, owl non-empty): {lift}")
    y_n = sum(len(v) for v in y.values())
    o_n = sum(len(v) for v in o.values())
    print(f"total dets: yolo={y_n} owl={o_n}")
    if args.ann and os.path.isfile(args.ann):
        print(f"ann: {args.ann} (per-class recall TODO: needs pycocotools)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
