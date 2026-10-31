#!/bin/bash
# eval/run_yolo_coco.sh — YOLOv8-n FP16 COCO val2017 full replay (board).
# Usage: bash eval/run_yolo_coco.sh [IMG_DIR] [OUT_DIR]
# No camera needed. Produces per-image detections + timing summary.
set -e
IMG_DIR="${1:-$HOME/jetson-deploy-lab/data/coco/val2017}"
OUT_DIR="${2:-$HOME/edgevisionrt/eval/out_yolo}"
BIN="$HOME/edgevisionrt/build/board/file_infer"
mkdir -p "$OUT_DIR"
"$BIN" --input "$IMG_DIR" --out "$OUT_DIR" 2>&1 | tee "$OUT_DIR/run.log"
echo "YOLO done -> $OUT_DIR"
