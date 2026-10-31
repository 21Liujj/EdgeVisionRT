#!/bin/bash
# eval/run_owl_coco.sh — NanoOWL COCO val2017 eval (board, background-friendly).
# Usage: bash eval/run_owl_coco.sh [IMG_DIR] [ANN_JSON] [OUT_DIR]
# Single model load, batched inference (~2s/image amortized => ~3h for 5000).
# Metrics via pycocotools COCOeval (bbox). Run with nohup.
set -e
IMG_DIR="${1:-$HOME/jetson-deploy-lab/data/coco/val2017}"
ANN_JSON="${2:-$HOME/jetson-deploy-lab/data/coco/annotations/instances_val2017.json}"
OUT_DIR="${3:-$HOME/edgevisionrt/eval/out_owl}"
export PYTHONPATH="$HOME/edgevisionrt/vendors/nanoowl:$HOME/jetson-deploy-lab/third_party/nanoowl:$HOME/jetson-deploy-lab/third_party/torch2trt/dist/torch2trt-0.5.0-py3.10.egg:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p "$OUT_DIR"
python3 "$HOME/edgevisionrt/vendors/nanoowl/eval_nanoowl_coco.py" \
  --images-dir "$IMG_DIR" \
  --annotations "$ANN_JSON" \
  --engine "$HOME/jetson-deploy-lab/third_party/nanoowl/data/owlvit_image_encoder_patch32.engine" \
  --output-dir "$OUT_DIR" 2>&1 | tee "$OUT_DIR/run.log"
echo "OWL done -> $OUT_DIR"
