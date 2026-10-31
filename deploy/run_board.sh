#!/bin/bash
# Run the fused EdgeVisionRT container on the Orin Nano board.
# CDI GPU mode (--device nvidia.com/gpu=all). Engines/sessions mount from host.
# Usage: bash deploy/run_board.sh [--shell]
set -e

IMAGE="${EDGEVISION_IMAGE:-edgevisionrt:board}"
HOST_ROOT="${EDGEVISION_HOST_ROOT:-/home/nvidia/edgevisionrt}"
# Donor build outputs (NOT in repo): qwen_ask binary + edgellm plugin.
# Mounted read-only; paths.py picks them up via env override.
EDGEQWEN_BUILD="${EDGEVISION_EDGEQWEN_BUILD:-/home/nvidia/edgeqwen/TensorRT-Edge-LLM/build}"
JDL="${EDGEVISION_JDL:-/home/nvidia/jetson-deploy-lab}"
GPU_FLAGS="--device nvidia.com/gpu=all"

if [ "${1:-}" = "--shell" ]; then
  exec docker run --rm -it \
    $GPU_FLAGS \
    -v "$HOST_ROOT/models:/opt/edgevisionrt/models:ro" \
    -v "$HOST_ROOT/sessions:/opt/edgevisionrt/sessions" \
    -v "$HOST_ROOT/pending_review:/opt/edgevisionrt/pending_review" \
    -v "$EDGEQWEN_BUILD/examples/runtime_sdk/qwen_ask:/opt/edgeqwen-mnt/qwen_ask:ro" \
    -v "$EDGEQWEN_BUILD/libNvInfer_edgellm_plugin.so:/opt/edgeqwen-mnt/libNvInfer_edgellm_plugin.so:ro" \
    -v "$JDL/third_party/nanoowl/data/owlvit_image_encoder_patch32.engine:/opt/edgevisionrt/models/owlvit_image_encoder_patch32.engine:ro" \
    -v "$JDL/third_party/nanosam/data/resnet18_image_encoder.engine:/opt/edgevisionrt/models/resnet18_image_encoder.engine:ro" \
    -v "$JDL/third_party/nanosam/data/mobile_sam_mask_decoder.engine:/opt/edgevisionrt/models/mobile_sam_mask_decoder.engine:ro" \
    -v "$JDL/cpp_runtime/build/nanosam_cpp:/opt/edgeqwen-mnt/nanosam_cpp:ro" \
    -e EDGEVISION_QWEN_ASK=/opt/edgeqwen-mnt/qwen_ask \
    -e EDGEVISION_NANOSAM=/opt/edgeqwen-mnt/nanosam_cpp \
    --entrypoint /bin/bash "$IMAGE"
fi

exec docker run --rm \
  --name edgevisionrt \
  $GPU_FLAGS \
  --device /dev/video-edgevision:/dev/video-edgevision \
  -v "$HOST_ROOT/models:/opt/edgevisionrt/models:ro" \
  -v "$HOST_ROOT/sessions:/opt/edgevisionrt/sessions" \
  -v "$HOST_ROOT/pending_review:/opt/edgevisionrt/pending_review" \
  -v "$EDGEQWEN_BUILD/examples/runtime_sdk/qwen_ask:/opt/edgeqwen-mnt/qwen_ask:ro" \
  -v "$EDGEQWEN_BUILD/libNvInfer_edgellm_plugin.so:/opt/edgeqwen-mnt/libNvInfer_edgellm_plugin.so:ro" \
  -v "$JDL/third_party/nanoowl/data/owlvit_image_encoder_patch32.engine:/opt/edgevisionrt/models/owlvit_image_encoder_patch32.engine:ro" \
  -v "$JDL/third_party/nanosam/data/resnet18_image_encoder.engine:/opt/edgevisionrt/models/resnet18_image_encoder.engine:ro" \
  -v "$JDL/third_party/nanosam/data/mobile_sam_mask_decoder.engine:/opt/edgevisionrt/models/mobile_sam_mask_decoder.engine:ro" \
  -v "$JDL/cpp_runtime/build/nanosam_cpp:/opt/edgeqwen-mnt/nanosam_cpp:ro" \
  -e EDGEVISION_QWEN_ASK=/opt/edgeqwen-mnt/qwen_ask \
  -e EDGEVISION_NANOSAM=/opt/edgeqwen-mnt/nanosam_cpp \
  -p 8080:8080 \
  "$IMAGE"
