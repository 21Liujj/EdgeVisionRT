# EdgeVisionRT fused board image (M9: single-container fusion, thin runtime).
#
# Strategy: board-native C++ binaries are built ON THE HOST (verified flow),
# the image only packs runtime: binaries + Python agent + vendored donors.
# This avoids TRT/DLA version skew between base image and host userspace.
#
# Base: local l4t-jetpack r36.4.0 (CUDA 12.6 runtime + TRT runtime included).
# GPU access on Orin Nano (CDI mode, no --runtime nvidia):
#   docker run --device nvidia.com/gpu=all ...
ARG BASE_IMAGE=nvcr.io/nvidia/l4t-jetpack:r36.4.0
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# --- system deps: Python runtime + V4L2 utils only (no build toolchain) ---
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      python3 \
      python3-pip \
      v4l-utils && \
    rm -rf /var/lib/apt/lists/*

# --- Python deps (pure wheels only; torch/TRT stay from base image) ---
# Worker chain needs: transformers (OWL text tower) + pillow + numpy.
# torch2trt/nanoowl ship in-repo (third_party/), added via PYTHONPATH.
RUN pip3 install --no-cache-dir \
      "transformers==5.0.0" \
      "pillow==12.3.0" \
      "numpy==1.26.4" \
      "pycocotools==2.0.11"

WORKDIR /opt/edgevisionrt

# --- prebuilt board binaries (built on host via cmake --preset board-native) ---
COPY build/board/edgevisiond build/board/edgevisiond
COPY build/board/file_infer build/board/file_infer

# --- Python control plane + vendored donors (code only, no engines) ---
COPY control_service/ control_service/
COPY vendors/ vendors/
COPY third_party/ third_party/
COPY models/MANIFEST.md models/MANIFEST.md
COPY config/ config/
COPY deploy/entrypoint.sh deploy/entrypoint.sh
RUN chmod +x deploy/entrypoint.sh build/board/edgevisiond build/board/file_infer && \
    python3 -c "import control_service.paths as p; print('paths OK:', p.RUN_OWL)"

# --- runtime env: vendored python pkgs on PYTHONPATH ---
# NOTE: qwen_ask binary + edgellm plugin + all *.engine files are NOT baked.
# They mount at runtime (see deploy/run_board.sh); override via env:
#   EDGEVISION_QWEN_ASK=/opt/edgeqwen-mnt/qwen_ask
#   EDGEVISION_NANOSAM=/opt/edgeqwen-mnt/nanosam_cpp
ENV PYTHONPATH=/opt/edgevisionrt/vendors/nanoowl:/opt/edgevisionrt/third_party/nanoowl:/opt/edgevisionrt/third_party/torch2trt/dist/torch2trt-0.5.0-py3.10.egg
ENV EDGEVISION_ROOT=/opt/edgevisionrt

# Runtime defaults: engines + sessions + socket are mounted, never baked.
VOLUME ["/opt/edgevisionrt/models", "/opt/edgevisionrt/sessions", "/opt/edgevisionrt/pending_review"]

EXPOSE 8080
ENTRYPOINT ["/opt/edgevisionrt/deploy/entrypoint.sh"]
