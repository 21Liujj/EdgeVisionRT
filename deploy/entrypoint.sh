#!/bin/bash
# EdgeVisionRT container entrypoint (M9 single-container fusion).
# PID1: start daemon in background, run agent in foreground.
# SIGTERM -> stop agent, then daemon, then exit (systemd-friendly).
set -e

ROOT=/opt/edgevisionrt
SOCK="${EDGEVISION_SOCK:-/tmp/edgevisiond.sock}"
ENGINE="${EDGEVISION_ENGINE:-/opt/edgevisionrt/models/yolov8n_fp16.engine}"
DEVICE="${EDGEVISION_DEVICE:-/dev/video-edgevision}"
HTTP_PORT="${EDGEVISION_HTTP:-8080}"
RTSP_PORT="${EDGEVISION_RTSP:-8554}"
RTSP_ENCODER="${EDGEVISION_RTSP_ENCODER:-x264}"

cleanup() {
  echo "[entrypoint] stopping..."
  kill "$AGENT_PID" 2>/dev/null || true
  wait "$AGENT_PID" 2>/dev/null || true
  kill "$DAEMON_PID" 2>/dev/null || true
  wait "$DAEMON_PID" 2>/dev/null || true
  echo "[entrypoint] stopped."
  exit 0
}
trap cleanup TERM INT

echo "[entrypoint] starting edgevisiond: engine=$ENGINE device=$DEVICE rtsp=:$RTSP_PORT/live($RTSP_ENCODER)"
"$ROOT/build/board/edgevisiond" \
  --engine "$ENGINE" \
  --device "$DEVICE" \
  --sock "$SOCK" \
  --sessions /opt/edgevisionrt/sessions \
  --http "$HTTP_PORT" \
  --rtsp "$RTSP_PORT" \
  --rtsp-encoder "$RTSP_ENCODER" &
DAEMON_PID=$!

# Wait for daemon socket before agent attaches (max 30s).
for i in $(seq 1 30); do
  if [ -S "$SOCK" ]; then
    echo "[entrypoint] daemon socket ready."
    break
  fi
  sleep 1
done

echo "[entrypoint] starting agent on $SOCK"
cd "$ROOT"
python3 control_service/agent.py --sock "$SOCK" &
AGENT_PID=$!

wait "$AGENT_PID"
# Agent exited (shouldn't in normal ops) -> shut daemon down too.
kill "$DAEMON_PID" 2>/dev/null || true
wait "$DAEMON_PID" 2>/dev/null || true
