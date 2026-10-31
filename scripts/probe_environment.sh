#!/usr/bin/env bash
# probe_environment.sh — M0 board baseline. Collects SoC/JetPack/kernel/power
# into a single JSON report. Safe to run anywhere; missing fields are null
# with reason HARDWARE_PENDING (never fake values).
#
# Usage: bash scripts/probe_environment.sh --out reports/environment.json
set -euo pipefail
OUT="reports/environment.json"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2;;
    *) echo "usage: $0 [--out PATH]"; exit 1;;
  esac
done

py() { python3 - "$@" <<'EOF'
import json, os, re, subprocess, sys, datetime

def run(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""

def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return ""

out = {}
out["collected_at"] = datetime.datetime.now().astimezone().isoformat()
out["hostname"] = run("hostname")
out["uname"] = run("uname -a")
out["os_release"] = read("/etc/os-release")
out["l4t"] = read("/etc/nv_tegra_release")
out["jetpack_apt"] = run("apt list --installed 2>/dev/null | grep -i -E 'jetpack|cuda|tensorrt' | head -n 20")

# SoC / clocks
out["soc_model"] = read("/sys/firmware/devicetree/base/model") or read("/proc/device-tree/model")
out["cpu_online"] = read("/sys/devices/system/cpu/online")
out["cpu_max_mhz"] = []
for i in range(8):
    v = read(f"/sys/devices/system/cpu/cpu{i}/cpufreq/cpuinfo_max_freq")
    if v: out["cpu_max_mhz"].append(int(v)//1000)

# GPU load (best effort)
out["tegrastats_snapshot"] = run("timeout 3 tegrastats --interval 1000 --count 1 2>/dev/null || true")
out["nvidia_smi"] = run("nvidia-smi -L 2>/dev/null || true")

# Memory / disk
out["meminfo_head"] = run("head -n 5 /proc/meminfo")
out["df_root"] = run("df -h / /data 2>/dev/null || df -h /")

# Power model
out["nvpmodel"] = run("nvpmodel -q 2>/dev/null || cat /etc/nvpmodel.conf 2>/dev/null | head -n 20 || true")

# Kernel video stack
out["kernel_video_modules"] = run("lsmod | grep -E 'uvcvideo|videobuf|videodev' || true")
out["v4l2_devices"] = run("ls -l /dev/video* 2>/dev/null || echo NONE")
out["usb_topology"] = run("lsusb -t 2>/dev/null || lsusb 2>/dev/null || echo NO_LSUSB")

missing = []
for k in ("soc_model","l4t"):
    if not out.get(k): missing.append(k)
out["hardware_pending"] = (len(missing) > 0)
out["missing_fields"] = missing
out["note"] = "Run ON the Jetson board. On other hosts soc_model/l4t are empty and hardware_pending=true."

print(json.dumps(out, indent=2, ensure_ascii=False))
EOF
}
mkdir -p "$(dirname "$OUT")"
py > "$OUT"
echo "[probe_environment] wrote $OUT"
