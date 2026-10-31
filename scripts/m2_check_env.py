# M2 environment check
import subprocess as sp, sys, json

def run(cmd):
    r = sp.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
    return (r.stdout.strip(), r.stderr.strip(), r.returncode)

checks = {
    "tensorrt": "python3 -c 'import tensorrt; print(tensorrt.__version__)'",
    "onnx": "python3 -c 'import onnx; print(onnx.__version__)'",
    "torch": "python3 -c 'import torch; print(torch.__version__)'",
    "ultralytics": "python3 -c 'import ultralytics; print(ultralytics.__version__)'",
    "opencv": "python3 -c 'import cv2; print(cv2.__version__)'",
    "numpy": "python3 -c 'import numpy; print(numpy.__version__)'",
    "cuda": "python3 -c 'import ctypes;lib=ctypes.CDLL(\"libcudart.so\");v=ctypes.c_int();lib.cudaRuntimeGetVersion(ctypes.byref(v));print(v.value)'",
    "trtexec": "which trtexec || ls /usr/src/tensorrt/bin/trtexec 2>/dev/null || echo NOT_FOUND",
    "dpkg_trt": "dpkg -l 2>/dev/null | grep -i tensorrt | head -5",
    "libnvinfer": "ls /usr/lib/aarch64-linux-gnu/libnvinfer*.so 2>/dev/null | head -5",
}

for name, cmd in checks.items():
    out, err, rc = run(cmd)
    short = out[:200] if out else ""
    if err and not short:
        short = f"ERR: {err[:200]}"
    print(f"[{name}] rc={rc}  {short}")

print("\n=== done ===")