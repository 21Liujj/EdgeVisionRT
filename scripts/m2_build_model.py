# M2 step 1: export YOLOv8-n to ONNX, then build TRT engine with trtexec.
import subprocess as sp, sys, os, json, shutil

MODEL_DIR = "/home/nvidia/edgevisionrt/models"
ONNX_PATH = f"{MODEL_DIR}/yolov8n.onnx"
ENGINE_PATH = f"{MODEL_DIR}/yolov8n_fp16.engine"
TRTEXEC = "/usr/src/tensorrt/bin/trtexec"

def run(cmd, timeout=600):
    print(f"\n>>> {cmd[:120]}...")
    r = sp.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if r.stdout:
        lines = r.stdout.strip().split("\n")
        for line in lines[-20:]:  # last 20 lines only
            print(f"    {line}")
    if r.stderr:
        err_lines = r.stderr.strip().split("\n")
        for line in err_lines[-5:]:
            print(f"    [e] {line}")
    print(f"  rc={r.returncode}")
    return r.returncode

def step1_export_onnx():
    print("=" * 50)
    print("STEP 1/3: Export YOLOv8-n to ONNX")
    print("=" * 50)

    if os.path.exists(ONNX_PATH):
        size = os.path.getsize(ONNX_PATH)
        print(f"ONNX already exists: {ONNX_PATH} ({size:,} bytes), skip.")
        return 0

    export_script = """
import torch
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=640, opset=12, simplify=True)
print("EXPORT_DONE")
"""
    with open("/tmp/m2_export.py", "w") as f:
        f.write(export_script)

    rc = run("cd /tmp && python3 m2_export.py", timeout=300)
    if rc != 0:
        return rc

    # YOLO saves to yolov8n.onnx in cwd; move to models/
    os.makedirs(MODEL_DIR, exist_ok=True)
    src = "/tmp/yolov8n.onnx"
    if os.path.exists(src):
        shutil.move(src, ONNX_PATH)
        print(f"ONNX saved to {ONNX_PATH} ({os.path.getsize(ONNX_PATH):,} bytes)")
    else:
        print("ERROR: ONNX file not found at /tmp/yolov8n.onnx")
        return 1
    return 0

def step2_build_engine():
    print("=" * 50)
    print("STEP 2/3: Build TensorRT FP16 engine")
    print("=" * 50)

    if not os.path.exists(ONNX_PATH):
        print("ERROR: ONNX not found, run step 1 first")
        return 1

    if os.path.exists(ENGINE_PATH):
        size = os.path.getsize(ENGINE_PATH)
        print(f"Engine already exists: {ENGINE_PATH} ({size:,} bytes), skip.")
        return 0

    cmd = (
        f"{TRTEXEC} "
        f"--onnx={ONNX_PATH} "
        f"--saveEngine={ENGINE_PATH} "
        f"--fp16 "
        f"--verbose "
        f"--workspace=1024 "
        f"2>&1"
    )
    rc = run(cmd, timeout=600)
    if rc != 0:
        return rc

    if os.path.exists(ENGINE_PATH):
        print(f"Engine saved: {ENGINE_PATH} ({os.path.getsize(ENGINE_PATH):,} bytes)")
    else:
        print("ERROR: engine file not created")
        return 1
    return 0

def step3_verify():
    print("=" * 50)
    print("STEP 3/3: Verify engine with trtexec --loadEngine")
    print("=" * 50)

    cmd = (
        f"{TRTEXEC} "
        f"--loadEngine={ENGINE_PATH} "
        f"--iterations=10 "
        f"--warmUp=200 "
        f"--duration=3 "
        f"2>&1"
    )
    rc = run(cmd, timeout=120)
    return rc

if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)

    rc = step1_export_onnx()
    if rc != 0:
        print("Export failed, stopping.")
        sys.exit(1)

    rc = step2_build_engine()
    if rc != 0:
        print("Build failed, stopping.")
        sys.exit(1)

    rc = step3_verify()
    if rc != 0:
        print("Verify failed.")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("M2 model pipeline DONE")
    print(f"  ONNX:   {ONNX_PATH}")
    print(f"  Engine: {ENGINE_PATH}")
    print("=" * 50)