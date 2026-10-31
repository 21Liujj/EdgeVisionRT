"""M8 contract test (host-side, no hardware): meta/result json schema.

Validates docs/m8_integration_plan.md §5.1/§5.2 field contracts so daemon,
workers, agent can never drift silently.
"""
import json
import os
import sys

META_REQUIRED = ["frame_no", "trigger", "active_model", "crop", "crop_rect",
                 "orig_wh", "last_max_conf", "yolo_boxes", "prompt_owl", "prompt_qwen"]
RESULT_REQUIRED = ["worker", "ok", "lat_ms"]

fails = 0

def check(name, cond, detail=""):
    global fails
    print(("PASS" if cond else "FAIL"), name, detail)
    fails += (not cond)

# 1) sample meta matches §5.1
meta = {
    "frame_no": 3, "trigger": "empty streak>=3", "active_model": "yolo+vlm?",
    "crop": "crop.jpg", "crop_rect": [320, 180, 640, 360], "orig_wh": [1280, 720],
    "last_max_conf": 0.0, "yolo_boxes": [],
    "prompt_owl": "unknown object in crop, describe and localize",
    "prompt_qwen": " describe the cropped object in one sentence, is it abnormal?",
}
for k in META_REQUIRED:
    check(f"meta has {k}", k in meta)
check("meta crop_rect 4 ints",
      isinstance(meta["crop_rect"], list) and len(meta["crop_rect"]) == 4)
check("meta yolo_boxes is list", isinstance(meta["yolo_boxes"], list))

# 2) sample OWL + Qwen results match §5.2
owl = {"worker": "nanoowl", "ok": True, "lat_ms": 245.0, "owl_ms": 150.0,
       "sam_ms": 95.0, "peak_mem_mb": 810, "owl_boxes": [], "mask_path": "mask.png",
       "prompts": "an object", "error": ""}
qwen = {"worker": "qwen", "ok": True, "lat_ms": 1000.0, "vision_ms": 229.0,
        "qwen_text": "a black cable.", "question": "What is this?", "error": ""}
for r, n in ((owl, "owl"), (qwen, "qwen")):
    for k in RESULT_REQUIRED:
        check(f"{n} result has {k}", k in r)

# 3) agent scheduler imports (pure-python parts)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "control_service"))
import scheduler
check("decide empty streak -> vlm",
      scheduler.decide(3, True).active_model == "yolo+vlm?")
check("decide low streak -> owl",
      scheduler.decide(3, False).active_model == "yolo+nanoowl?")
check("decide calm -> yolo", scheduler.decide(0, False).active_model == "yolo")
check("scan_reviews empty dir -> []",
      scheduler.scan_reviews("/nonexistent_dir_xyz") == [])

print("CONTRACT_TEST", "PASS" if fails == 0 else f"FAIL({fails})")
sys.exit(1 if fails else 0)
