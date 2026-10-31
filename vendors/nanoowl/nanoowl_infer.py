# -*- coding: utf-8 -*-
"""Stage 1J-Fallback-2 - offline NanoOWL open-vocabulary detection.

Image + text prompt -> NanoOWL (OWL-ViT) + TensorRT engine -> box/label/score.
Uses official sample image and OwlPredictor API from the fixed NanoOWL commit.
"""
import argparse
import json
import os

import PIL.Image

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from nanoowl.owl_predictor import OwlPredictor
from nanoowl.owl_drawing import draw_owl_output


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="/home/nvidia/jetson-deploy-lab/third_party/nanoowl/assets/owl_glove_small.jpg")
    ap.add_argument("--prompts", default="an owl,a glove,a finger")
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--model", default="google/owlvit-base-patch32")
    ap.add_argument("--engine", default="/home/nvidia/jetson-deploy-lab/third_party/nanoowl/data/owlvit_image_encoder_patch32.engine")
    ap.add_argument("--out_dir", default="/home/nvidia/jetson-deploy-lab/results/nanoowl_baseline")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    text = [t.strip() for t in args.prompts.split(",") if t.strip()]

    predictor = OwlPredictor(args.model, image_encoder_engine=args.engine)
    image = PIL.Image.open(args.image).convert("RGB")

    text_encodings = predictor.encode_text(text)
    output = predictor.predict(
        image=image, text=text, text_encodings=text_encodings,
        threshold=args.threshold, pad_square=False,
    )

    labels = output.labels.detach().cpu().tolist()
    scores = output.scores.detach().cpu().tolist()
    boxes = [[float(v) for v in b] for b in output.boxes.detach().cpu().tolist()]

    detections = []
    for lab, sc, bx in zip(labels, scores, boxes):
        detections.append({
            "label_index": int(lab),
            "label": text[int(lab)] if 0 <= int(lab) < len(text) else str(int(lab)),
            "score": float(sc),
            "box_xyxy": bx,
        })
    detections.sort(key=lambda d: d["score"], reverse=True)

    result = {
        "model": args.model,
        "engine": args.engine,
        "image": args.image,
        "prompts": text,
        "threshold": args.threshold,
        "num_detections": len(detections),
        "detections": detections,
    }
    print(json.dumps(result, indent=2))

    with open(os.path.join(args.out_dir, "detections.json"), "w") as f:
        json.dump(result, f, indent=2)

    # Save input image copy
    image.save(os.path.join(args.out_dir, "input_image.jpg"))

    # Visualization: draw on a writable numpy copy (OpenCV 4.x rejects read-only PIL arrays).
    import numpy as np
    from PIL import Image as _Image
    vis_arr = np.array(image).copy()  # writable copy
    _ = draw_owl_output(vis_arr, output, text=text, draw_text=True)
    vis = _Image.fromarray(vis_arr)
    vis.save(os.path.join(args.out_dir, "result_visualization.jpg"))
    print("VIS_SAVED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
