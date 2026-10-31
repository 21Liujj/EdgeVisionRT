#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create deterministic image-level NanoOWL COCO evaluation manifests."""
import json
import random
from pathlib import Path

ANNOTATIONS = Path("data/coco/annotations/instances_val2017.json")
OUTPUTS = {
    "nanoowl_coco_quick_images.json": 100,
    "nanoowl_coco_standard_images.json": 500,
}
SEED = 2027


def main():
    document = json.loads(ANNOTATIONS.read_text())
    all_image_ids = sorted(int(item["id"]) for item in document["images"])
    annotation_count_by_image = {}
    for annotation in document["annotations"]:
        image_id = int(annotation["image_id"])
        annotation_count_by_image[image_id] = annotation_count_by_image.get(image_id, 0) + 1

    rng = random.Random(SEED)
    standard_ids = sorted(rng.sample(all_image_ids, OUTPUTS["nanoowl_coco_standard_images.json"]))
    quick_ids = standard_ids[:OUTPUTS["nanoowl_coco_quick_images.json"]]
    selections = {
        "nanoowl_coco_quick_images.json": quick_ids,
        "nanoowl_coco_standard_images.json": standard_ids,
    }

    output_dir = Path("data/eval_manifests")
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, image_ids in selections.items():
        payload = {
            "task": "open_vocabulary_detection",
            "dataset": "COCO 2017 val",
            "sampling_unit": "image",
            "seed": SEED,
            "image_count": len(image_ids),
            "gt_instance_count": sum(annotation_count_by_image.get(image_id, 0) for image_id in image_ids),
            "image_ids": image_ids,
        }
        (output_dir / filename).write_text(json.dumps(payload, indent=2) + "\n")
        print(filename, payload["image_count"], payload["gt_instance_count"])


if __name__ == "__main__":
    main()
