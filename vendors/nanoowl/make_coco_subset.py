#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
EdgePerception — deterministic COCO val2017 subset manifest generator (Stage 5A).

Generates results/coco_subsets/{subset}_manifest.json per the protocol in
docs/evaluation_protocol.md §3.

Selection rule (frozen, §3.1):
  1. load instances_val2017.json
  2. filter: iscrowd==0, valid bbox (w>0,h>0), decodable non-empty seg (annToMask)
  3. numpy.random.seed(2027)
  4. per-category shuffle, round-robin sample across categories until target N
     (broad category coverage, not concentrated in few classes)
  5. NOT size-biased: equal-probability sampling across size buckets
  6. stop at target N; record category count + size-bucket distribution

Stage 5A: framework only. Running requires data + pycocotools.
  - dataset missing  -> DATASET_MISSING, exit 2
  - pycocotools miss -> MISSING_DEPENDENCY: pycocotools, exit 3
  - does NOT fabricate manifest instance content.

This script does NOT rebuild engines, download data, or modify upstream.
"""

import argparse
import json
import os
import sys


SUBSET_TARGETS = {
    "smoke": 20,
    "quick": 100,
    "standard": 500,
}
SEED = 2027


def _check_pycocotools():
    try:
        from pycocotools.coco import COCO  # noqa: F401
        return True
    except Exception:
        return False


def _size_bucket(area):
    if area < 32 * 32:
        return "small"
    if area <= 96 * 96:
        return "medium"
    return "large"


def build_legal_instances(coco, coco_root):
    """Return list of legal instance dicts (status OK) grouped for sampling."""
    import numpy as np
    legal = []
    for img_id in coco.getImgIds():
        anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id))
        for ann in anns:
            if ann.get("iscrowd", 0) == 1:
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            try:
                gt = coco.annToMask(ann)
            except Exception:
                continue
            if gt.sum() == 0:
                continue
            cat_id = ann["category_id"]
            cat = coco.loadCats(cat_id)[0]
            img_info = coco.loadImgs(ann["image_id"])[0]
            legal.append({
                "image_id": ann["image_id"],
                "annotation_id": ann["id"],
                "category_id": cat_id,
                "category_name": cat["name"],
                "image_path": os.path.join(coco_root, img_info["file_name"]),
                "bbox_xywh": [float(x), float(y), float(w), float(h)],
                "area": float(ann["area"]),
                "size_bucket": _size_bucket(float(ann["area"])),
                "iscrowd": 0,
            })
    return legal


def sample_subset(legal, target):
    """Round-robin across categories with a fixed seed, until target reached.

    Broad category coverage (cycle through all categories before repeating),
    not size-biased (inner per-category order is shuffled, not sorted by area).
    """
    import numpy as np
    np.random.seed(SEED)
    by_cat = {}
    for inst in legal:
        by_cat.setdefault(inst["category_name"], []).append(inst)
    for cat in by_cat:
        np.random.shuffle(by_cat[cat])
    cat_order = sorted(by_cat.keys())
    selected = []
    idx = {cat: 0 for cat in cat_order}
    while len(selected) < target:
        progressed = False
        for cat in cat_order:
            if idx[cat] < len(by_cat[cat]):
                selected.append(by_cat[cat][idx[cat]])
                idx[cat] += 1
                progressed = True
                if len(selected) >= target:
                    break
        if not progressed:
            break  # exhausted all legal instances before reaching target
    return selected


def main():
    parser = argparse.ArgumentParser(description="Deterministic COCO subset manifest generator (Stage 5A)")
    parser.add_argument("--coco-root", default="data/coco/val2017")
    parser.add_argument("--annotations", default="data/coco/annotations/instances_val2017.json")
    parser.add_argument("--subset", choices=["smoke", "quick", "standard"], required=True)
    parser.add_argument("--output-dir", default="results/coco_subsets")
    args = parser.parse_args()

    if not os.path.isfile(args.annotations):
        print(f"DATASET_MISSING: annotations not found: {args.annotations}")
        sys.exit(2)
    if not os.path.isdir(args.coco_root):
        print(f"DATASET_MISSING: val2017 images dir not found: {args.coco_root}")
        sys.exit(2)
    if not _check_pycocotools():
        print("MISSING_DEPENDENCY: pycocotools")
        sys.exit(3)

    from pycocotools.coco import COCO
    coco = COCO(args.annotations)
    legal = build_legal_instances(coco, args.coco_root)
    target = SUBSET_TARGETS[args.subset]
    selected = sample_subset(legal, target)

    cats = sorted({i["category_name"] for i in selected})
    buckets = {"small": 0, "medium": 0, "large": 0}
    for i in selected:
        buckets[i["size_bucket"]] += 1

    manifest = {
        "subset": args.subset,
        "seed": SEED,
        "target_instances": target,
        "num_instances": len(selected),
        "num_categories": len(cats),
        "size_bucket_counts": buckets,
        "instances": selected,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"{args.subset}_manifest.json")
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"WROTE {out}: {len(selected)} instances, {len(cats)} categories, buckets={buckets}")
    sys.exit(0)


if __name__ == "__main__":
    main()
