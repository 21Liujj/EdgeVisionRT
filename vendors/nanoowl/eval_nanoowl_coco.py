#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate the accepted NanoOWL deployment on COCO 2017 val bounding boxes.

The evaluator caches all COCO text embeddings once, runs NanoOWL once per image
at a low score threshold, and derives threshold/top-k policies offline. Formal
metrics are produced only by pycocotools COCOeval with iouType="bbox".
"""
import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from nanoowl.owl_predictor import OwlPredictor

DEFAULT_MODEL = "google/owlvit-base-patch32"
DEFAULT_THRESHOLD = 0.10
DEFAULT_TOP_K = 100
RAW_THRESHOLD = 0.05
THRESHOLD_POLICIES = (0.05, 0.10, 0.20, 0.30)
TOP_K_POLICIES = (10, 50, 100)
COCO_STATS = (
    "AP", "AP50", "AP75", "APsmall", "APmedium", "APlarge",
    "AR1", "AR10", "AR100", "ARsmall", "ARmedium", "ARlarge",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="NanoOWL open-vocabulary COCO bbox evaluator"
    )
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--prompt-template", default="a {category_name}")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--raw-threshold", type=float, default=RAW_THRESHOLD)
    parser.add_argument("--visualization-count", type=int, default=0)
    parser.add_argument("--category-min-gt", type=int, default=5)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n")


def finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def quantiles(values):
    if not values:
        return {
            "count": 0, "min": None, "mean": None, "median": None,
            "p05": None, "p10": None, "p90": None, "p95": None,
            "p99": None, "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def prompt_for(template, category_name):
    try:
        return template.format(
            category_name=category_name,
            category=category_name,
        )
    except (KeyError, IndexError, ValueError) as error:
        raise ValueError(
            "--prompt-template must use {category_name} or {category}"
        ) from error


def resolve_image_ids(coco, manifest_path, max_images):
    all_ids = sorted(coco.getImgIds())
    if manifest_path:
        document = json.loads(Path(manifest_path).read_text())
        if isinstance(document, list):
            image_ids = [int(value) for value in document]
        else:
            image_ids = [int(value) for value in document["image_ids"]]
    else:
        image_ids = all_ids

    if len(image_ids) != len(set(image_ids)):
        raise ValueError("manifest contains duplicate image IDs")
    unknown = sorted(set(image_ids) - set(all_ids))
    if unknown:
        raise ValueError("manifest contains unknown image IDs: %s" % unknown[:10])
    if max_images is not None:
        if max_images <= 0:
            raise ValueError("--max-images must be positive")
        image_ids = image_ids[:max_images]
    if not image_ids:
        raise ValueError("evaluation image set is empty")
    return image_ids


def build_prompt_mapping(coco, template):
    categories = sorted(coco.loadCats(coco.getCatIds()), key=lambda item: item["id"])
    mapping = []
    for prompt_index, category in enumerate(categories):
        mapping.append({
            "prompt_index": prompt_index,
            "category_id": int(category["id"]),
            "category_name": category["name"],
            "prompt": prompt_for(template, category["name"]),
        })
    if len(mapping) != 80:
        raise ValueError("expected 80 COCO categories, got %d" % len(mapping))
    if len({item["prompt_index"] for item in mapping}) != 80:
        raise ValueError("prompt indices are not unique")
    if len({item["category_id"] for item in mapping}) != 80:
        raise ValueError("COCO category IDs are not unique")
    return mapping


def validate_paths(args, image_infos):
    for path in (args.images_dir, args.annotations, args.engine):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    missing = [
        info["file_name"] for info in image_infos
        if not (Path(args.images_dir) / info["file_name"]).is_file()
    ]
    if missing:
        raise FileNotFoundError("missing COCO images: %s" % missing[:10])


def clip_box_xyxy(box, width, height):
    x1, y1, x2, y2 = [float(value) for value in box]
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None, "non_finite"
    if x2 < x1 or y2 < y1:
        return None, "negative_extent"
    x1 = min(max(x1, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    x2 = min(max(x2, 0.0), float(width))
    y2 = min(max(y2, 0.0), float(height))
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0.0 or box_height <= 0.0:
        return None, "degenerate_after_clip"
    return (x1, y1, box_width, box_height), None


def run_inference(args, predictor, text_encodings, prompts, image_infos):
    candidates_by_image = {}
    invalid = Counter()
    raw_counts = []
    image_seconds = []

    loop_start = time.perf_counter()
    for index, info in enumerate(image_infos, 1):
        image_path = Path(args.images_dir) / info["file_name"]
        image = Image.open(image_path).convert("RGB")
        start = time.perf_counter()
        output = predictor.predict(
            image=image,
            text=prompts,
            text_encodings=text_encodings,
            threshold=args.raw_threshold,
            pad_square=False,
        )
        labels = output.labels.detach().cpu().numpy()
        scores = output.scores.detach().float().cpu().numpy()
        boxes = output.boxes.detach().float().cpu().numpy()
        elapsed = time.perf_counter() - start
        image_seconds.append(elapsed)

        if labels.ndim != 1 or scores.ndim != 1 or boxes.ndim != 2 or boxes.shape[1] != 4:
            raise RuntimeError(
                "unexpected NanoOWL output shapes labels=%s scores=%s boxes=%s"
                % (labels.shape, scores.shape, boxes.shape)
            )
        if not (len(labels) == len(scores) == len(boxes)):
            raise RuntimeError("NanoOWL output lengths differ")

        rows = []
        for label, score, box in zip(labels, scores, boxes):
            label_index = int(label)
            score = float(score)
            if label_index < 0 or label_index >= len(prompts):
                invalid["invalid_prompt_index"] += 1
                continue
            if not math.isfinite(score):
                invalid["non_finite_score"] += 1
                continue
            clipped, reason = clip_box_xyxy(box, info["width"], info["height"])
            if reason is not None:
                invalid[reason] += 1
                continue
            x, y, width, height = clipped
            rows.append((
                float(label_index), score, x, y, width, height,
            ))
        rows.sort(key=lambda row: row[1], reverse=True)
        if rows:
            candidates_by_image[int(info["id"])] = np.asarray(rows, dtype=np.float32)
        else:
            candidates_by_image[int(info["id"])] = np.empty((0, 6), dtype=np.float32)
        raw_counts.append(len(rows))

        if index == 1 or index % 25 == 0 or index == len(image_infos):
            print(
                "inference %d/%d image_id=%s candidates=%d elapsed=%.1fs"
                % (index, len(image_infos), info["id"], len(rows), time.perf_counter() - loop_start),
                flush=True,
            )

    return {
        "candidates_by_image": candidates_by_image,
        "invalid_counts": dict(invalid),
        "raw_candidate_counts": raw_counts,
        "per_image_seconds": image_seconds,
        "image_loop_seconds": time.perf_counter() - loop_start,
    }


def policy_predictions(candidates_by_image, image_ids, prompt_to_category, threshold, top_k):
    predictions = []
    per_image_counts = []
    for image_id in image_ids:
        rows = candidates_by_image[image_id]
        if rows.size:
            rows = rows[rows[:, 1] > threshold]
            rows = rows[:top_k]
        per_image_counts.append(int(len(rows)))
        for row in rows:
            prompt_index = int(row[0])
            record = {
                "image_id": int(image_id),
                "category_id": int(prompt_to_category[prompt_index]),
                "bbox": [float(value) for value in row[2:6]],
                "score": float(row[1]),
            }
            if not all(finite_number(value) for value in record["bbox"] + [record["score"]]):
                raise RuntimeError("non-finite prediction generated")
            if record["bbox"][2] <= 0.0 or record["bbox"][3] <= 0.0:
                raise RuntimeError("non-positive COCO bbox generated")
            predictions.append(record)
    return predictions, per_image_counts


def empty_detection_coco(coco_gt):
    result = COCO()
    result.dataset = {
        "images": list(coco_gt.dataset.get("images", [])),
        "categories": list(coco_gt.dataset.get("categories", [])),
        "annotations": [],
    }
    with contextlib.redirect_stdout(io.StringIO()):
        result.createIndex()
    return result


def evaluate_coco(coco_gt, predictions, image_ids):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        if predictions:
            coco_dt = coco_gt.loadRes(predictions)
        else:
            coco_dt = empty_detection_coco(coco_gt)
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.params.imgIds = list(image_ids)
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    stats = {
        name: float(value) for name, value in zip(COCO_STATS, evaluator.stats.tolist())
    }
    return stats, evaluator, stream.getvalue()


def mean_valid(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[values > -1]
    return float(np.mean(values)) if values.size else None


def category_metrics(coco, evaluator, image_ids, predictions, min_gt_count):
    precision = evaluator.eval["precision"]
    area_labels = list(evaluator.params.areaRngLbl)
    max_det_index = len(evaluator.params.maxDets) - 1
    all_area_index = area_labels.index("all")
    small_area_index = area_labels.index("small")
    large_area_index = area_labels.index("large")
    iou50_index = int(np.where(np.isclose(evaluator.params.iouThrs, 0.50))[0][0])

    gt_counts = Counter()
    noncrowd_counts = Counter()
    for annotation in coco.loadAnns(coco.getAnnIds(imgIds=image_ids)):
        category_id = int(annotation["category_id"])
        gt_counts[category_id] += 1
        if not annotation.get("iscrowd", 0):
            noncrowd_counts[category_id] += 1
    prediction_counts = Counter(int(item["category_id"]) for item in predictions)

    category_ids = list(evaluator.params.catIds)
    category_names = {item["id"]: item["name"] for item in coco.loadCats(category_ids)}
    records = []
    for category_index, category_id in enumerate(category_ids):
        category_precision = precision[:, :, category_index, all_area_index, max_det_index]
        category_ap50 = precision[iou50_index, :, category_index, all_area_index, max_det_index]
        small_precision = precision[:, :, category_index, small_area_index, max_det_index]
        large_precision = precision[:, :, category_index, large_area_index, max_det_index]
        ap_small = mean_valid(small_precision)
        ap_large = mean_valid(large_precision)
        records.append({
            "category_id": int(category_id),
            "category_name": category_names[category_id],
            "gt_count": int(gt_counts[category_id]),
            "noncrowd_gt_count": int(noncrowd_counts[category_id]),
            "prediction_count": int(prediction_counts[category_id]),
            "AP": mean_valid(category_precision),
            "AP50": mean_valid(category_ap50),
            "APsmall": ap_small,
            "APlarge": ap_large,
            "large_minus_small_AP": (
                float(ap_large - ap_small)
                if ap_large is not None and ap_small is not None else None
            ),
        })

    eligible = [
        item for item in records
        if item["gt_count"] >= min_gt_count and item["AP"] is not None
    ]
    best = sorted(eligible, key=lambda item: item["AP"], reverse=True)[:10]
    worst = sorted(eligible, key=lambda item: item["AP"])[:10]
    size_gap = sorted(
        [item for item in eligible if item["large_minus_small_AP"] is not None],
        key=lambda item: item["large_minus_small_AP"],
        reverse=True,
    )[:10]
    frequent_low_ap = sorted(
        eligible,
        key=lambda item: (-item["gt_count"], item["AP"]),
    )[:20]
    return {
        "minimum_gt_count_for_rankings": min_gt_count,
        "categories": records,
        "best_categories": best,
        "worst_categories": worst,
        "largest_large_minus_small_AP": size_gap,
        "high_frequency_review_pool": frequent_low_ap,
    }


def xywh_iou(first, second):
    ax1, ay1, aw, ah = first
    bx1, by1, bw, bh = second
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0.0 else 0.0


def analysis_match(coco, image_ids, predictions):
    predictions_by_image = defaultdict(list)
    for prediction in predictions:
        predictions_by_image[int(prediction["image_id"])].append(prediction)
    category_names = {
        int(item["id"]): item["name"] for item in coco.loadCats(coco.getCatIds())
    }

    image_records = []
    confusion_counts = Counter()
    for image_id in image_ids:
        ground_truth = [
            annotation for annotation in coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
            if not annotation.get("iscrowd", 0)
        ]
        predictions_for_image = sorted(
            predictions_by_image.get(image_id, []),
            key=lambda item: item["score"],
            reverse=True,
        )
        unmatched_gt = set(range(len(ground_truth)))
        matched_predictions = set()
        matches = []

        for prediction_index, prediction in enumerate(predictions_for_image):
            candidates = []
            for gt_index in unmatched_gt:
                annotation = ground_truth[gt_index]
                if int(annotation["category_id"]) != int(prediction["category_id"]):
                    continue
                overlap = xywh_iou(prediction["bbox"], annotation["bbox"])
                candidates.append((overlap, gt_index))
            if not candidates:
                continue
            overlap, gt_index = max(candidates)
            if overlap >= 0.50:
                unmatched_gt.remove(gt_index)
                matched_predictions.add(prediction_index)
                matches.append({
                    "prediction_index": prediction_index,
                    "annotation_id": int(ground_truth[gt_index]["id"]),
                    "iou": float(overlap),
                })

        semantic_confusions = []
        for prediction_index, prediction in enumerate(predictions_for_image):
            if prediction_index in matched_predictions:
                continue
            best = None
            for annotation in ground_truth:
                if int(annotation["category_id"]) == int(prediction["category_id"]):
                    continue
                overlap = xywh_iou(prediction["bbox"], annotation["bbox"])
                if best is None or overlap > best[0]:
                    best = (overlap, annotation)
            if best is not None and best[0] >= 0.50:
                annotation = best[1]
                predicted_name = category_names[int(prediction["category_id"])]
                gt_name = category_names[int(annotation["category_id"])]
                confusion_counts[(gt_name, predicted_name)] += 1
                semantic_confusions.append({
                    "prediction_index": prediction_index,
                    "predicted_category_id": int(prediction["category_id"]),
                    "predicted_category_name": predicted_name,
                    "gt_annotation_id": int(annotation["id"]),
                    "gt_category_id": int(annotation["category_id"]),
                    "gt_category_name": gt_name,
                    "iou": float(best[0]),
                    "score": float(prediction["score"]),
                })

        unmatched_annotations = [ground_truth[index] for index in sorted(unmatched_gt)]
        image_records.append({
            "image_id": int(image_id),
            "gt_count": len(ground_truth),
            "prediction_count": len(predictions_for_image),
            "tp_iou50": len(matches),
            "fp_iou50": len(predictions_for_image) - len(matches),
            "fn_iou50": len(unmatched_annotations),
            "small_fn_iou50": sum(
                1 for annotation in unmatched_annotations if float(annotation.get("area", 0.0)) < 32.0 ** 2
            ),
            "semantic_confusion_count": len(semantic_confusions),
            "semantic_confusions": semantic_confusions[:20],
        })

    successful = sorted(
        [item for item in image_records if item["tp_iou50"] > 0],
        key=lambda item: (
            item["fp_iou50"] + item["fn_iou50"],
            -item["tp_iou50"],
            item["image_id"],
        ),
    )[:10]
    fp_heavy = sorted(
        image_records,
        key=lambda item: (item["fp_iou50"], item["prediction_count"], -item["image_id"]),
        reverse=True,
    )[:10]
    fn_heavy = sorted(
        image_records,
        key=lambda item: (item["fn_iou50"], item["gt_count"], -item["image_id"]),
        reverse=True,
    )[:10]
    small_failure = sorted(
        image_records,
        key=lambda item: (item["small_fn_iou50"], item["fn_iou50"], -item["image_id"]),
        reverse=True,
    )[:10]
    semantic_confusion = sorted(
        [item for item in image_records if item["semantic_confusion_count"] > 0],
        key=lambda item: (item["semantic_confusion_count"], item["fp_iou50"], -item["image_id"]),
        reverse=True,
    )[:10]

    confusion_pairs = [
        {
            "gt_category": pair[0],
            "predicted_category": pair[1],
            "count": count,
        }
        for pair, count in confusion_counts.most_common(30)
    ]
    return {
        "matcher": "analysis-only greedy class-aware matching at IoU >= 0.50; not COCOeval",
        "groups": {
            "successful_detections": successful,
            "false_positive_heavy": fp_heavy,
            "false_negative_heavy": fn_heavy,
            "small_object_failure": small_failure,
            "semantic_confusion": semantic_confusion,
        },
        "semantic_confusion_pairs": confusion_pairs,
        "per_image": image_records,
    }


def detection_statistics(coco, image_ids, predictions, per_image_counts, invalid_counts):
    gt_counts = []
    for image_id in image_ids:
        gt_counts.append(len(coco.getAnnIds(imgIds=[image_id])))
    scores = [float(item["score"]) for item in predictions]
    box_areas = [float(item["bbox"][2] * item["bbox"][3]) for item in predictions]
    area_buckets = {
        "small": sum(area < 32.0 ** 2 for area in box_areas),
        "medium": sum(32.0 ** 2 <= area < 96.0 ** 2 for area in box_areas),
        "large": sum(area >= 96.0 ** 2 for area in box_areas),
    }
    return {
        "total_images": len(image_ids),
        "total_predictions": len(predictions),
        "images_with_zero_predictions": sum(count == 0 for count in per_image_counts),
        "predictions_per_image": quantiles(per_image_counts),
        "gt_objects_per_image": quantiles(gt_counts),
        "score_distribution": quantiles(scores),
        "bbox_area_distribution_pixels": quantiles(box_areas),
        "bbox_area_bucket_counts": area_buckets,
        "invalid_raw_predictions": invalid_counts,
    }


def draw_label(draw, position, text, color):
    font = ImageFont.load_default()
    left, top, right, bottom = draw.textbbox(position, text, font=font)
    draw.rectangle((left - 2, top - 1, right + 2, bottom + 1), fill=(0, 0, 0))
    draw.text(position, text, fill=color, font=font)


def create_visualizations(coco, images_dir, predictions, failures, output_dir, total_count):
    if total_count <= 0:
        return []
    group_order = [
        "successful_detections", "false_positive_heavy", "false_negative_heavy",
        "small_object_failure", "semantic_confusion",
    ]
    per_group = max(1, total_count // len(group_order))
    predictions_by_image = defaultdict(list)
    for prediction in predictions:
        predictions_by_image[int(prediction["image_id"])].append(prediction)
    category_names = {
        int(item["id"]): item["name"] for item in coco.loadCats(coco.getCatIds())
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for group_name in group_order:
        group = failures["groups"].get(group_name, [])
        for rank, record in enumerate(group[:per_group], 1):
            image_id = int(record["image_id"])
            info = coco.loadImgs([image_id])[0]
            image = Image.open(Path(images_dir) / info["file_name"]).convert("RGB")
            draw = ImageDraw.Draw(image)
            for annotation in coco.loadAnns(coco.getAnnIds(imgIds=[image_id])):
                x, y, width, height = [float(value) for value in annotation["bbox"]]
                draw.rectangle((x, y, x + width, y + height), outline=(0, 255, 0), width=2)
                draw_label(
                    draw, (x, max(0.0, y - 11.0)),
                    "GT " + category_names[int(annotation["category_id"])],
                    (0, 255, 0),
                )
            ranked_predictions = sorted(
                predictions_by_image.get(image_id, []),
                key=lambda item: item["score"],
                reverse=True,
            )[:30]
            for prediction in ranked_predictions:
                x, y, width, height = prediction["bbox"]
                draw.rectangle((x, y, x + width, y + height), outline=(255, 64, 64), width=2)
                draw_label(
                    draw, (x, min(float(info["height"]) - 12.0, y + 2.0)),
                    "%s %.3f" % (
                        category_names[int(prediction["category_id"])],
                        prediction["score"],
                    ),
                    (255, 96, 96),
                )
            filename = "%s_%02d_image_%012d.jpg" % (group_name, rank, image_id)
            image.save(output_dir / filename, quality=90)
            written.append({
                "group": group_name,
                "rank": rank,
                "image_id": image_id,
                "path": str(output_dir / filename),
                "drawn_predictions": len(ranked_predictions),
            })
            if len(written) >= total_count:
                return written
    return written


def evaluate_policy(coco, candidates, image_ids, prompt_to_category, threshold, top_k):
    predictions, per_image_counts = policy_predictions(
        candidates, image_ids, prompt_to_category, threshold, top_k
    )
    metrics, evaluator, stdout = evaluate_coco(coco, predictions, image_ids)
    summary = {
        "threshold": float(threshold),
        "top_k_per_image": int(top_k),
        "prediction_count": len(predictions),
        "zero_prediction_images": sum(count == 0 for count in per_image_counts),
        "metrics": metrics,
    }
    return summary, predictions, per_image_counts, evaluator, stdout


def main():
    args = parse_args()
    total_start = time.perf_counter()
    if args.raw_threshold > min(THRESHOLD_POLICIES + (args.threshold,)):
        raise ValueError("--raw-threshold must not exceed any evaluated threshold")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coco = COCO(args.annotations)
    image_ids = resolve_image_ids(coco, args.manifest, args.max_images)
    image_infos = coco.loadImgs(image_ids)
    info_by_id = {int(item["id"]): item for item in image_infos}
    image_infos = [info_by_id[image_id] for image_id in image_ids]
    validate_paths(args, image_infos)

    mapping = build_prompt_mapping(coco, args.prompt_template)
    prompts = [item["prompt"] for item in mapping]
    prompt_to_category = [item["category_id"] for item in mapping]
    write_json(output_dir / "category_prompt_mapping.json", {
        "prompt_template": args.prompt_template,
        "count": len(mapping),
        "text_embeddings_cached_once": True,
        "mapping": mapping,
    })

    load_start = time.perf_counter()
    predictor = OwlPredictor(args.model, image_encoder_engine=args.engine)
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_start

    text_start = time.perf_counter()
    text_encodings = predictor.encode_text(prompts)
    torch.cuda.synchronize()
    text_encoding_seconds = time.perf_counter() - text_start
    if tuple(text_encodings.text_embeds.shape[:1]) != (80,):
        raise RuntimeError("expected 80 cached text embeddings, got %s" % (text_encodings.text_embeds.shape,))

    inference = run_inference(args, predictor, text_encodings, prompts, image_infos)
    candidates = inference["candidates_by_image"]

    primary_summary, primary_predictions, primary_counts, primary_eval, primary_stdout = evaluate_policy(
        coco, candidates, image_ids, prompt_to_category, args.threshold, args.top_k
    )
    write_json(output_dir / "coco_detections.json", primary_predictions)
    (output_dir / "coco_eval_stdout.txt").write_text(primary_stdout)
    write_json(output_dir / "coco_eval_summary.json", {
        "protocol": "NanoOWL open-vocabulary detection; COCO category-name text prompts",
        "iou_type": "bbox",
        "image_count": len(image_ids),
        "gt_instance_count": len(coco.getAnnIds(imgIds=image_ids)),
        **primary_summary,
    })

    category = category_metrics(
        coco, primary_eval, image_ids, primary_predictions, args.category_min_gt
    )
    category["scope"] = {
        "image_count": len(image_ids),
        "gt_instance_count": len(coco.getAnnIds(imgIds=image_ids)),
        "threshold": args.threshold,
        "top_k_per_image": args.top_k,
    }
    write_json(output_dir / "category_analysis.json", category)
    del primary_eval
    gc.collect()

    statistics_doc = detection_statistics(
        coco, image_ids, primary_predictions, primary_counts,
        inference["invalid_counts"],
    )
    write_json(output_dir / "detection_statistics.json", statistics_doc)

    failures = analysis_match(coco, image_ids, primary_predictions)
    visualizations = create_visualizations(
        coco, args.images_dir, primary_predictions, failures,
        output_dir / "visualizations", args.visualization_count,
    )
    failures["visualizations"] = visualizations
    write_json(output_dir / "failure_cases.json", failures)

    threshold_results = []
    top_k_results = []
    policy_cache = {
        (float(args.threshold), int(args.top_k)): primary_summary,
    }

    def get_policy(threshold, top_k):
        key = (float(threshold), int(top_k))
        if key in policy_cache:
            return policy_cache[key]
        summary, predictions, counts, evaluator, stdout = evaluate_policy(
            coco, candidates, image_ids, prompt_to_category, threshold, top_k
        )
        policy_cache[key] = summary
        del predictions, counts, evaluator, stdout
        gc.collect()
        return summary

    for threshold in THRESHOLD_POLICIES:
        threshold_results.append(get_policy(threshold, 100))
    for top_k in TOP_K_POLICIES:
        top_k_results.append(get_policy(0.10, top_k))
    write_json(output_dir / "threshold_ablation.json", {
        "fixed_top_k_per_image": 100,
        "single_inference_raw_threshold": args.raw_threshold,
        "policies": threshold_results,
    })
    write_json(output_dir / "topk_ablation.json", {
        "fixed_threshold": 0.10,
        "policies": top_k_results,
    })

    artifact_path = output_dir / "coco_detections.json"
    timing = {
        "model_load_seconds": model_load_seconds,
        "text_encoding_seconds": text_encoding_seconds,
        "image_loop_seconds": inference["image_loop_seconds"],
        "image_loop_images_per_second": len(image_ids) / inference["image_loop_seconds"],
        "per_image_seconds": quantiles(inference["per_image_seconds"]),
        "total_wall_seconds": time.perf_counter() - total_start,
    }
    metadata = {
        "stage": "B2",
        "model": args.model,
        "engine": {
            "path": str(Path(args.engine)),
            "size_bytes": Path(args.engine).stat().st_size,
            "sha256": sha256_file(args.engine),
        },
        "annotations": {
            "path": str(Path(args.annotations)),
            "size_bytes": Path(args.annotations).stat().st_size,
            "sha256": sha256_file(args.annotations),
        },
        "manifest": (
            {
                "path": str(Path(args.manifest)),
                "size_bytes": Path(args.manifest).stat().st_size,
                "sha256": sha256_file(args.manifest),
            } if args.manifest else None
        ),
        "image_count": len(image_ids),
        "image_ids": image_ids,
        "gt_instance_count": len(coco.getAnnIds(imgIds=image_ids)),
        "prompt_template": args.prompt_template,
        "prompt_count": len(prompts),
        "text_embeddings_cached_once": True,
        "text_embedding_shape": list(text_encodings.text_embeds.shape),
        "raw_inference_threshold": args.raw_threshold,
        "primary_threshold": args.threshold,
        "primary_top_k_per_image": args.top_k,
        "pad_square": False,
        "candidate_contract": "per-patch prompt argmax, sigmoid score, score > threshold; no NMS",
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pycocotools": importlib.metadata.version("pycocotools"),
            "pillow": PIL.__version__,
        },
        "raw_candidate_counts": quantiles(inference["raw_candidate_counts"]),
        "timing": timing,
        "primary_detection_artifact": {
            "path": str(artifact_path),
            "size_bytes": artifact_path.stat().st_size,
            "sha256": sha256_file(artifact_path),
        },
    }
    write_json(output_dir / "run_metadata.json", metadata)

    print(json.dumps({
        "status": "PASS",
        "image_count": len(image_ids),
        "gt_instance_count": metadata["gt_instance_count"],
        "primary": primary_summary,
        "timing": timing,
        "output_dir": str(output_dir),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
