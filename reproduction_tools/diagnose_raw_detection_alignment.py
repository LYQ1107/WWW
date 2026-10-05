#!/usr/bin/env python3
"""Diagnose raw COCO prediction alignment against VisionTrack annotations.

This tool deliberately bypasses TrackEval, COCO->MOT conversion, and the
cross-view converter.  It is a diagnostic only: it never changes predictions,
annotations, thresholds in the model, or official evaluation artifacts.

The default matching is category-aware and uses the raw XYWH boxes.  The
reported TP/FP/FN values use deterministic greedy one-to-one matching at IoU
0.5; the recall-at-IoU values use the requested per-GT max-IoU diagnostic.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Callable, DefaultDict, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = Path("/data/DATASETS/TRACKING/JDE/VisionTrack")
DEFAULT_MODEL_SIZE = 1560.0
IOU_THRESHOLDS = (0.1, 0.3, 0.5, 0.75)
SCORE_THRESHOLDS = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60)
FRAME_OFFSETS = (-2, -1, 0, 1, 2)


Box = Tuple[float, float, float, float]
Prediction = Dict[str, Any]
ImageInfo = Dict[str, Any]
GroundTruth = Dict[str, Any]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: Sequence[float], quantile: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def describe(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p1": None,
            "p5": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "count": len(values),
        "min": float(min(values)),
        "p1": percentile(values, 0.01),
        "p5": percentile(values, 0.05),
        "p25": percentile(values, 0.25),
        "median": float(statistics.median(values)),
        "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": float(max(values)),
    }


def xywh_to_xyxy(box: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, width, height = (float(value) for value in box)
    return x, y, x + width, y + height


def iou_xywh(first: Sequence[float], second: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = xywh_to_xyxy(first)
    bx1, by1, bx2, by2 = xywh_to_xyxy(second)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def image_key(image: ImageInfo) -> Tuple[str, int, int]:
    return (
        str(image["file_name"]).split("/", 1)[0].rsplit("_", 1)[0],
        int(image["view_id"]),
        int(image["frame_id"]),
    )


def scene_name(image: ImageInfo) -> str:
    return str(image["file_name"]).split("/", 1)[0].rsplit("_", 1)[0]


def view_name(image: ImageInfo) -> str:
    return f"{scene_name(image)}_View{int(image['view_id'])}"


def video_name(image: ImageInfo) -> str:
    return str(int(image["video_id"]))


def group_names(image: ImageInfo) -> Dict[str, str]:
    return {
        "scene": scene_name(image),
        "view": view_name(image),
        "video": video_name(image),
        "video_view": f"{video_name(image)}_View{int(image['view_id'])}",
    }


def prediction_box(prediction: Prediction) -> Box:
    bbox = prediction["bbox"]
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError("prediction bbox must be a four-element XYWH list")
    return tuple(float(value) for value in bbox)  # type: ignore[return-value]


def transformed_box(prediction: Prediction, image: ImageInfo, variant: str, model_width: float, model_height: float) -> Box:
    x, y, width, height = prediction_box(prediction)
    if variant == "raw":
        return x, y, width, height
    if variant == "model_to_image":
        sx = float(image["width"]) / model_width
        sy = float(image["height"]) / model_height
        return x * sx, y * sy, width * sx, height * sy
    raise ValueError(f"unknown box variant: {variant}")


def empty_stats() -> Dict[str, Any]:
    return {
        "gt_count": 0,
        "prediction_count": 0,
        "max_ious": [],
        "tp_at_iou_0.5": 0,
        "fp_at_iou_0.5": 0,
        "fn_at_iou_0.5": 0,
        "images_with_gt": 0,
        "images_with_predictions": 0,
    }


def update_stats(
    stats: MutableMapping[str, Any],
    predictions: Sequence[Prediction],
    ground_truth: Sequence[GroundTruth],
    image: ImageInfo,
    variant: str,
    model_width: float,
    model_height: float,
) -> None:
    stats["gt_count"] += len(ground_truth)
    stats["prediction_count"] += len(predictions)
    if ground_truth:
        stats["images_with_gt"] += 1
    if predictions:
        stats["images_with_predictions"] += 1

    pred_boxes = [
        (transformed_box(prediction, image, variant, model_width, model_height), int(prediction.get("category_id", -1)))
        for prediction in predictions
    ]
    gt_boxes = [(tuple(float(value) for value in gt["bbox"]), int(gt.get("category_id", -1))) for gt in ground_truth]

    for gt_box, gt_category in gt_boxes:
        candidates = [
            iou_xywh(gt_box, pred_box)
            for pred_box, pred_category in pred_boxes
            if pred_category == gt_category
        ]
        stats["max_ious"].append(max(candidates) if candidates else 0.0)

    pairs: List[Tuple[float, int, int]] = []
    for prediction_index, (pred_box, pred_category) in enumerate(pred_boxes):
        for gt_index, (gt_box, gt_category) in enumerate(gt_boxes):
            if pred_category != gt_category:
                continue
            overlap = iou_xywh(pred_box, gt_box)
            if overlap >= 0.5:
                pairs.append((overlap, prediction_index, gt_index))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    matched_predictions = set()
    matched_ground_truth = set()
    for _overlap, prediction_index, gt_index in pairs:
        if prediction_index in matched_predictions or gt_index in matched_ground_truth:
            continue
        matched_predictions.add(prediction_index)
        matched_ground_truth.add(gt_index)
    stats["tp_at_iou_0.5"] += len(matched_predictions)
    stats["fp_at_iou_0.5"] += len(predictions) - len(matched_predictions)
    stats["fn_at_iou_0.5"] += len(ground_truth) - len(matched_ground_truth)


def finalize_stats(stats: Mapping[str, Any]) -> Dict[str, Any]:
    max_ious = list(stats["max_ious"])
    gt_count = int(stats["gt_count"])
    prediction_count = int(stats["prediction_count"])
    tp = int(stats["tp_at_iou_0.5"])
    fp = int(stats["fp_at_iou_0.5"])
    fn = int(stats["fn_at_iou_0.5"])
    recalls = {
        f"{threshold:g}": sum(value >= threshold for value in max_ious) / gt_count if gt_count else None
        for threshold in IOU_THRESHOLDS
    }
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2.0 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {
        "gt_count": gt_count,
        "prediction_count": prediction_count,
        "mean_max_iou": sum(max_ious) / len(max_ious) if max_ious else None,
        "median_max_iou": statistics.median(max_ious) if max_ious else None,
        "recall_at_iou_by_gt_max": recalls,
        "precision_at_iou_0.5": precision,
        "recall_at_iou_0.5": recall,
        "f1_at_iou_0.5": f1,
        "tp_at_iou_0.5": tp,
        "fp_at_iou_0.5": fp,
        "fn_at_iou_0.5": fn,
        "images_with_gt": int(stats["images_with_gt"]),
        "images_with_predictions": int(stats["images_with_predictions"]),
    }


def new_group_stats() -> Dict[str, Dict[str, Any]]:
    return defaultdict(empty_stats)  # type: ignore[return-value]


def load_inputs(annotation_path: Path, prediction_path: Path) -> Tuple[Dict[int, ImageInfo], Dict[int, List[GroundTruth]], List[Prediction]]:
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    images: Dict[int, ImageInfo] = {int(image["id"]): image for image in annotations["images"]}
    ground_truth: DefaultDict[int, List[GroundTruth]] = defaultdict(list)
    for item in annotations.get("annotations", []):
        ground_truth[int(item["image_id"])].append(item)
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise ValueError("prediction JSON must contain a list")
    return images, ground_truth, predictions


def parse_scene_filter(values: Optional[Sequence[str]]) -> Optional[set[str]]:
    if not values:
        return None
    result: set[str] = set()
    for value in values:
        result.update(part.strip() for part in value.split(",") if part.strip())
    return result or None


def selected_image_ids(images: Mapping[int, ImageInfo], scenes: Optional[set[str]]) -> set[int]:
    if scenes is None:
        return set(images)
    return {image_id for image_id, image in images.items() if scene_name(image) in scenes}


def make_predictions_by_image(
    predictions: Sequence[Prediction],
    images: Mapping[int, ImageInfo],
    selected_ids: set[int],
) -> Tuple[DefaultDict[int, List[Prediction]], Dict[str, Any]]:
    by_image: DefaultDict[int, List[Prediction]] = defaultdict(list)
    unknown_ids: List[int] = []
    outside_selected = 0
    category_counts: Counter[str] = Counter()
    track_id_missing = 0
    for item in predictions:
        image_id = int(item["image_id"])
        category_counts[str(int(item.get("category_id", -1)))] += 1
        if "track_id" not in item:
            track_id_missing += 1
        if image_id not in images:
            unknown_ids.append(image_id)
            continue
        if image_id not in selected_ids:
            outside_selected += 1
            continue
        by_image[image_id].append(item)
    return by_image, {
        "total_rows": len(predictions),
        "rows_in_selected_images": sum(len(rows) for rows in by_image.values()),
        "rows_outside_selected_scenes": outside_selected,
        "unknown_image_id_rows": len(unknown_ids),
        "unknown_image_ids_first_20": sorted(set(unknown_ids))[:20],
        "category_id_distribution": dict(sorted(category_counts.items())),
        "missing_track_id_rows": track_id_missing,
    }


def coordinate_audit(
    predictions_by_image: Mapping[int, Sequence[Prediction]],
    images: Mapping[int, ImageInfo],
    variant: str,
    model_width: float,
    model_height: float,
) -> Dict[str, Any]:
    negative_x = negative_y = beyond_right = beyond_bottom = invalid_size = 0
    widths: List[float] = []
    heights: List[float] = []
    center_x: List[float] = []
    center_y: List[float] = []
    area_ratio: List[float] = []
    for image_id, predictions in predictions_by_image.items():
        image = images[image_id]
        image_width = float(image["width"])
        image_height = float(image["height"])
        for prediction in predictions:
            x, y, width, height = transformed_box(prediction, image, variant, model_width, model_height)
            if x < 0.0:
                negative_x += 1
            if y < 0.0:
                negative_y += 1
            if x + width > image_width:
                beyond_right += 1
            if y + height > image_height:
                beyond_bottom += 1
            if width <= 0.0 or height <= 0.0:
                invalid_size += 1
            widths.append(width / image_width)
            heights.append(height / image_height)
            center_x.append((x + width / 2.0) / image_width)
            center_y.append((y + height / 2.0) / image_height)
            area_ratio.append((width * height) / (image_width * image_height))
    return {
        "variant": variant,
        "prediction_count": len(widths),
        "boxes_with_x_lt_0": negative_x,
        "boxes_with_y_lt_0": negative_y,
        "boxes_with_x_plus_w_gt_image_width": beyond_right,
        "boxes_with_y_plus_h_gt_image_height": beyond_bottom,
        "boxes_with_nonpositive_size": invalid_size,
        "normalized_width": describe(widths),
        "normalized_height": describe(heights),
        "normalized_center_x": describe(center_x),
        "normalized_center_y": describe(center_y),
        "area_ratio": describe(area_ratio),
    }


def score_audit(predictions: Sequence[Prediction]) -> Dict[str, Any]:
    scores = [float(item.get("score", 1.0)) for item in predictions]
    return {
        "score": describe(scores),
        "nonfinite_scores": sum(not math.isfinite(value) for value in scores),
        "scores_lt_0": sum(value < 0.0 for value in scores),
        "scores_gt_1": sum(value > 1.0 for value in scores),
    }


def evaluate_variant(
    predictions_by_image: Mapping[int, Sequence[Prediction]],
    images: Mapping[int, ImageInfo],
    ground_truth_by_image: Mapping[int, Sequence[GroundTruth]],
    ground_truth_by_key: Mapping[Tuple[str, int, int], Sequence[GroundTruth]],
    selected_ids: set[int],
    variant: str,
    model_width: float,
    model_height: float,
    frame_offset: int = 0,
    score_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    total = empty_stats()
    groups: Dict[str, DefaultDict[str, Dict[str, Any]]] = {
        kind: new_group_stats() for kind in ("scene", "view", "video", "video_view")
    }
    for image_id in sorted(selected_ids):
        image = images[image_id]
        raw_predictions = list(predictions_by_image.get(image_id, ()))
        if score_threshold is not None:
            raw_predictions = [item for item in raw_predictions if float(item.get("score", 1.0)) >= score_threshold]
        if frame_offset == 0:
            ground_truth = list(ground_truth_by_image.get(image_id, ()))
        else:
            scene, view, frame = image_key(image)
            ground_truth = list(ground_truth_by_key.get((scene, view, frame + frame_offset), ()))
        update_stats(total, raw_predictions, ground_truth, image, variant, model_width, model_height)
        names = group_names(image)
        for kind, name in names.items():
            update_stats(groups[kind][name], raw_predictions, ground_truth, image, variant, model_width, model_height)
    return {
        "variant": variant,
        "frame_offset": frame_offset,
        "score_threshold": score_threshold,
        "global": finalize_stats(total),
        "by_scene": {name: finalize_stats(value) for name, value in sorted(groups["scene"].items())},
        "by_view": {name: finalize_stats(value) for name, value in sorted(groups["view"].items())},
        "by_video": {name: finalize_stats(value) for name, value in sorted(groups["video"].items())},
        "by_video_view": {name: finalize_stats(value) for name, value in sorted(groups["video_view"].items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--scene", action="append", help="scene name(s), comma-separated; omit for the full split")
    parser.add_argument("--model-width", type=float, default=DEFAULT_MODEL_SIZE)
    parser.add_argument("--model-height", type=float, default=DEFAULT_MODEL_SIZE)
    args = parser.parse_args()

    prediction_path = args.predictions.resolve()
    annotation_path = args.annotations.resolve()
    output_path = args.output.resolve()
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite diagnostic output: {output_path}")

    images, ground_truth_by_image, predictions = load_inputs(annotation_path, prediction_path)
    scenes = parse_scene_filter(args.scene)
    selected_ids = selected_image_ids(images, scenes)
    predictions_by_image, prediction_audit = make_predictions_by_image(predictions, images, selected_ids)
    ground_truth_by_key: DefaultDict[Tuple[str, int, int], List[GroundTruth]] = defaultdict(list)
    duplicate_frame_keys = 0
    for image in images.values():
        key = image_key(image)
        if ground_truth_by_key[key]:
            duplicate_frame_keys += 1
        ground_truth_by_key[key].extend(ground_truth_by_image.get(int(image["id"]), ()))

    selected_predictions = [
        prediction
        for image_id, rows in predictions_by_image.items()
        if image_id in selected_ids
        for prediction in rows
    ]
    result: Dict[str, Any] = {
        "status": "PASS",
        "diagnostic": "raw_prediction_vs_annotation_bbox_alignment",
        "split": args.split,
        "dataset": str(args.dataset.resolve()),
        "annotations": str(annotation_path),
        "annotations_sha256": sha256(annotation_path),
        "predictions": str(prediction_path),
        "predictions_sha256": sha256(prediction_path),
        "matching": {
            "box_format": "XYWH",
            "category_policy": "same category_id",
            "max_iou_recall": "per-GT maximum over same-category predictions in the frame",
            "tp_fp_fn": "greedy descending-IoU one-to-one matching at IoU=0.5",
            "ground_truth_rows_retained_without_deduplication": True,
        },
        "model_coordinate_diagnostic": {
            "model_width": args.model_width,
            "model_height": args.model_height,
            "scale_policy": "x/width and y/height scaled independently from model square to annotation image size",
        },
        "image_count": len(images),
        "selected_image_count": len(selected_ids),
        "annotation_ground_truth_count": sum(len(ground_truth_by_image.get(image_id, ())) for image_id in selected_ids),
        "selected_scenes": sorted(scenes) if scenes is not None else None,
        "duplicate_scene_view_frame_keys": duplicate_frame_keys,
        "prediction_audit": prediction_audit,
        "category_id_distribution_selected": dict(sorted(Counter(str(int(item.get("category_id", -1))) for item in selected_predictions).items())),
        "score_audit_selected": score_audit(selected_predictions),
        "coordinate_audit": {
            variant: coordinate_audit(predictions_by_image, images, variant, args.model_width, args.model_height)
            for variant in ("raw", "model_to_image")
        },
        "raw": evaluate_variant(
            predictions_by_image,
            images,
            ground_truth_by_image,
            ground_truth_by_key,
            selected_ids,
            "raw",
            args.model_width,
            args.model_height,
        ),
        "model_to_image": evaluate_variant(
            predictions_by_image,
            images,
            ground_truth_by_image,
            ground_truth_by_key,
            selected_ids,
            "model_to_image",
            args.model_width,
            args.model_height,
        ),
        "frame_offset_audit_raw": {
            str(offset): evaluate_variant(
                predictions_by_image,
                images,
                ground_truth_by_image,
                ground_truth_by_key,
                selected_ids,
                "raw",
                args.model_width,
                args.model_height,
                frame_offset=offset,
            )["global"]
            for offset in FRAME_OFFSETS
        },
        "score_threshold_audit_raw": {
            f"{threshold:g}": evaluate_variant(
                predictions_by_image,
                images,
                ground_truth_by_image,
                ground_truth_by_key,
                selected_ids,
                "raw",
                args.model_width,
                args.model_height,
                score_threshold=threshold,
            )["global"]
            for threshold in SCORE_THRESHOLDS
        },
        "tool": str(Path(__file__).resolve()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "output": str(output_path),
        "selected_images": result["selected_image_count"],
        "selected_predictions": result["prediction_audit"]["rows_in_selected_images"],
        "selected_gt": result["annotation_ground_truth_count"],
        "raw_global": result["raw"]["global"],
        "model_to_image_global": result["model_to_image"]["global"],
        "frame_offset_global": result["frame_offset_audit_raw"],
    }, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # pragma: no cover - CLI error path
        print(f"ERROR: {error}", file=sys.stderr)
        raise
