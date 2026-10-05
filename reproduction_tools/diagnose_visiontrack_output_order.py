#!/usr/bin/env python3
"""Audit the VisionTrack multi-view output/image-id ordering contract.

The GMT model receives a video as view-block order but
``sliding_inference_GMT`` returns instances in frame-major order.  This tool
does not alter official predictions.  It only applies the deterministic
label-to-source-image mapping implied by those two source paths and compares
raw bbox alignment before and after that mapping.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from diagnose_raw_detection_alignment import (
    DEFAULT_DATASET,
    DEFAULT_MODEL_SIZE,
    evaluate_variant,
    load_inputs,
    make_predictions_by_image,
    parse_scene_filter,
    selected_image_ids,
    sha256,
)


ROOT = Path(__file__).resolve().parents[1]


def build_label_to_source_mapping(images: Mapping[int, Mapping[str, Any]]) -> Tuple[Dict[int, int], Dict[str, Any]]:
    by_video: Dict[int, List[Mapping[str, Any]]] = defaultdict(list)
    for image in images.values():
        by_video[int(image["video_id"])].append(image)

    mapping: Dict[int, int] = {}
    video_audit: Dict[str, Any] = {}
    for video_id, video_images in sorted(by_video.items()):
        ordered_input = sorted(video_images, key=lambda image: int(image["id"]))
        views = sorted({int(image["view_id"]) for image in video_images})
        by_view = {
            view: sorted(
                [image for image in video_images if int(image["view_id"]) == view],
                key=lambda image: int(image["frame_id"]),
            )
            for view in views
        }
        lengths = {str(view): len(rows) for view, rows in by_view.items()}
        if len(views) <= 1 or len(set(lengths.values())) != 1:
            video_audit[str(video_id)] = {
                "status": "IDENTITY_OR_UNSUPPORTED",
                "views": views,
                "frames_by_view": lengths,
            }
            for image in ordered_input:
                mapping[int(image["id"])] = int(image["id"])
            continue

        frames_per_view = next(iter(lengths.values()))
        expected_count = len(views) * frames_per_view
        if len(ordered_input) != expected_count:
            raise ValueError(
                f"video {video_id} has inconsistent image count: "
                f"{len(ordered_input)} != {expected_count}"
            )
        # Input labels are view-block ordered.  Model outputs are frame-major:
        # output k=(frame*view_count+view) came from input index
        # view*frames_per_view+frame.  The evaluator currently labels output k
        # with input[k], so this is the deterministic correction.
        for output_position, label_image in enumerate(ordered_input):
            frame_index = output_position // len(views)
            view_index = output_position % len(views)
            source_image = by_view[views[view_index]][frame_index]
            mapping[int(label_image["id"])] = int(source_image["id"])
        video_audit[str(video_id)] = {
            "status": "FRAME_MAJOR_OUTPUT_VS_VIEW_BLOCK_INPUT",
            "views": views,
            "frames_per_view": frames_per_view,
            "input_order": "view-block",
            "model_output_order": "frame-major",
            "mapping_changed": sum(
                mapping[int(image["id"])] != int(image["id"])
                for image in ordered_input
            ),
        }
    return mapping, video_audit


def remap_predictions(predictions: Sequence[Mapping[str, Any]], mapping: Mapping[int, int]) -> List[Dict[str, Any]]:
    remapped = []
    for item in predictions:
        row = dict(item)
        image_id = int(row["image_id"])
        row["image_id"] = int(mapping.get(image_id, image_id))
        remapped.append(row)
    return remapped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--scene", action="append")
    parser.add_argument("--model-width", type=float, default=DEFAULT_MODEL_SIZE)
    parser.add_argument("--model-height", type=float, default=DEFAULT_MODEL_SIZE)
    args = parser.parse_args()

    prediction_path = args.predictions.resolve()
    annotation_path = args.annotations.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite diagnostic output: {output_path}")
    images, ground_truth_by_image, predictions = load_inputs(annotation_path, prediction_path)
    scenes = parse_scene_filter(args.scene)
    selected_ids = selected_image_ids(images, scenes)
    current_by_image, current_audit = make_predictions_by_image(predictions, images, selected_ids)

    mapping, video_audit = build_label_to_source_mapping(images)
    remapped = remap_predictions(predictions, mapping)
    remapped_by_image, remapped_audit = make_predictions_by_image(remapped, images, selected_ids)

    # This helper needs the same frame-key index as the raw audit, but the
    # ordering comparison itself is evaluated at offset zero only.
    ground_truth_by_key: Dict[Tuple[str, int, int], List[Mapping[str, Any]]] = defaultdict(list)
    for image in images.values():
        scene = str(image["file_name"]).split("/", 1)[0].rsplit("_", 1)[0]
        key = (scene, int(image["view_id"]), int(image["frame_id"]))
        ground_truth_by_key[key].extend(ground_truth_by_image.get(int(image["id"]), ()))

    before = evaluate_variant(
        current_by_image,
        images,
        ground_truth_by_image,
        ground_truth_by_key,
        selected_ids,
        "raw",
        args.model_width,
        args.model_height,
    )
    after = evaluate_variant(
        remapped_by_image,
        images,
        ground_truth_by_image,
        ground_truth_by_key,
        selected_ids,
        "raw",
        args.model_width,
        args.model_height,
    )

    samples = []
    for label_id, source_id in sorted(mapping.items())[:1000]:
        label = images[label_id]
        source = images[source_id]
        samples.append({
            "label_image_id": label_id,
            "label_file_name": label["file_name"],
            "label_frame_id": int(label["frame_id"]),
            "label_view_id": int(label["view_id"]),
            "source_image_id": source_id,
            "source_file_name": source["file_name"],
            "source_frame_id": int(source["frame_id"]),
            "source_view_id": int(source["view_id"]),
        })

    result = {
        "status": "PASS",
        "diagnostic": "visiontrack_evaluator_output_order",
        "split": args.split,
        "dataset": str(args.dataset.resolve()),
        "annotations": str(annotation_path),
        "annotations_sha256": sha256(annotation_path),
        "predictions": str(prediction_path),
        "predictions_sha256": sha256(prediction_path),
        "selected_scenes": sorted(scenes) if scenes is not None else None,
        "selected_image_count": len(selected_ids),
        "mapping_definition": {
            "input_order": "view-block, as produced by GMTDatasetMapper for inference",
            "output_order": "frame-major, as produced by GTRRCNN.sliding_inference_GMT final batch reorder",
            "evaluator_behavior": "MOTEvaluator.process zips original inputs with reordered outputs",
            "correction": "current evaluator label image k -> source input image (view=k%V, frame=k//V)",
        },
        "video_audit": video_audit,
        "current_prediction_audit": current_audit,
        "remapped_prediction_audit": remapped_audit,
        "before_raw_alignment": before["global"],
        "after_deterministic_order_remap_raw_alignment": after["global"],
        "mapping_changed_rows": sum(label != source for label, source in mapping.items()),
        "mapping_sample_first_1000": samples,
        "source_evidence": {
            "mapper": "gtr/data/gtr_dataset_mapper.py",
            "model": "gtr/modeling/meta_arch/gtr_rcnn.py::sliding_inference_GMT",
            "evaluator": "gtr/evaluation/mot_evaluation.py::MOTEvaluator.process",
        },
        "tool": str(Path(__file__).resolve()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "output": str(output_path),
        "mapping_changed_rows": result["mapping_changed_rows"],
        "before": result["before_raw_alignment"],
        "after": result["after_deterministic_order_remap_raw_alignment"],
    }, indent=2))


if __name__ == "__main__":
    main()
