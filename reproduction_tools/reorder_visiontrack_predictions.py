#!/usr/bin/env python3
"""Write a deterministic diagnostic copy of VisionTrack COCO predictions.

This applies the source-derived view-block -> frame-major label correction
audited by ``diagnose_visiontrack_output_order.py``.  It never overwrites the
input or any official inference output, and the resulting JSON is diagnostic
until the source evaluator fix is independently reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

from diagnose_raw_detection_alignment import load_inputs, sha256
from diagnose_visiontrack_output_order import build_label_to_source_mapping, remap_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset", type=Path, default=Path("/data/DATASETS/TRACKING/JDE/VisionTrack"))
    args = parser.parse_args()

    prediction_path = args.predictions.resolve()
    annotation_path = args.annotations.resolve()
    output_path = args.output.resolve()
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite diagnostic prediction copy: {output_path}")

    images, _ground_truth, predictions = load_inputs(annotation_path, prediction_path)
    mapping, video_audit = build_label_to_source_mapping(images)
    remapped = remap_predictions(predictions, mapping)
    changed_rows = sum(
        int(old["image_id"]) != int(new["image_id"])
        for old, new in zip(predictions, remapped)
    )
    result: Dict[str, Any] = {
        "status": "PASS",
        "diagnostic_only": True,
        "split": args.split,
        "dataset": str(args.dataset.resolve()),
        "annotations": str(annotation_path),
        "annotations_sha256": sha256(annotation_path),
        "input_predictions": str(prediction_path),
        "input_predictions_sha256": sha256(prediction_path),
        "mapping_changed_image_rows": changed_rows,
        "mapping_image_ids_changed": sum(label != source for label, source in mapping.items()),
        "prediction_rows": len(predictions),
        "video_audit": video_audit,
        "mapping_source": {
            "mapper": "gtr/data/gtr_dataset_mapper.py",
            "model": "gtr/modeling/meta_arch/gtr_rcnn.py::sliding_inference_GMT",
            "evaluator": "gtr/evaluation/mot_evaluation.py::MOTEvaluator.process",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(remapped, separators=(",", ":")), encoding="utf-8")
    result["output_predictions"] = str(output_path)
    result["output_predictions_sha256"] = sha256(output_path)
    result_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
