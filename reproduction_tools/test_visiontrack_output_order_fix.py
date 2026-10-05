#!/usr/bin/env python3
"""Regression tests for the VisionTrack evaluator input/output order fix.

The unit test checks the source-level view-block -> frame-major contract.  The
optional real-artifact check compares the fixed evaluator labeling semantics
against the earlier deterministic diagnostic remap on a small set of videos;
it does not run inference.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import unittest
from typing import Any, Dict, Iterable, List, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/CenterNet2"))
sys.path.insert(0, str(ROOT / "reproduction_tools"))

from diagnose_visiontrack_output_order import (  # noqa: E402
    build_label_to_source_mapping,
)
from gtr.evaluation.mot_evaluation import (  # noqa: E402
    align_visiontrack_inputs_to_outputs,
)


class OutputOrderFixTest(unittest.TestCase):
    def test_two_view_frame_major_mapping(self) -> None:
        def record(image_id: int, frame: int, view: int) -> Dict[str, int]:
            return {
                "image_id": image_id,
                "video_id": 7,
                "frame_id": frame,
                "view_id": view,
                "view_num": 2,
            }

        inputs = [
            record(101, 1, 1),
            record(103, 2, 1),
            record(105, 3, 1),
            record(102, 1, 2),
            record(104, 2, 2),
            record(106, 3, 2),
        ]
        outputs = [object() for _ in inputs]
        aligned = align_visiontrack_inputs_to_outputs("VISION_test", inputs, outputs)
        self.assertEqual(
            [(item["image_id"], item["frame_id"], item["view_id"]) for item in aligned],
            [
                (101, 1, 1),
                (102, 1, 2),
                (103, 2, 1),
                (104, 2, 2),
                (105, 3, 1),
                (106, 3, 2),
            ],
        )

    def test_single_view_and_non_visiontrack_are_identity(self) -> None:
        single_view = [{"image_id": 1, "view_num": 1}, {"image_id": 2, "view_num": 1}]
        outputs = [object(), object()]
        self.assertIs(
            align_visiontrack_inputs_to_outputs("VISION_test", single_view, outputs),
            single_view,
        )
        other = [{"image_id": 1, "view_num": 2}, {"image_id": 2, "view_num": 2}]
        self.assertIs(
            align_visiontrack_inputs_to_outputs("OTHER", other, outputs),
            other,
        )

    def test_structural_errors_are_rejected(self) -> None:
        inputs = [{"image_id": 1, "view_num": 2}, {"image_id": 2, "view_num": 2}]
        with self.assertRaisesRegex(ValueError, "lengths"):
            align_visiontrack_inputs_to_outputs("VISION_test", inputs, [object()])
        with self.assertRaisesRegex(ValueError, "divisible"):
            align_visiontrack_inputs_to_outputs(
                "VISION_test",
                inputs + [{"image_id": 3, "view_num": 2}],
                [object(), object(), object()],
            )


def _read_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_real_artifact_equivalence(
    annotations_path: Path,
    stream_path: Path,
    predictions_path: Path,
    video_ids: Sequence[int],
) -> Dict[str, Any]:
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    images = {int(image["id"]): image for image in annotations["images"]}
    mapping, video_audit = build_label_to_source_mapping(images)
    selected_ids = {
        image_id for image_id, image in images.items() if int(image["video_id"]) in set(video_ids)
    }
    if not selected_ids:
        raise ValueError(f"no annotation images found for video_ids={list(video_ids)}")

    # This models the source-fixed evaluator: each output position receives
    # the image record returned by align_visiontrack_inputs_to_outputs.  The
    # existing stream already contains the same output positions, but with
    # the old labels.
    source_fixed_rows: List[Dict[str, Any]] = []
    stream_records = 0
    selected_stream_records = 0
    for record in _read_jsonl(stream_path):
        stream_records += 1
        old_id = int(record["image_id"])
        source_id = int(mapping.get(old_id, old_id))
        if source_id not in selected_ids:
            continue
        selected_stream_records += 1
        for instance in record.get("instances", []):
            row = copy.deepcopy(instance)
            row["image_id"] = source_id
            # MOTEvaluator._eval_predictions converts contiguous model labels
            # to dataset category IDs while flattening the stream.  The
            # existing stream stores the contiguous label (0); mirror that
            # unrelated, deterministic conversion for the equality check.
            if "category_id" in row:
                row["category_id"] = int(row["category_id"]) + 1
            source_fixed_rows.append(row)

    # This models the previous deterministic diagnostic remap on the flattened
    # COCO prediction file.  It must produce byte-level JSON structure equality
    # (float values included) for the selected rows.
    diagnostic_rows: List[Dict[str, Any]] = []
    for row in json.loads(predictions_path.read_text(encoding="utf-8")):
        remapped_id = int(mapping.get(int(row["image_id"]), int(row["image_id"])))
        if remapped_id in selected_ids:
            corrected = dict(row)
            corrected["image_id"] = remapped_id
            diagnostic_rows.append(corrected)

    if source_fixed_rows != diagnostic_rows:
        for index, (source_row, diagnostic_row) in enumerate(
            zip(source_fixed_rows, diagnostic_rows)
        ):
            if source_row != diagnostic_row:
                raise AssertionError(
                    f"real artifact mismatch at row {index}: "
                    f"source_fixed={source_row!r} diagnostic={diagnostic_row!r}"
                )
        raise AssertionError(
            "real artifact row counts differ: "
            f"source_fixed={len(source_fixed_rows)} diagnostic={len(diagnostic_rows)}"
        )

    return {
        "status": "PASS",
        "video_ids": [int(value) for value in video_ids],
        "selected_image_count": len(selected_ids),
        "stream_records": stream_records,
        "selected_stream_records": selected_stream_records,
        "prediction_rows_compared": len(source_fixed_rows),
        "discrete_fields": "exact",
        "float_fields": "exact in serialized JSON; within 1e-6 by implication",
        "video_audit": {str(value): video_audit.get(str(value)) for value in video_ids},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-subset", action="store_true")
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/test.json"),
    )
    parser.add_argument(
        "--stream",
        type=Path,
        default=Path(
            "/data1/liuyeqiang/WWW/outputs/research_final_v2/off/inference_test/"
            "inference_VISION_test/predictions_stream.jsonl"
        ),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "/data1/liuyeqiang/WWW/outputs/research_final_v2/off/inference_test/"
            "inference_VISION_test/coco_instances_results.json"
        ),
    )
    parser.add_argument("--video-ids", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(OutputOrderFixTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.real_subset:
        report = verify_real_artifact_equivalence(
            args.annotations.resolve(),
            args.stream.resolve(),
            args.predictions.resolve(),
            args.video_ids,
        )
        report.update(
            {
                "annotations": str(args.annotations.resolve()),
                "annotations_sha256": _sha256(args.annotations.resolve()),
                "stream": str(args.stream.resolve()),
                "stream_sha256": _sha256(args.stream.resolve()),
                "predictions": str(args.predictions.resolve()),
                "predictions_sha256": _sha256(args.predictions.resolve()),
            }
        )
        if args.report is not None:
            args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
            args.report.resolve().write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
