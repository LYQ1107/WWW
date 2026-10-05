#!/usr/bin/env python3
"""Audit the COCO XYWH -> VisionTrack MOT text round-trip.

This reads an already prepared diagnostic evaluation directory and checks the
actual written MOT rows against the source COCO rows.  It does not change the
prepared input or any official result.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Dict, List, Mapping, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261005)
    args = parser.parse_args()

    predictions_path = args.predictions.resolve()
    annotations_path = args.annotations.resolve()
    prepared = args.prepared.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite round-trip audit: {output_path}")

    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    images = {int(image["id"]): image for image in annotations["images"]}
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    by_sequence: Dict[str, List[Tuple[int, int, float, float, float, float, float, int]]] = defaultdict(list)
    unknown_image_ids = 0
    for index, item in enumerate(predictions):
        image_id = int(item["image_id"])
        if image_id not in images:
            unknown_image_ids += 1
            continue
        image = images[image_id]
        sequence = str(image["file_name"]).split("/", 1)[0]
        x, y, width, height = (float(value) for value in item["bbox"])
        by_sequence[sequence].append(
            (
                int(image["frame_id"]),
                int(item["track_id"]),
                x,
                y,
                width,
                height,
                float(item.get("score", 1.0)),
                index,
            )
        )

    mismatch_count = 0
    row_count = 0
    max_bbox_abs_error = 0.0
    max_score_abs_error = 0.0
    missing_prepared_sequences: List[str] = []
    expected_rows: List[Dict[str, Any]] = []
    actual_rows: List[Dict[str, Any]] = []
    for sequence, rows in sorted(by_sequence.items()):
        rows = sorted(rows, key=lambda row: (row[0], row[1]))
        actual_path = prepared / "trackeval" / "trackers" / "GMT" / "data" / f"{sequence}.txt"
        if not actual_path.is_file():
            missing_prepared_sequences.append(sequence)
            continue
        parsed = []
        for line in actual_path.read_text(encoding="utf-8").splitlines():
            fields = line.split(",")
            if len(fields) < 7:
                raise ValueError(f"malformed MOT row in {actual_path}: {line!r}")
            parsed.append(
                {
                    "frame": int(float(fields[0])),
                    "track_id": int(float(fields[1])),
                    "bbox": [float(fields[2]), float(fields[3]), float(fields[4]), float(fields[5])],
                    "score": float(fields[6]),
                }
            )
        if len(parsed) != len(rows):
            mismatch_count += abs(len(parsed) - len(rows))
        for position, source in enumerate(rows):
            row_count += 1
            if position >= len(parsed):
                mismatch_count += 1
                continue
            actual = parsed[position]
            expected = {
                "frame": source[0],
                "track_id": source[1],
                "bbox": list(source[2:6]),
                "score": source[6],
                "source_index": source[7],
                "sequence": sequence,
            }
            actual["sequence"] = sequence
            expected_rows.append(expected)
            actual_rows.append(actual)
            bbox_error = max(abs(float(a) - float(b)) for a, b in zip(expected["bbox"], actual["bbox"]))
            score_error = abs(float(expected["score"]) - float(actual["score"]))
            max_bbox_abs_error = max(max_bbox_abs_error, bbox_error)
            max_score_abs_error = max(max_score_abs_error, score_error)
            if expected["frame"] != actual["frame"] or expected["track_id"] != actual["track_id"] or bbox_error > 1e-5 or score_error > 1e-5:
                mismatch_count += 1

    sample_size = min(args.sample_size, len(expected_rows))
    rng = random.Random(args.seed)
    sample_indices = rng.sample(range(len(expected_rows)), sample_size) if sample_size else []
    sampled_roundtrip = [
        {
            "expected": expected_rows[index],
            "actual": actual_rows[index],
            "max_bbox_abs_error": max(
                abs(a - b)
                for a, b in zip(expected_rows[index]["bbox"], actual_rows[index]["bbox"])
            ),
            "score_abs_error": abs(expected_rows[index]["score"] - actual_rows[index]["score"]),
        }
        for index in sample_indices
    ]
    result = {
        "status": "PASS" if not unknown_image_ids and not missing_prepared_sequences and mismatch_count == 0 and max_bbox_abs_error < 1e-5 else "FAIL",
        "predictions": str(predictions_path),
        "predictions_sha256": sha256(predictions_path),
        "annotations": str(annotations_path),
        "annotations_sha256": sha256(annotations_path),
        "prepared": str(prepared),
        "prediction_rows": len(predictions),
        "rows_compared": row_count,
        "unknown_image_id_rows": unknown_image_ids,
        "missing_prepared_sequences": missing_prepared_sequences,
        "mismatch_count": mismatch_count,
        "max_bbox_abs_error": max_bbox_abs_error,
        "max_score_abs_error": max_score_abs_error,
        "sample": {
            "seed": args.seed,
            "requested": args.sample_size,
            "used": sample_size,
            "rows": sampled_roundtrip,
        },
        "box_policy": "COCO XYWH -> MOT XYWH, six decimal output; no XYXY conversion in this audit",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in (
        "status", "prediction_rows", "rows_compared", "unknown_image_id_rows",
        "missing_prepared_sequences", "mismatch_count", "max_bbox_abs_error",
        "max_score_abs_error",
    )}, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
