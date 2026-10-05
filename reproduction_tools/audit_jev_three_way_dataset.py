"""Lightweight provenance audit for the shared H=8 policy dataset.

The audit checks a deterministic sample against the canonical TRAIN trace and
VisionTrack annotation index.  It never opens official TEST annotations and
does not materialize another copy of the trace or dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from jev_dataset_contract import validate_record


CHECKPOINT_SHA256 = "cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8"
VIDEO_ID_RE = re.compile(rb'"video_id"\s*:\s*(-?\d+)')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def annotation_index(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    videos = {int(item["id"]): item for item in payload["videos"]}
    images = {
        (int(item["video_id"]), int(item["view_id"]), int(item["frame_id"])): item
        for item in payload["images"]
    }
    return videos, images


def selected_records(path: Path, sample_size: int):
    total = 0
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            total += 1
            record = json.loads(line)
            # First records plus a deterministic stride make the audit stable
            # without requiring a second copy or random-access index.
            if len(records) < sample_size or total % max(1, total // max(1, sample_size)) == 0:
                if len(records) < sample_size:
                    records.append((line_number, record))
    if total == 0:
        raise ValueError("shared policy dataset is empty")
    # If the first-record condition filled the sample, retain that exact
    # deterministic prefix.  This is intentionally conservative and cheap.
    return total, records[:sample_size]


def trace_matches(trace: Path, samples: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    wanted = {}
    for record in samples:
        context = record["state"]["online_context"]
        key = (int(context["video_id"]), int(context["event_order"]), str(record["question_type"]))
        wanted[key] = record
    found = {}
    wanted_videos = {key[0] for key in wanted}
    with trace.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            match = VIDEO_ID_RE.search(line.encode("utf-8"))
            if match is not None and int(match.group(1)) not in wanted_videos:
                continue
            event = json.loads(line)
            context = event.get("context", {})
            key = (
                int(context.get("video_id", -1)),
                int(context.get("event_order", -1)),
                str(event.get("question")),
            )
            if key not in wanted or key in found:
                continue
            record = wanted[key]
            record_context = record["state"]["online_context"]
            features = event.get("state_feature_vector")
            expected = record["state"]["feature_vector"]
            if not isinstance(features, list) or len(features) != len(expected):
                feature_error = float("inf")
            else:
                feature_error = max(
                    (abs(float(left) - float(right)) for left, right in zip(features, expected)),
                    default=float("inf"),
                )
            found[key] = {
                "frame_match": int(context.get("frame", -1)) == int(record_context["frame"]),
                "view_match": int(context.get("view", -1)) == int(record_context["view"]),
                "question_match": str(event.get("question")) == str(record["question_type"]),
                "legal_actions_match": list(event.get("legal_actions", ())) == list(record["legal_actions"]),
                "feature_max_abs_error": feature_error,
            }
            if len(found) == len(wanted):
                break
    return {
        "requested": len(wanted),
        "found": len(found),
        "all_found": len(found) == len(wanted),
        "all_fields_match": bool(found) and all(
            item["frame_match"]
            and item["view_match"]
            and item["question_match"]
            and item["legal_actions_match"]
            and item["feature_max_abs_error"] <= 1e-7
            for item in found.values()
        ),
        "max_feature_abs_error": max(
            (item["feature_max_abs_error"] for item in found.values()),
            default=None,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    args = parser.parse_args()
    if args.sample_size < 1:
        raise ValueError("sample-size must be positive")
    if args.annotations.name != "train.json":
        raise ValueError("shared policy audit accepts TRAIN annotations only")

    videos, images = annotation_index(args.annotations)
    total, samples = selected_records(args.dataset, args.sample_size)
    dimensions = set()
    sequences = set()
    seen_keys = set()
    invalid = []
    correspondence = []
    for line_number, record in samples:
        try:
            validate_record(record, allow_future_gt=True)
            if int(record["horizon"]) != 8:
                raise ValueError("record horizon is not H=8")
            if str(record["gmt_checkpoint_sha256"]) not in {
                CHECKPOINT_SHA256,
                "sha256:" + CHECKPOINT_SHA256,
            }:
                raise ValueError("record is not bound to canonical model_20000")
            state = record["state"]
            features = state["feature_vector"]
            dimensions.add(len(features))
            sequences.add(str(record["sequence"]))
            context = state["online_context"]
            key = (int(context["video_id"]), int(context["event_order"]), str(record["question_type"]))
            if key in seen_keys:
                raise ValueError("duplicate sampled event identity")
            seen_keys.add(key)
            video_id = int(context["video_id"])
            frame = int(context["frame"])
            view = int(context["view"])
            image = images.get((video_id, view + 1, frame)) or images.get((video_id, view, frame))
            if image is None:
                raise ValueError("no annotation image for video/frame/view")
            if str(record["sequence"]) != str(videos[video_id]["file_name"]):
                raise ValueError("sequence does not match annotation video")
            if any(not math.isfinite(float(value)) for value in features):
                raise ValueError("non-finite state feature")
            correspondence.append({"video_id": video_id, "frame": frame, "view": view, "image_id": int(image["id"])})
        except Exception as exc:  # noqa: BLE001 - report all sampled failures
            invalid.append({"line": line_number, "error": str(exc)})

    trace_report = trace_matches(args.trace, [record for _, record in samples])
    report = {
        "status": "PASS" if total and len(samples) == min(total, args.sample_size) and not invalid and trace_report["all_found"] and trace_report["all_fields_match"] else "FAIL",
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256(args.dataset),
        "trace": str(args.trace.resolve()),
        "trace_sha256": sha256(args.trace),
        "annotations": str(args.annotations.resolve()),
        "annotations_sha256": sha256(args.annotations),
        "horizon": 8,
        "records": total,
        "sample_size": len(samples),
        "feature_dimensions": sorted(dimensions),
        "sampled_sequences": len(sequences),
        "invalid_sample_records": invalid,
        "trace_correspondence": trace_report,
        "annotation_correspondence_records": len(correspondence),
        "official_test_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
