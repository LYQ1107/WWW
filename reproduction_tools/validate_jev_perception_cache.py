"""Validate a frozen JEV perception cache against VisionTrack annotations."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Mapping, Set, Tuple

from gtr.modeling.jev_perception_cache import CACHE_VERSION, FrozenPerceptionCache


def expected_keys(annotation_path: Path) -> Set[Tuple[int, int, int]]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    return {
        (int(image["video_id"]), int(image["frame_id"]) - 1, int(image["view_id"]) - 1)
        for image in payload["images"]
    }


def validate(cache_root: Path, annotation_path: Path, *, require_complete: bool) -> Mapping[str, object]:
    expected = expected_keys(annotation_path)
    cache = FrozenPerceptionCache(cache_root)
    seen: Set[Tuple[int, int, int]] = set()
    duplicate_keys = []
    unexpected_keys = []
    payload_errors = []
    by_video = Counter()
    detection_count = 0

    for key in cache.keys():
        key = tuple(int(value) for value in key)
        if key in seen:
            duplicate_keys.append(key)
        seen.add(key)
        by_video[key[0]] += 1
        if key not in expected:
            unexpected_keys.append(key)
        try:
            payload = cache.load(*key)
            if str(payload.get("cache_version")) != CACHE_VERSION:
                raise ValueError("cache version mismatch")
            boxes = payload["pred_boxes"]
            # The frozen payload stores the canonical score tensor under
            # ``detection_scores``.  ``proposal_metadata['scores']`` is only
            # an optional audit copy and is not guaranteed to be present.
            scores = payload["detection_scores"]
            features = payload["reid_features"]
            if int(boxes.shape[0]) != int(scores.shape[0]) or int(boxes.shape[0]) != int(features.shape[0]):
                raise ValueError("detection tensor length mismatch")
            detection_count += int(boxes.shape[0])
        except Exception as exc:  # noqa: BLE001 - report every bad record
            payload_errors.append({"key": list(key), "error": repr(exc)})

    missing = sorted(expected - seen)
    complete = not duplicate_keys and not unexpected_keys and not payload_errors and not missing
    status = "PASS" if complete or (not require_complete and not payload_errors) else "FAIL"
    if not complete and not require_complete and status == "PASS":
        status = "INCOMPLETE"
    return {
        "status": status,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cache": str(cache_root),
        "cache_version": CACHE_VERSION,
        "annotations": str(annotation_path),
        "expected_records": len(expected),
        "observed_records": len(seen),
        "missing_records": len(missing),
        "duplicate_records": len(duplicate_keys),
        "unexpected_records": len(unexpected_keys),
        "payload_errors": len(payload_errors),
        "total_detections": detection_count,
        "videos_observed": dict(sorted((str(k), v) for k, v in by_video.items())),
        "missing_examples": [list(key) for key in missing[:20]],
        "unexpected_examples": [list(key) for key in unexpected_keys[:20]],
        "payload_error_examples": payload_errors[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    report = validate(
        args.cache.resolve(),
        args.annotations.resolve(),
        require_complete=args.require_complete,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] == "FAIL":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
