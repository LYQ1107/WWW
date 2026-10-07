"""Replay a narrow current-head v2 window around native reactivation mismatches.

The builder warms the mutable OFF state in production order but emits
counterfactual records only for the requested frame window.  This is a
diagnostic sample, never an official dataset or a substitute for video01 v2.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import time

from build_jev_counterfactual_v2 import (
    build_formal_gmt_engine,
    build_v2_records,
    cache_digest,
    ordered_production_keys,
    sha256,
    source_commit,
)
from gtr.modeling.jev_perception_cache import FrozenPerceptionCache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video-id", type=int, default=1)
    parser.add_argument("--frame-start", type=int, default=210)
    parser.add_argument("--frame-end", type=int, default=220)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--view-num", type=int, default=2)
    parser.add_argument("--history-limit", type=int, default=80)
    args = parser.parse_args()
    if args.frame_end < args.frame_start:
        raise ValueError("frame-end must be >= frame-start")

    trace = args.trace.resolve()
    cache_root = args.cache.resolve()
    annotations = args.annotations.resolve()
    checkpoint = args.checkpoint.resolve()
    config_file = args.config_file.resolve()
    source_root = Path(__file__).resolve().parents[1]
    cache = FrozenPerceptionCache(cache_root)
    keys = [key for key in cache.keys() if int(key[0]) == int(args.video_id)]
    ordered, seed_key = ordered_production_keys(keys, lambda key: cache.load(*key))
    selected = [
        index
        for index, key in enumerate(ordered)
        if int(args.frame_start) <= int(key[1]) <= int(args.frame_end)
    ]
    if not selected:
        raise RuntimeError("requested frame window has no cache keys")
    selected_start = min(selected)
    selected_end = max(selected) + 1
    started = time.perf_counter()
    engine = build_formal_gmt_engine(
        config_file=config_file,
        checkpoint=checkpoint,
        device=args.device,
        view_num=args.view_num,
        history_limit=args.history_limit,
    )
    records, stats, skipped = build_v2_records(
        trace=trace,
        cache_root=cache_root,
        annotations=annotations,
        checkpoint_hash=sha256(checkpoint),
        horizon=args.horizon,
        association_backend="formal_gmt_transformer",
        engine=engine,
        cache_obj=cache,
        cache_keys_by_video={int(args.video_id): ordered},
        key_start_index=1,
        key_end_index=selected_end,
        selected_key_range=(selected_start, selected_end),
        video_ids=[int(args.video_id)],
    )
    wall_seconds = time.perf_counter() - started
    args.records_output.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.records_output.resolve().open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    reactivation = [
        record
        for record in records
        if record.get("question_type") == "REACTIVATION_DECISION"
    ]
    report = {
        "status": "DIAGNOSTIC_ONLY",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "not_for_official_records": True,
        "not_for_selection": True,
        "video_id": int(args.video_id),
        "frame_window": [int(args.frame_start), int(args.frame_end)],
        "production_key_range": [int(selected_start), int(selected_end)],
        "seed_key": list(seed_key),
        "records": len(records),
        "records_by_question": stats,
        "reactivation_records": len(reactivation),
        "skipped_events": int(skipped),
        "wall_seconds": float(wall_seconds),
        "source_root": str(source_root),
        "source_commit": source_commit(source_root),
        "trace": str(trace),
        "trace_sha256": sha256(trace),
        "cache": str(cache_root),
        "cache_index_sha256": cache_digest(cache_root),
        "annotations": str(annotations),
        "annotations_sha256": sha256(annotations),
        "checkpoint_sha256": sha256(checkpoint),
        "config_file": str(config_file),
        "config_sha256": sha256(config_file),
        "association_backend": "formal_gmt_transformer",
        "records_output": str(args.records_output.resolve()),
        "reactivation_keys": [
            list(record["state"]["online_context"].values())
            for record in reactivation
        ],
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
