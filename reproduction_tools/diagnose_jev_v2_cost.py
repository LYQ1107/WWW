"""Measure formal v2 counterfactual replay cost on a bounded sample.

This is a diagnostic-only runner.  It must never be used as an official
artifact or as a replacement for the full video1 builder.  The wrappers time
the actual formal proposal/step/branch operations and count calls while
reusing the production builder on an explicitly bounded event prefix.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import time
from typing import Any, Callable

import torch

import build_jev_counterfactual_v2 as builder
from build_jev_counterfactual_v2 import (
    build_formal_gmt_engine,
    build_v2_records,
    cache_digest,
    sha256,
    source_commit,
)
from jev_counterfactual_v2 import CachedPerceptionMutableAssociationV2, MutableGMTState


def _stat() -> dict[str, Any]:
    return {"calls": 0, "seconds": 0.0, "max_seconds": 0.0}


def _record(stat: dict[str, Any], elapsed: float) -> None:
    stat["calls"] += 1
    stat["seconds"] += float(elapsed)
    stat["max_seconds"] = max(float(stat["max_seconds"]), float(elapsed))


def _sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--records-output",
        type=Path,
        help="optional diagnostic JSONL; never an official training artifact",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--video-id", type=int, default=1)
    parser.add_argument("--max-events", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--view-num", type=int, default=2)
    parser.add_argument("--history-limit", type=int, default=80)
    args = parser.parse_args()
    if args.max_events < 1:
        raise ValueError("--max-events must be positive")

    timings = {
        "proposal": _stat(),
        "state_clone": _stat(),
        "reactivation_context": _stat(),
        "step": _stat(),
        "utility_score_rollout": _stat(),
    }

    original_propose = CachedPerceptionMutableAssociationV2.propose
    original_step = CachedPerceptionMutableAssociationV2.step
    original_clone = MutableGMTState.clone
    original_reactivation = builder._prepare_reactivation_context
    original_rollout = builder.score_rollout

    def timed_propose(self, *call_args, **call_kwargs):
        _sync(args.device)
        start = time.perf_counter()
        result = original_propose(self, *call_args, **call_kwargs)
        _sync(args.device)
        _record(timings["proposal"], time.perf_counter() - start)
        return result

    def timed_step(self, *call_args, **call_kwargs):
        _sync(args.device)
        start = time.perf_counter()
        result = original_step(self, *call_args, **call_kwargs)
        _sync(args.device)
        _record(timings["step"], time.perf_counter() - start)
        return result

    def timed_clone(self, *call_args, **call_kwargs):
        start = time.perf_counter()
        result = original_clone(self, *call_args, **call_kwargs)
        _record(timings["state_clone"], time.perf_counter() - start)
        return result

    def timed_reactivation(*call_args, **call_kwargs):
        start = time.perf_counter()
        result = original_reactivation(*call_args, **call_kwargs)
        _record(timings["reactivation_context"], time.perf_counter() - start)
        return result

    def timed_rollout(*call_args, **call_kwargs):
        start = time.perf_counter()
        result = original_rollout(*call_args, **call_kwargs)
        _record(timings["utility_score_rollout"], time.perf_counter() - start)
        return result

    CachedPerceptionMutableAssociationV2.propose = timed_propose
    CachedPerceptionMutableAssociationV2.step = timed_step
    MutableGMTState.clone = timed_clone
    builder._prepare_reactivation_context = timed_reactivation
    builder.score_rollout = timed_rollout
    try:
        trace = args.trace.resolve()
        cache = args.cache.resolve()
        annotations = args.annotations.resolve()
        checkpoint = args.checkpoint.resolve()
        config_file = args.config_file.resolve()
        wall_start = time.perf_counter()
        engine = build_formal_gmt_engine(
            config_file=config_file,
            checkpoint=checkpoint,
            device=args.device,
            view_num=args.view_num,
            history_limit=args.history_limit,
        )
        records, stats, skipped = build_v2_records(
            trace=trace,
            cache_root=cache,
            annotations=annotations,
            checkpoint_hash=sha256(checkpoint),
            horizon=args.horizon,
            association_backend="formal_gmt_transformer",
            engine=engine,
            max_events=args.max_events,
            video_ids=[args.video_id],
        )
        wall_seconds = time.perf_counter() - wall_start
    finally:
        CachedPerceptionMutableAssociationV2.propose = original_propose
        CachedPerceptionMutableAssociationV2.step = original_step
        MutableGMTState.clone = original_clone
        builder._prepare_reactivation_context = original_reactivation
        builder.score_rollout = original_rollout

    for value in timings.values():
        calls = int(value["calls"])
        value["mean_seconds"] = (
            float(value["seconds"]) / calls if calls else 0.0
        )

    report = {
        "status": "DIAGNOSTIC_ONLY",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "not_for_selection": True,
        "not_for_official_records": True,
        "video_id": int(args.video_id),
        "max_events": int(args.max_events),
        "horizon": int(args.horizon),
        "records_emitted": len(records),
        "records_by_question": stats,
        "skipped_events": int(skipped),
        "wall_seconds": float(wall_seconds),
        "timings": timings,
        "source_root": str(Path(__file__).resolve().parents[1]),
        "source_commit": source_commit(Path(__file__).resolve().parents[1]),
        "trace": str(trace),
        "trace_sha256": sha256(trace),
        "cache": str(cache),
        "cache_index_sha256": cache_digest(cache),
        "annotations": str(annotations),
        "checkpoint_sha256": sha256(checkpoint),
        "config_file": str(config_file),
        "config_sha256": sha256(config_file),
        "engine": "cached_perception_mutable_association_v2",
        "association_backend": "formal_gmt_transformer",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.records_output is not None:
        args.records_output.parent.mkdir(parents=True, exist_ok=True)
        with args.records_output.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
