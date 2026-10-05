"""Bounded exact old-vs-fast formal replay equivalence check."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import random
import tempfile
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bounded_legacy_trace(partition: Path, order_index: Path, output: Path, limit: int) -> int:
    selected = []
    with partition.open("rb") as raw_handle, order_index.open(encoding="utf-8") as order_handle:
        for raw, order_line in zip(raw_handle, order_handle):
            if not raw.strip():
                continue
            payload = json.loads(order_line)
            source_line = int(payload["source_line"] if isinstance(payload, dict) else payload)
            selected.append((source_line, raw))
            if len(selected) >= int(limit):
                break
    if len(selected) < int(limit):
        raise ValueError(f"partition has only {len(selected)} lines; requested {limit}")
    with output.open("wb") as handle:
        previous = -1
        for source_line, raw in selected:
            gap = int(source_line) - previous - 1
            if gap < 0:
                raise ValueError("partition order sidecar is not strictly increasing")
            handle.write(b"\n" * gap)
            handle.write(raw)
            previous = int(source_line)
    return len(selected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--video-id", type=int, required=True)
    parser.add_argument("--events", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--view-num", type=int, default=2)
    parser.add_argument("--history-limit", type=int, default=80)
    args = parser.parse_args()
    if args.events < 1:
        raise ValueError("--events must be positive")

    import sys

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "reproduction_tools"))
    from build_jev_counterfactual_dataset import load_gt
    from build_jev_counterfactual_v2 import build_formal_gmt_engine, build_v2_records
    from gtr.modeling.jev_perception_cache import FrozenPerceptionCache

    partition_root = args.partition_root.resolve()
    partition_manifest = json.loads(
        (partition_root / "partition_manifest.json").read_text(encoding="utf-8")
    )
    video = str(int(args.video_id))
    partition = Path(partition_manifest["trace_by_video"]) / f"video_{int(args.video_id):02d}.jsonl"
    order_index = partition.with_name(partition.stem + ".orders.jsonl")
    cache_keys = json.loads((partition_root / "cache_keys_by_video.json").read_text(encoding="utf-8"))
    cache_keys_for_video = cache_keys[video]
    gt_bundle = load_gt(args.annotations.resolve())
    checkpoint_hash = sha256(args.checkpoint.resolve())
    engine = build_formal_gmt_engine(
        config_file=args.config_file.resolve(),
        checkpoint=args.checkpoint.resolve(),
        device=args.device,
        view_num=args.view_num,
        history_limit=args.history_limit,
    )

    with tempfile.TemporaryDirectory(prefix="jev_equivalence_", dir="/home/liuyeqiang") as name:
        legacy = Path(name) / "legacy_aligned.jsonl"
        selected = bounded_legacy_trace(partition, order_index, legacy, args.events)
        # The released GMT transformer assigns trajectory embedding slots with
        # the process-global Python RNG.  Preserve that existing behavior while
        # comparing implementations by replaying the identical RNG state for
        # both bounded builds; this test is about the speed path, not a new
        # randomization policy.
        random_state = random.getstate()
        old_records, old_stats, old_skipped = build_v2_records(
            trace=legacy,
            cache_root=args.cache.resolve(),
            annotations=args.annotations.resolve(),
            checkpoint_hash=checkpoint_hash,
            horizon=8,
            association_backend="formal_gmt_transformer",
            engine=engine,
            video_ids=[int(args.video_id)],
            max_events_per_video=args.events,
        )
        random.setstate(random_state)
        fast_cache = FrozenPerceptionCache(args.cache.resolve())
        fast_records, fast_stats, fast_skipped = build_v2_records(
            trace=partition,
            cache_root=args.cache.resolve(),
            annotations=args.annotations.resolve(),
            checkpoint_hash=checkpoint_hash,
            horizon=8,
            association_backend="formal_gmt_transformer",
            engine=engine,
            cache_obj=fast_cache,
            gt_bundle=gt_bundle,
            cache_keys_by_video={int(args.video_id): cache_keys_for_video},
            order_index=order_index,
            max_events_per_video=args.events,
        )

    exact = old_records == fast_records and old_stats == fast_stats and old_skipped == fast_skipped
    mismatch = None
    if not exact:
        for index, (old, fast) in enumerate(zip(old_records, fast_records)):
            if old != fast:
                mismatch = {
                    "index": index,
                    "old_state_digest": old.get("state_digest"),
                    "fast_state_digest": fast.get("state_digest"),
                    "old": old,
                    "fast": fast,
                }
                break
        if mismatch is None and len(old_records) != len(fast_records):
            mismatch = {"old_records": len(old_records), "fast_records": len(fast_records)}
    report = {
        "status": "PASS" if exact else "FAIL",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "video_id": int(args.video_id),
        "bounded_events": int(selected),
        "horizon": 8,
        "association_backend": "formal_gmt_transformer",
        "old_records": len(old_records),
        "fast_records": len(fast_records),
        "old_stats": old_stats,
        "fast_stats": fast_stats,
        "old_skipped": old_skipped,
        "fast_skipped": fast_skipped,
        "exact_record_equality": exact,
        "trajectory_slot_randomization": "existing process-global Python RNG; identical state replayed for old and optimized builds",
        "source_trace_sha256": partition_manifest["trace_sha256"],
        "cache_index_sha256": partition_manifest["cache_index_sha256"],
        "gmt_checkpoint_sha256": "sha256:" + checkpoint_hash,
        "mismatch": mismatch,
        "semantic_scope": [
            "raw partition line selection",
            "global source event_order restoration",
            "cache object reuse",
            "pre-indexed cache keys",
            "formal GMT association replay",
        ],
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not exact:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
