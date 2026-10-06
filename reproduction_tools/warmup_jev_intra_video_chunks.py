"""Reconstruct exact OFF states at deterministic chunk starts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--chunk-plan", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--view-num", type=int, default=2)
    parser.add_argument("--history-limit", type=int, default=80)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "reproduction_tools"))
    sys.path.insert(0, str(ROOT / "third_party/CenterNet2"))
    from build_jev_counterfactual_dataset import load_gt, normalize_events
    from build_jev_counterfactual_v2 import (
        advance_off_state_for_key,
        build_formal_gmt_engine,
        event_maps,
        ordered_production_keys,
        source_commit,
    )
    from jev_counterfactual_v2 import seed_production_state_from_payload
    from jev_intra_video_chunking import file_sha256, save_state_snapshot
    from gtr.modeling.jev_perception_cache import FrozenPerceptionCache
    from run_jev_full_h8_fast_worker import PayloadLRU

    partition = json.loads(args.partition_manifest.read_text(encoding="utf-8"))
    plan = json.loads(args.chunk_plan.read_text(encoding="utf-8"))
    video_id = int(plan["video_id"])
    provenance_source_commit = os.environ.get("JEV_PROVENANCE_SOURCE_COMMIT", "").strip()
    if not provenance_source_commit:
        provenance_source_commit = source_commit(ROOT)
    trace = Path(partition["trace_by_video"]) / f"video_{video_id:02d}.jsonl"
    order_index = Path(partition["trace_by_video"]) / f"video_{video_id:02d}.orders.jsonl"
    events = normalize_events(trace, order_index=order_index, minimal=True)[video_id]
    actions, memories, by_key = event_maps(events)
    cache_keys = json.loads(
        (args.partition_manifest.parent / "cache_keys_by_video.json").read_text(encoding="utf-8")
    )[str(video_id)]
    cache = FrozenPerceptionCache(args.cache.resolve())
    cache_obj = PayloadLRU(cache, max_entries=512)
    keys, _seed_key = ordered_production_keys(
        cache_keys, lambda key: cache_obj.load(*key)
    )
    if not plan.get("chunks"):
        raise ValueError("chunk plan is empty")

    engine = build_formal_gmt_engine(
        config_file=args.config_file.resolve(),
        checkpoint=args.checkpoint.resolve(),
        device=args.device,
        view_num=args.view_num,
        history_limit=args.history_limit,
    )
    state, _seed_payload = seed_production_state_from_payload(
        cache_obj.load(*keys[0]), keys[0]
    )
    output = args.output.resolve()
    snapshot_dir = output / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshots = []
    cursor = 1
    for chunk in plan["chunks"]:
        start = int(chunk["key_start"])
        if start < cursor:
            raise ValueError("chunk plan key ranges overlap")
        for key_index in range(cursor, start):
            key = keys[key_index]
            advance_off_state_for_key(
                payload=cache_obj.load(*key),
                key=key,
                by_key=by_key,
                actions=actions,
                memories=memories,
                engine=engine,
                state=state,
            )
        snapshot = snapshot_dir / f"chunk_{int(chunk['chunk_index']):04d}.pt"
        metadata = {
            "schema_version": "jev_deterministic_intra_video_chunk_v1",
            "video_id": video_id,
            "chunk_index": int(chunk["chunk_index"]),
            "key_start": start,
            "trace": str(trace),
            "trace_sha256": file_sha256(trace),
            "order_index": str(order_index),
            "order_index_sha256": file_sha256(order_index),
            "cache_index_sha256": partition["cache_index_sha256"],
            "trajectory_rng_seed": state.trajectory_rng_seed,
            "trajectory_rng_calls": int(state.trajectory_rng_calls),
            "source_commit": provenance_source_commit,
            "warmup_semantics": "production_order_gmt_off",
        }
        digest = save_state_snapshot(snapshot, state, metadata)
        snapshots.append({"chunk": chunk, "path": str(snapshot), "sha256": digest, "metadata": metadata})
        cursor = start

    report = {
        "schema_version": "jev_deterministic_intra_video_chunk_v1",
        "status": "PASS",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "FUTURE_CANONICAL_H8_ONLY_NOT_ACTIVE_SMALL_GATE",
        "video_id": video_id,
        "source_commit": provenance_source_commit,
        "source_commit_capture": (
            "explicit_env_override"
            if os.environ.get("JEV_PROVENANCE_SOURCE_COMMIT", "").strip()
            else "process_completion_fallback"
        ),
        "chunk_plan": str(args.chunk_plan.resolve()),
        "trace_sha256": file_sha256(trace),
        "order_index_sha256": file_sha256(order_index),
        "warmup_policy": "one production-order OFF pass; explicit tracker/memory/history/RNG state snapshot at each chunk start",
        "snapshots": snapshots,
        "final_warmup_key_index": cursor,
        "source_keys": len(keys),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "warmup_manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "video_id": video_id, "snapshots": len(snapshots), "warmup_keys": cursor}, indent=2))


if __name__ == "__main__":
    main()
