"""Run one deterministic intra-video H=8 chunk from an OFF state snapshot.

The scheduler may launch this command once per chunk on different GPUs. It is
not used by the active video1 small-gate worker.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-manifest", type=Path, required=True)
    parser.add_argument("--chunk-plan", type=Path, required=True)
    parser.add_argument(
        "--warmup-root",
        type=Path,
        help="directory containing snapshots/ from warmup_jev_intra_video_chunks.py",
    )
    parser.add_argument("--chunk-index", type=int, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--view-num", type=int, default=2)
    parser.add_argument("--history-limit", type=int, default=80)
    parser.add_argument("--horizon", type=int, default=8)
    args = parser.parse_args()

    os.environ.setdefault("PYTHONPATH", ":".join((str(ROOT), str(ROOT / "reproduction_tools"), str(ROOT / "third_party/CenterNet2"))))
    import sys

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "reproduction_tools"))
    sys.path.insert(0, str(ROOT / "third_party/CenterNet2"))
    from build_jev_counterfactual_dataset import load_gt
    from build_jev_counterfactual_v2 import (
        STATE_SCHEMA_VERSION,
        UTILITY_DEFINITION,
        TRAJECTORY_RNG_MASTER_SEED,
        TRAJECTORY_RNG_POLICY,
        build_formal_gmt_engine,
        ordered_production_keys,
        source_commit,
    )
    from jev_intra_video_chunking import load_state_snapshot, file_sha256
    from run_jev_full_h8_fast_worker import PayloadLRU
    from gtr.modeling.jev_perception_cache import FrozenPerceptionCache

    partition = json.loads(args.partition_manifest.read_text(encoding="utf-8"))
    plan = json.loads(args.chunk_plan.read_text(encoding="utf-8"))
    chunk_index = int(args.chunk_index)
    chunks = plan.get("chunks", [])
    if chunk_index < 0 or chunk_index >= len(chunks):
        raise IndexError(f"chunk index {chunk_index} outside plan")
    chunk = chunks[chunk_index]
    video_id = int(plan["video_id"])
    video_name = f"video_{video_id:02d}"
    trace = Path(partition["trace_by_video"]) / f"{video_name}.jsonl"
    order_index = Path(partition["trace_by_video"]) / f"{video_name}.orders.jsonl"
    cache_keys = json.loads(
        (args.partition_manifest.parent / "cache_keys_by_video.json").read_text(encoding="utf-8")
    )[str(video_id)]
    cache_for_order = FrozenPerceptionCache(args.cache.resolve())
    cache_keys, _seed_key = ordered_production_keys(
        cache_keys, lambda key: cache_for_order.load(*key)
    )
    warmup_root = (
        args.warmup_root.resolve()
        if args.warmup_root is not None
        else args.chunk_plan.parent.resolve()
    )
    snapshot = warmup_root / "snapshots" / f"chunk_{chunk_index:04d}.pt"
    state, snapshot_metadata = load_state_snapshot(snapshot)
    if int(snapshot_metadata.get("key_start", -1)) != int(chunk["key_start"]):
        raise ValueError("chunk snapshot key_start does not match chunk plan")
    if str(snapshot_metadata.get("trace_sha256")) != str(plan["trace_sha256"]):
        raise ValueError("chunk snapshot trace provenance mismatch")

    output = args.output_root.resolve() / video_name / f"chunk_{chunk_index:04d}"
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "records.jsonl"
    temporary = output / f"records.jsonl.tmp.{os.getpid()}"
    handle = temporary.open("w", encoding="utf-8")

    def sink(record):
        handle.write(json.dumps(record, sort_keys=True) + "\n")

    try:
        cache = FrozenPerceptionCache(args.cache.resolve())
        cache_obj = PayloadLRU(cache, max_entries=512)
        engine = build_formal_gmt_engine(
            config_file=args.config_file.resolve(),
            checkpoint=args.checkpoint.resolve(),
            device=args.device,
            view_num=args.view_num,
            history_limit=args.history_limit,
        )
        gt_bundle = load_gt(args.annotations.resolve())
        from build_jev_counterfactual_v2 import build_v2_records

        _records, stats, skipped = build_v2_records(
            trace=trace,
            cache_root=args.cache.resolve(),
            annotations=args.annotations.resolve(),
            checkpoint_hash=sha256(args.checkpoint.resolve()),
            horizon=int(args.horizon),
            association_backend="formal_gmt_transformer",
            engine=engine,
            cache_obj=cache_obj,
            gt_bundle=gt_bundle,
            cache_keys_by_video={video_id: cache_keys},
            order_index=order_index,
            record_sink=sink,
            minimal_events=True,
            video_ids=[video_id],
            initial_state=state,
            key_start_index=int(chunk["key_start"]),
            key_end_index=int(chunk["key_end"]),
            selected_key_range=(int(chunk["key_start"]), int(chunk["key_end"])),
        )
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        temporary.replace(records_path)
    except Exception:
        handle.close()
        raise

    record_count = sum(int(value) for value in stats.values())
    provenance_source_commit = os.environ.get("JEV_PROVENANCE_SOURCE_COMMIT", "").strip()
    if not provenance_source_commit:
        provenance_source_commit = source_commit(ROOT)
    manifest = {
        "status": "COMPLETE",
        "schema_version": "jev_deterministic_intra_video_chunk_v1",
        "created_utc": utc_now(),
        "video_id": video_id,
        "chunk_index": chunk_index,
        "chunk": chunk,
        "trace": str(trace),
        "trace_sha256": plan["trace_sha256"],
        "order_index": str(order_index),
        "order_index_sha256": plan["order_index_sha256"],
        "cache_index_sha256": partition["cache_index_sha256"],
        "checkpoint_sha256": "sha256:" + sha256(args.checkpoint.resolve()),
        "horizon": int(args.horizon),
        "association_backend": "formal_gmt_transformer",
        "counterfactual_engine": "cached_perception_mutable_association_v2",
        "state_schema_version": STATE_SCHEMA_VERSION,
        "record_schema_version": 1,
        "feature_schema_version": "jev_runtime_state_v2",
        "utility_definition": UTILITY_DEFINITION,
        "trajectory_rng_policy": TRAJECTORY_RNG_POLICY,
        "trajectory_rng_master_seed": TRAJECTORY_RNG_MASTER_SEED,
        "trajectory_rng_video_seed": TRAJECTORY_RNG_MASTER_SEED + video_id,
        "source_commit": provenance_source_commit,
        "source_commit_capture": (
            "explicit_env_override"
            if os.environ.get("JEV_PROVENANCE_SOURCE_COMMIT", "").strip()
            else "process_completion_fallback"
        ),
        "initial_state_snapshot": str(snapshot),
        "initial_state_snapshot_sha256": file_sha256(snapshot),
        "initial_state_metadata": snapshot_metadata,
        "records": record_count,
        "stats": stats,
        "skipped": int(skipped),
        "records_path": str(records_path),
        "records_sha256": "sha256:" + sha256(records_path),
        "canonical_authority": "FUTURE_CANONICAL_ONLY_AFTER_EQUIVALENCE_GATE",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "video_id": video_id, "chunk_index": chunk_index, "records": record_count, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
