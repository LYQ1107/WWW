"""Run one deterministic intra-video H=8 chunk from an OFF state snapshot.

The scheduler may launch this command once per chunk on different GPUs. It is
not used by the active video1 small-gate worker. Future workers may use the
durable ``--progress-file``/``--checkpoint-file`` boundary and ``--resume``;
the active video01 v2 process predates and does not use this protocol.
"""

from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time


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
    parser.add_argument(
        "--progress-file",
        type=Path,
        help="atomically updated progress JSON; defaults inside the chunk output",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=Path,
        help="atomically replaced mutable-state checkpoint; defaults inside the chunk output",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from the durable checkpoint and partial JSONL for this chunk",
    )
    parser.add_argument(
        "--canonical-commit",
        required=True,
        help="full SHA pinned to /data1/liuyeqiang/WWW_h8_frozen_<SHA>",
    )
    args = parser.parse_args()

    from run_jev_full_h8_fast_worker import assert_frozen_source

    assert_frozen_source(ROOT, args.canonical_commit)

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
    )
    from jev_intra_video_chunking import load_state_snapshot, file_sha256
    from run_jev_full_h8_fast_worker import PayloadLRU
    from gtr.modeling.jev_perception_cache import FrozenPerceptionCache
    from jev_v2_progress_recovery import (
        Progress,
        append_record,
        load_checkpoint,
        save_checkpoint,
        utc_now,
        write_progress,
    )

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

    output = args.output_root.resolve() / video_name / f"chunk_{chunk_index:04d}"
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "records.jsonl"
    partial_records = output / "records.jsonl.partial"
    progress_path = (
        args.progress_file.resolve()
        if args.progress_file is not None
        else output / "progress.json"
    )
    checkpoint_path = (
        args.checkpoint_file.resolve()
        if args.checkpoint_file is not None
        else output / "state.checkpoint.pt"
    )
    if records_path.exists():
        raise RuntimeError(f"refusing to overwrite completed chunk: {records_path}")

    trace_sha256 = str(plan["trace_sha256"])
    checkpoint_sha256 = "sha256:" + sha256(args.checkpoint.resolve())
    snapshot_sha256 = file_sha256(snapshot) if snapshot.is_file() else None
    provenance = {
        "schema_version": "jev_v2_chunk_resume_provenance_v1",
        "canonical_commit": str(args.canonical_commit),
        "video_id": video_id,
        "chunk_index": chunk_index,
        "key_start": int(chunk["key_start"]),
        "key_end": int(chunk["key_end"]),
        "trace_sha256": trace_sha256,
        "order_index_sha256": str(plan["order_index_sha256"]),
        "partition_manifest_sha256": file_sha256(args.partition_manifest.resolve()),
        "chunk_plan_sha256": file_sha256(args.chunk_plan.resolve()),
        "initial_state_snapshot_sha256": snapshot_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "horizon": int(args.horizon),
        "association_backend": "formal_gmt_transformer",
        "state_schema_version": int(STATE_SCHEMA_VERSION),
    }

    prior_progress = None
    if args.resume:
        if not partial_records.is_file() or not checkpoint_path.is_file():
            raise RuntimeError("--resume requires both partial records and checkpoint")
        state, prior_progress, loaded_provenance = load_checkpoint(
            checkpoint_path, expected_provenance=provenance
        )
        if loaded_provenance != provenance:
            raise RuntimeError("loaded checkpoint provenance differs after validation")
        next_key_index = int(prior_progress["next_key_index"])
        expected_records = int(prior_progress["completed_events"])
        lines = partial_records.read_bytes().splitlines()
        if len(lines) < expected_records:
            raise RuntimeError(
                "partial records are shorter than the durable progress checkpoint"
            )
        if len(lines) != expected_records:
            # A crash may leave fsync'ed records from the current key after the
            # last checkpoint.  Discard only those records and resume at the
            # checkpointed key boundary.
            temporary = partial_records.with_name(
                partial_records.name + f".reconcile.{os.getpid()}"
            )
            with temporary.open("wb") as handle:
                for line in lines[:expected_records]:
                    handle.write(line + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, partial_records)
        resume_key_start = next_key_index
    else:
        if partial_records.exists() or progress_path.exists() or checkpoint_path.exists():
            raise RuntimeError(
                "stale recovery artifacts exist; use --resume or move them aside"
            )
        state, snapshot_metadata = load_state_snapshot(snapshot)
        if int(snapshot_metadata.get("key_start", -1)) != int(chunk["key_start"]):
            raise ValueError("chunk snapshot key_start does not match chunk plan")
        if str(snapshot_metadata.get("trace_sha256")) != str(plan["trace_sha256"]):
            raise ValueError("chunk snapshot trace provenance mismatch")
        resume_key_start = int(chunk["key_start"])

    if args.resume:
        snapshot_metadata = {"resumed_from_checkpoint": True}

    started = time.monotonic()

    def sink(record):
        append_record(partial_records, record)

    def on_progress(info, current_state):
        previous_completed = int((prior_progress or {}).get("completed_events", 0))
        previous_branches = int((prior_progress or {}).get("branch_count", 0))
        total_events = int(
            (prior_progress or {}).get("total_events", info["total_events"])
        )
        progress = Progress(
            frame=int(info["key"][1]),
            view=int(info["key"][2]),
            event_order=int(info["key_index"]),
            completed_events=previous_completed + int(info["completed_events"]),
            total_events=total_events,
            branch_count=previous_branches + int(info["branch_count"]),
            elapsed_seconds=float(time.monotonic() - started),
            last_update_utc=utc_now(),
            next_key_index=int(info["next_key_index"]),
            total_keys=int(chunk["key_end"]),
        )
        write_progress(progress_path, progress)
        save_checkpoint(
            checkpoint_path,
            state=current_state,
            provenance=provenance,
            progress=progress,
        )

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
            key_start_index=resume_key_start,
            key_end_index=int(chunk["key_end"]),
            selected_key_range=(resume_key_start, int(chunk["key_end"])),
            progress_callback=on_progress,
        )
        if partial_records.is_file():
            with partial_records.open("ab") as handle:
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(partial_records, records_path)
    except Exception:
        raise

    provenance_source_commit = args.canonical_commit
    transformer_sha256 = "sha256:" + sha256(
        ROOT / "gtr/modeling/roi_heads/transformer.py"
    )
    counterfactual_engine_sha256 = "sha256:" + sha256(
        ROOT / "reproduction_tools/jev_counterfactual_v2.py"
    )
    adapter_sha256 = "sha256:" + sha256(
        ROOT / "reproduction_tools/jev_gmt_association_adapter.py"
    )
    record_question_counts = Counter()
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record_question_counts[str(json.loads(line)["question_type"])] += 1
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
        "source_commit_capture": "canonical_commit_argument_and_startup_git_rev_parse",
        "transformer_sha256": transformer_sha256,
        "counterfactual_engine_sha256": counterfactual_engine_sha256,
        "adapter_sha256": adapter_sha256,
        "initial_state_snapshot": str(snapshot),
        "initial_state_snapshot_sha256": snapshot_sha256,
        "initial_state_metadata": snapshot_metadata,
        "records": int(sum(record_question_counts.values())),
        "stats": dict(record_question_counts),
        "skipped": int(skipped),
        "records_path": str(records_path),
        "records_sha256": "sha256:" + sha256(records_path),
        "canonical_authority": "FUTURE_CANONICAL_ONLY_AFTER_EQUIVALENCE_GATE",
        "durable_recovery": {
            "progress": str(progress_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_provenance": provenance,
            "resume_supported": True,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "video_id": video_id, "chunk_index": chunk_index, "records": int(sum(record_question_counts.values())), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
