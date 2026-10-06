"""Persistent one-GPU worker for the full formal H=8 counterfactual build."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
import threading
import time
import traceback
from typing import Any, Callable, Mapping, Optional


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class PayloadLRU:
    """Bound the host-side immutable payload cache without changing values."""

    def __init__(self, cache: Any, max_entries: int = 512):
        self.cache = cache
        self.max_entries = max(32, int(max_entries))
        self.payloads = OrderedDict()

    def load(self, video_id: int, frame: int, view: int):
        key = (int(video_id), int(frame), int(view))
        payload = self.payloads.get(key)
        if payload is None:
            payload = self.cache.load(*key)
            self.payloads[key] = payload
            while len(self.payloads) > self.max_entries:
                self.payloads.popitem(last=False)
        else:
            self.payloads.move_to_end(key)
        return payload


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def modify_queue(path: Path, update: Callable[[dict[str, Any]], Any]) -> Any:
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state = json.loads(path.read_text(encoding="utf-8"))
        result = update(state)
        state["updated_utc"] = utc_now()
        atomic_json(path, state)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return result


def complete_artifact(video_root: Path) -> bool:
    manifest_path = video_root / "manifest.json"
    records_path = video_root / "records.jsonl"
    if not manifest_path.is_file() or not records_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return manifest.get("status") == "COMPLETE" and int(manifest.get("records", -1)) >= 0


def claim_next(queue_path: Path, worker_id: str) -> Optional[dict[str, Any]]:
    def update(state: dict[str, Any]):
        pending = [
            item
            for item in state["videos"].values()
            if item.get("status") == "PENDING"
        ]
        if not pending:
            return None
        item = max(
            pending,
            key=lambda value: (
                int(value.get("workload_score", 0)),
                int(value.get("main_decisions", 0)),
                -int(value["video_id"]),
            ),
        )
        video_id = str(item["video_id"])
        item["status"] = "RUNNING"
        item["worker"] = worker_id
        item["pid"] = os.getpid()
        item["started_utc"] = utc_now()
        item["updated_utc"] = utc_now()
        return dict(item)

    return modify_queue(queue_path, update)


def update_video(queue_path: Path, video_id: int, **fields: Any) -> None:
    def update(state: dict[str, Any]):
        item = state["videos"][str(int(video_id))]
        item.update(fields)
        item["updated_utc"] = utc_now()
        return None

    modify_queue(queue_path, update)


class Heartbeat:
    def __init__(self, queue_path: Path, video_id: int, worker_id: str, interval: float = 60.0):
        self.queue_path = queue_path
        self.video_id = int(video_id)
        self.worker_id = worker_id
        self.interval = float(interval)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name=f"heartbeat-{video_id}", daemon=True)

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            try:
                update_video(
                    self.queue_path,
                    self.video_id,
                    heartbeat_utc=utc_now(),
                    worker=self.worker_id,
                    pid=os.getpid(),
                )
            except Exception:
                # The build itself remains the source of truth.  A transient
                # queue heartbeat failure must not corrupt an in-progress shard.
                pass

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback_value):
        self.stop_event.set()
        self.thread.join(timeout=5.0)


def open_record_sink(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    handle = temporary.open("w", encoding="utf-8")

    def sink(record: Mapping[str, Any]) -> None:
        handle.write(json.dumps(record, sort_keys=True) + "\n")

    return handle, temporary, sink


def build_one(
    *,
    task: Mapping[str, Any],
    args: argparse.Namespace,
    partition_manifest: Mapping[str, Any],
    cache_keys_by_video: Mapping[str, Any],
    cache_obj: Any,
    gt_bundle: Any,
    engine: Any,
    checkpoint_hash: str,
    source_commit_value: str,
) -> dict[str, Any]:
    # Imports happen after the worker has been launched with its physical GPU
    # in CUDA_VISIBLE_DEVICES.  The model and cache are still reused between
    # all videos claimed by this process.
    from build_jev_counterfactual_v2 import (
        TRAJECTORY_RNG_MASTER_SEED,
        TRAJECTORY_RNG_POLICY,
        build_v2_records,
    )

    video_id = int(task["video_id"])
    video_name = f"video_{video_id:02d}"
    trace_path = Path(partition_manifest["trace_by_video"]) / f"{video_name}.jsonl"
    order_path = Path(partition_manifest["trace_by_video"]) / f"{video_name}.orders.jsonl"
    video_meta = partition_manifest["videos"][str(video_id)]
    output_root = args.output_root.resolve() / video_name
    records_path = output_root / "records.jsonl"
    manifest_path = output_root / "manifest.json"
    if complete_artifact(output_root):
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if not trace_path.is_file() or not order_path.is_file():
        raise FileNotFoundError(f"missing partition for video {video_id}: {trace_path}")

    record_handle, temporary_records, record_sink = open_record_sink(records_path)
    try:
        _records, stats, skipped = build_v2_records(
            trace=trace_path,
            cache_root=args.cache.resolve(),
            annotations=args.annotations.resolve(),
            checkpoint_hash=checkpoint_hash,
            horizon=int(args.horizon),
            association_backend="formal_gmt_transformer",
            engine=engine,
            cache_obj=cache_obj,
            gt_bundle=gt_bundle,
            cache_keys_by_video={video_id: cache_keys_by_video[str(video_id)]},
            order_index=order_path,
            record_sink=record_sink,
            minimal_events=True,
        )
        record_handle.flush()
        os.fsync(record_handle.fileno())
        record_handle.close()
        os.replace(temporary_records, records_path)
    except Exception:
        record_handle.close()
        raise
    artifact_hash = sha256(records_path)
    record_count = sum(int(value) for value in stats.values())
    report = {
        "status": "COMPLETE",
        "created_utc": utc_now(),
        "video_id": video_id,
        "trace_partition": str(trace_path),
        "trace_partition_sha256": "sha256:" + sha256(trace_path),
        "source_order_index": str(order_path),
        "source_order_index_sha256": "sha256:" + sha256(order_path),
        "source_trace": partition_manifest["trace"],
        "source_trace_sha256": partition_manifest["trace_sha256"],
        "source_line_count": int(video_meta["lines"]),
        "source_main_decisions": int(video_meta["main_decisions"]),
        "cache": str(args.cache.resolve()),
        "cache_index_sha256": partition_manifest["cache_index_sha256"],
        "annotations": str(args.annotations.resolve()),
        "annotations_sha256": sha256(args.annotations.resolve()),
        "gmt_checkpoint": str(args.checkpoint.resolve()),
        "gmt_checkpoint_sha256": "sha256:" + checkpoint_hash,
        "horizon": int(args.horizon),
        "association_backend": "formal_gmt_transformer",
        "formal_gmt_association_adapter": True,
        "counterfactual_engine": "cached_perception_mutable_association_v2",
        "state_schema_version": 2,
        "utility_definition": "future_correct_identity_duration - 0.5*future_identity_switches - 0.25*future_fragmentation - 0.5*future_collisions - memory_contamination; sample_weight=0 for uninformative futures",
        "trajectory_rng_policy": TRAJECTORY_RNG_POLICY,
        "trajectory_rng_master_seed": TRAJECTORY_RNG_MASTER_SEED,
        "trajectory_rng_video_seed": TRAJECTORY_RNG_MASTER_SEED + video_id,
        "trajectory_rng_state_cloned_per_counterfactual_branch": True,
        "proposal_reused_across_legal_actions": True,
        "reassociate_reuses_score_matrix": True,
        "second_transformer_call_for_reassociate": False,
        "trajectory_rng_transformer_sha256": "sha256:" + sha256(
            Path(__file__).resolve().parents[1]
            / "gtr" / "modeling" / "roi_heads" / "transformer.py"
        ),
        "trajectory_rng_counterfactual_engine_sha256": "sha256:" + sha256(
            Path(__file__).resolve().parents[1]
            / "reproduction_tools" / "jev_counterfactual_v2.py"
        ),
        "trajectory_rng_adapter_sha256": "sha256:" + sha256(
            Path(__file__).resolve().parents[1]
            / "reproduction_tools" / "jev_gmt_association_adapter.py"
        ),
        "records": record_count,
        "records_by_question": stats,
        "skipped_events": int(skipped),
        "records_artifact": str(records_path),
        "records_artifact_sha256": "sha256:" + artifact_hash,
        "source_commit": source_commit_value,
        "official_test_read": False,
        "sampling": False,
        "truncation": False,
        "worker": args.worker_id,
        "device": args.device,
    }
    atomic_json(manifest_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--view-num", type=int, default=2)
    parser.add_argument("--history-limit", type=int, default=80)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--payload-cache-limit", type=int, default=512)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_jev_counterfactual_dataset import load_gt
    from build_jev_counterfactual_v2 import build_formal_gmt_engine, source_commit
    from gtr.modeling.jev_perception_cache import FrozenPerceptionCache

    partition_root = args.partition_root.resolve()
    partition_manifest = json.loads(
        (partition_root / "partition_manifest.json").read_text(encoding="utf-8")
    )
    if partition_manifest.get("status") != "COMPLETE":
        raise RuntimeError("full H=8 trace partition is not complete")
    cache_keys_by_video = json.loads(
        (partition_root / "cache_keys_by_video.json").read_text(encoding="utf-8")
    )
    args.output_root.resolve().mkdir(parents=True, exist_ok=True)
    cache_obj = PayloadLRU(
        FrozenPerceptionCache(args.cache.resolve()),
        max_entries=args.payload_cache_limit,
    )
    gt_bundle = load_gt(args.annotations.resolve())
    checkpoint_hash = sha256(args.checkpoint.resolve())
    source_commit_value = source_commit(Path(__file__).resolve().parents[1])
    engine = build_formal_gmt_engine(
        config_file=args.config_file.resolve(),
        checkpoint=args.checkpoint.resolve(),
        device=args.device,
        view_num=args.view_num,
        history_limit=args.history_limit,
    )

    while True:
        task = claim_next(args.queue.resolve(), args.worker_id)
        if task is None:
            return
        video_id = int(task["video_id"])
        update_video(
            args.queue.resolve(),
            video_id,
            output_root=str(args.output_root.resolve() / f"video_{video_id:02d}"),
            status="RUNNING",
        )
        try:
            with Heartbeat(args.queue.resolve(), video_id, args.worker_id):
                report = build_one(
                    task=task,
                    args=args,
                    partition_manifest=partition_manifest,
                    cache_keys_by_video=cache_keys_by_video,
                    cache_obj=cache_obj,
                    gt_bundle=gt_bundle,
                    engine=engine,
                    checkpoint_hash=checkpoint_hash,
                    source_commit_value=source_commit_value,
                )
            update_video(
                args.queue.resolve(),
                video_id,
                status="COMPLETE",
                finished_utc=utc_now(),
                records=int(report["records"]),
                manifest=str(args.output_root.resolve() / f"video_{video_id:02d}" / "manifest.json"),
            )
        except Exception as exc:  # noqa: BLE001 - preserve failure in queue state
            error = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-8000:]
            update_video(
                args.queue.resolve(),
                video_id,
                status="FAILED",
                finished_utc=utc_now(),
                error=error,
            )
            print(error, flush=True)


if __name__ == "__main__":
    main()
