"""Build v2 JEV labels from a frozen perception cache.

The builder is intentionally explicit about the association backend.  The
default cosine backend is a deterministic contract backend for validating the
branch protocol; a formal GMT result must pass a GMT association-transformer
adapter and is marked accordingly in the manifest.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from build_jev_counterfactual_dataset import (
    event_key,
    iou,
    load_gt,
    normalize_events,
    observed_gt,
    sha256,
)
from gtr.modeling.jev_perception_cache import FrozenPerceptionCache
from jev_counterfactual_v2 import (
    ENGINE_VERSION,
    CachedPerceptionMutableAssociationV2,
    MutableGMTState,
)
from jev_dataset_tools import make_record
from jev_mutable_branch import is_main_decision


CANONICAL_CHECKPOINT_SHA256 = (
    "cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8"
)
FINAL_SELECTION_LOCK = Path(
    "/data1/liuyeqiang/WWW/outputs/research_final_v2/manifests/FINAL_SELECTION_LOCK.json"
)
STATE_SCHEMA_VERSION = 2
UTILITY_DEFINITION = (
    "future_correct_identity_duration - 0.5*future_identity_switches - "
    "0.25*future_fragmentation - 0.5*future_collisions - memory_contamination; "
    "sample_weight=0 for uninformative futures"
)
PRELOCK_SENTINEL = "DO_NOT_USE_FOR_SELECTION"


def source_commit(source_root: Path) -> str:
    """Return the exact source revision that produced a shard."""

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=source_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot determine source commit for {source_root}") from exc


def cache_digest(cache_root: Path) -> str:
    index = cache_root / "index.jsonl"
    return sha256(index)


def detection_target(
    *,
    video_id: int,
    frame: int,
    view: int,
    box: Sequence[float],
    image_size: Sequence[int],
    images: Mapping[Tuple[int, int, int], int],
    gt_by_image: Mapping[int, Sequence[Tuple[int, Sequence[float]]]],
    image_meta: Mapping[int, Mapping[str, Any]],
) -> Optional[int]:
    image_id = images.get((video_id, view + 1, frame))
    if image_id is None:
        image_id = images.get((video_id, view, frame))
    if image_id is None or image_id not in image_meta:
        return None
    image = image_meta[image_id]
    model_h, model_w = (float(image_size[0]), float(image_size[1]))
    scaled = list(float(value) for value in box[:4])
    if model_h > 0 and model_w > 0:
        scaled = [
            scaled[0] * float(image["width"]) / model_w,
            scaled[1] * float(image["height"]) / model_h,
            scaled[2] * float(image["width"]) / model_w,
            scaled[3] * float(image["height"]) / model_h,
        ]
    candidates = gt_by_image.get(image_id, ())
    if not candidates:
        return None
    target, overlap = max(
        ((int(instance_id), iou(scaled, target_box)) for instance_id, target_box in candidates),
        key=lambda item: item[1],
    )
    return target if overlap >= 0.5 else None


def score_rollout(
    steps: Sequence[Mapping[str, Any]],
    *,
    images,
    gt_by_image,
    image_meta,
    start_frame: Optional[int] = None,
    max_horizon: Optional[int] = None,
) -> Dict[str, float]:
    track_targets: Dict[int, int] = {}
    target_tracks: Dict[int, int] = {}
    correct_duration = 0
    switches = 0
    fragments = 0
    collisions = 0
    memory_contamination = 0
    contamination_duration = 0
    false_reactivation = 0
    new_id_fragmentation = 0
    informative = 0
    memory_targets: Dict[int, int] = {}
    for item in steps:
        payload = item["payload"]
        if start_frame is not None and max_horizon is not None:
            if int(payload["frame"]) > int(start_frame) + int(max_horizon):
                continue
        result = item["result"]
        video_id = int(payload["video_id"])
        frame = int(payload["frame"])
        view = int(payload["view"])
        boxes = payload["pred_boxes"]
        image_size = payload["image_size"]
        committed = result["committed_track_ids"]
        actions = item.get("actions", {})
        memory_actions = item.get("memory_actions", {})
        frame_target_tracks: Dict[int, int] = {}
        for row, track_id in committed.items():
            target = detection_target(
                video_id=video_id,
                frame=frame,
                view=view,
                box=boxes[int(row)].tolist(),
                image_size=image_size,
                images=images,
                gt_by_image=gt_by_image,
                image_meta=image_meta,
            )
            if target is None:
                continue
            informative += 1
            track_id = int(track_id)
            previous_target = track_targets.get(track_id)
            if previous_target is not None and previous_target != target:
                switches += 1
            previous_track = target_tracks.get(target)
            if previous_track is not None and previous_track != track_id:
                fragments += 1
                if actions.get(int(row)) == "START_NEW":
                    new_id_fragmentation += 1
            if target in frame_target_tracks and frame_target_tracks[target] != track_id:
                collisions += 1
            frame_target_tracks[target] = track_id
            track_targets[track_id] = int(target)
            target_tracks[int(target)] = track_id
            correct_duration += int(previous_target is None or previous_target == target)
            if actions.get(int(row)) == "REACTIVATE_OLD" and previous_target != target:
                false_reactivation += 1
            if int(row) in memory_actions and memory_actions[int(row)] == "WRITE_MEMORY":
                old_memory = memory_targets.get(track_id)
                if old_memory is not None and old_memory != target:
                    memory_contamination += 1
                    contamination_duration += 1
                memory_targets[track_id] = int(target)

    utility = (
        float(correct_duration)
        - 0.5 * float(switches)
        - 0.25 * float(fragments)
        - 0.5 * float(collisions)
        - float(memory_contamination)
    )
    return {
        "utility": utility,
        "sample_weight": 1.0 if informative else 0.0,
        "informative": bool(informative),
        "future_correct_identity_duration": float(correct_duration),
        "future_identity_switches": float(switches),
        "future_fragmentation": float(fragments),
        "future_collisions": float(collisions),
        "memory_contamination": float(memory_contamination),
        "contamination_duration": float(contamination_duration),
        "recovery_latency": 0.0,
        "false_reactivation": float(false_reactivation),
        "new_id_fragmentation": float(new_id_fragmentation),
        "window_idf1": float(correct_duration / max(1, informative)),
        "window_assa_proxy": float(correct_duration / max(1, informative)),
        "future_events": float(informative),
    }


def event_maps(events: Sequence[Mapping[str, Any]]):
    actions: Dict[Tuple[int, int, int], Dict[int, str]] = {}
    memories: Dict[Tuple[int, int, int], Dict[int, str]] = {}
    by_key: Dict[Tuple[int, int, int], List[Mapping[str, Any]]] = {}
    for event in events:
        if not is_main_decision(event):
            continue
        context = event["context"]
        key = (
            int(event["_video_id"]),
            int(event["_frame"]),
            int(event["_view"]),
        )
        by_key.setdefault(key, []).append(event)
        row = context.get("detection_index")
        if row is None:
            continue
        row = int(row)
        action = str(event.get("off_action") or event.get("proposed_action"))
        if event.get("question") == "MEMORY_DECISION":
            memories.setdefault(key, {})[row] = action
        elif event.get("question") in {"MATCH_DECISION", "REACTIVATION_DECISION"}:
            actions.setdefault(key, {})[row] = action
    return actions, memories, by_key


def build_v2_records(
    *,
    trace: Path,
    cache_root: Path,
    annotations: Path,
    checkpoint_hash: str,
    horizon: int,
    association_backend: str,
    engine: Optional[CachedPerceptionMutableAssociationV2] = None,
    cache_obj: Optional[FrozenPerceptionCache] = None,
    gt_bundle: Optional[Tuple[Any, Any, Any, Any]] = None,
    cache_keys_by_video: Optional[Mapping[int, Sequence[Sequence[int]]]] = None,
    order_index: Optional[Path] = None,
    record_sink: Optional[Callable[[Mapping[str, Any]], None]] = None,
    minimal_events: bool = False,
    max_events: Optional[int] = None,
    video_ids: Optional[Sequence[int]] = None,
    max_events_per_video: Optional[int] = None,
    derive_horizons: Optional[Sequence[int]] = None,
):
    if association_backend not in {"cosine_contract", "formal_gmt_transformer"}:
        raise ValueError(f"unsupported association backend: {association_backend}")
    horizons = sorted({int(value) for value in (derive_horizons or (horizon,))})
    if not horizons or any(value < 1 or value > int(horizon) for value in horizons):
        raise ValueError("derive_horizons must be positive and no greater than horizon")
    if int(horizon) not in horizons:
        horizons.append(int(horizon))
        horizons.sort()
    if gt_bundle is None:
        videos, images, gt_by_image, image_meta = load_gt(annotations)
    else:
        videos, images, gt_by_image, image_meta = gt_bundle
    cache = cache_obj if cache_obj is not None else FrozenPerceptionCache(cache_root)
    # Keep the legacy/equivalence path eager, while allowing the production
    # worker to inject a bounded lazy cache.  Both paths return the same
    # immutable payload values; only host-side residency differs.
    eager_payloads = None
    if cache_obj is None:
        eager_payloads = {}
    grouped = normalize_events(
        trace,
        video_ids=video_ids,
        order_index=order_index,
        minimal=minimal_events,
    )
    if video_ids is not None:
        requested = {int(value) for value in video_ids}
        missing = sorted(requested - set(grouped))
        if missing:
            raise ValueError(f"requested video ids are absent from trace: {missing}")
        grouped = {video_id: grouped[video_id] for video_id in sorted(requested)}
    if max_events_per_video is not None:
        if max_events_per_video < 1:
            raise ValueError("max_events_per_video must be positive")
        grouped = {
            video_id: events[: int(max_events_per_video)]
            for video_id, events in grouped.items()
        }
    if max_events is not None:
        if max_events < 1:
            raise ValueError("max_events must be positive")
        remaining = int(max_events)
        limited = {}
        for video_id in sorted(grouped):
            if remaining <= 0:
                break
            limited[video_id] = grouped[video_id][:remaining]
            remaining -= len(limited[video_id])
        grouped = limited
    if engine is None:
        if association_backend != "cosine_contract":
            raise ValueError("formal_gmt_transformer requires an injected GMT engine")
        engine = CachedPerceptionMutableAssociationV2()
    records = [] if record_sink is None else None
    stats: Dict[str, int] = {}
    skipped = 0
    for video_id, events in grouped.items():
        sequence = str(videos.get(video_id, {}).get("file_name", video_id))
        actions, memories, by_key = event_maps(events)
        if cache_keys_by_video is None:
            keys = [key for key in cache.keys() if key[0] == int(video_id)]
        else:
            keys = [tuple(int(value) for value in key) for key in cache_keys_by_video.get(int(video_id), ())]
        keys.sort(key=lambda key: (key[1], key[2]))
        # A bounded protocol subset does not need to advance the mutable
        # state through the remainder of the sequence.  Keep enough cached
        # frames to evaluate every selected event's future horizon, while
        # leaving the unbounded/full replay path unchanged.
        if (max_events is not None or max_events_per_video is not None) and events:
            last_selected_frame = max(int(event["_frame"]) for event in events)
            replay_until = last_selected_frame + int(horizon)
            keys = [key for key in keys if int(key[1]) <= replay_until]
        if eager_payloads is not None:
            eager_payloads = {key: cache.load(*key) for key in keys}

        def payload_for(key):
            return eager_payloads[key] if eager_payloads is not None else cache.load(*key)

        state = MutableGMTState()
        for key_index, key in enumerate(keys):
            payload = payload_for(key)
            current_events = by_key.get(key, ())
            for event in current_events:
                question = str(event.get("question"))
                if question not in {"MATCH_DECISION", "MEMORY_DECISION", "REACTIVATION_DECISION"}:
                    continue
                legal = list(event.get("legal_actions", ()))
                row = event["context"].get("detection_index")
                if row is None or not legal:
                    skipped += 1
                    continue
                row = int(row)
                outcome_map = {}
                for candidate in legal:
                    branch = state.clone()
                    branch_steps = []
                    current_actions = dict(actions.get(key, {}))
                    current_memories = dict(memories.get(key, {}))
                    if question == "MEMORY_DECISION":
                        current_memories[row] = candidate
                    else:
                        current_actions[row] = candidate
                    current_result = engine.step(
                        payload,
                        branch,
                        actions=current_actions,
                        memory_actions=current_memories,
                    )
                    branch_steps.append(
                        {
                            "payload": payload,
                            "result": current_result,
                            "actions": current_actions,
                            "memory_actions": current_memories,
                        }
                    )
                    for future_key in keys[key_index + 1 :]:
                        if int(future_key[1]) > int(key[1]) + horizon:
                            break
                        future_payload = payload_for(future_key)
                        future_actions = dict(actions.get(future_key, {}))
                        future_memories = dict(memories.get(future_key, {}))
                        future_result = engine.step(
                            future_payload,
                            branch,
                            actions=future_actions,
                            memory_actions=future_memories,
                        )
                        branch_steps.append(
                            {
                                "payload": future_payload,
                                "result": future_result,
                                "actions": future_actions,
                                "memory_actions": future_memories,
                            }
                        )
                    outcome_map[candidate] = {
                        str(branch_horizon): score_rollout(
                            branch_steps,
                            images=images,
                            gt_by_image=gt_by_image,
                            image_meta=image_meta,
                            start_frame=int(key[1]),
                            max_horizon=branch_horizon,
                        )
                        for branch_horizon in horizons
                    }

                state_data = {
                    "feature_vector": [float(value) for value in event["state_feature_vector"]],
                    "online_context": {
                        "video_id": int(video_id),
                        "frame": int(event["_frame"]),
                        "view": int(event["_view"]),
                        "event_order": int(event["_order"]),
                        "detection_index": row,
                        "proposal_track_id": event["context"].get("proposal_track_id"),
                        "alternate_track_id": event["context"].get("alternate_track_id"),
                        "track_id": event["context"].get("track_id"),
                    },
                    "counterfactual_engine": ENGINE_VERSION,
                    "association_backend": association_backend,
                    "perception_cache_version": str(payload.get("cache_version")),
                }
                record = make_record(
                    dataset="VisionTrack",
                    sequence=sequence,
                    frame=int(event["_frame"]),
                    view=int(event["_view"]),
                    question_type=question,
                    state=state_data,
                    legal_actions=legal,
                    action_outcomes={
                        candidate: outcome_map[candidate][str(horizon)]
                        for candidate in legal
                    },
                    gmt_checkpoint_sha256=checkpoint_hash,
                    horizon=horizon,
                )
                # Preserve raw cumulative outcomes from one max-horizon rollout
                # so smaller policy horizons can be derived without a second
                # association simulation.
                record["horizon_outcomes"] = {
                    str(branch_horizon): {
                        candidate: outcome_map[candidate][str(branch_horizon)]
                        for candidate in legal
                    }
                    for branch_horizon in horizons
                }
                if record_sink is None:
                    records.append(record)
                else:
                    record_sink(record)
                stats[question] = stats.get(question, 0) + 1

            state_actions = actions.get(key)
            state_memories = memories.get(key)
            engine.step(payload, state, actions=state_actions, memory_actions=state_memories)
    return (records if records is not None else []), stats, skipped


def build_formal_gmt_engine(
    *,
    config_file: Path,
    checkpoint: Path,
    device: str,
    view_num: int,
    history_limit: Optional[int],
) -> CachedPerceptionMutableAssociationV2:
    """Load the fixed GMT checkpoint used by formal cached replay.

    The detector is never called here: all proposals come from the immutable
    cache.  Only the repository association transformer is evaluated for the
    branch-local historical state.
    """
    import torch
    from detectron2.checkpoint import DetectionCheckpointer
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model
    from centernet.config import add_centernet_config
    from gtr.config import add_gtr_config
    from jev_counterfactual_v2 import CachedPerceptionMutableAssociationV2
    from jev_gmt_association_adapter import GMTAssociationTransformerAdapter

    cfg = get_cfg()
    add_centernet_config(cfg)
    add_gtr_config(cfg)
    cfg.merge_from_file(str(config_file))
    cfg.defrost()
    cfg.MODEL.WEIGHTS = str(checkpoint)
    # The cache already contains detector/ReID outputs.  This flag documents
    # that formal replay must not invoke the live JEV runtime.
    cfg.MODEL.JEV.ENABLED = False
    cfg.freeze()
    model = build_model(cfg)
    model.to(torch.device(device))
    model.eval()
    DetectionCheckpointer(model).resume_or_load(str(checkpoint), resume=False)
    if history_limit is None:
        history_limit = int(cfg.INPUT.VIDEO.TEST_LEN) * max(1, int(view_num))
    adapter = GMTAssociationTransformerAdapter(
        model,
        view_num=max(1, int(view_num)),
        history_limit=max(1, int(history_limit)),
    )
    configured_threshold = float(cfg.VIDEO_TEST.OVERLAP_THRESH)
    engine = CachedPerceptionMutableAssociationV2(
        association_fn=adapter,
        acceptance_threshold=configured_threshold,
        not_mult_thresh=bool(cfg.VIDEO_TEST.NOT_MULT_THRESH),
        history_limit=max(1, int(history_limit)),
    )
    if abs(float(engine.acceptance_threshold) - configured_threshold) > 1e-12:
        raise AssertionError(
            "formal replay threshold diverged from cfg.VIDEO_TEST.OVERLAP_THRESH"
        )
    return engine


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gmt-checkpoint", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument(
        "--association-backend",
        choices=("cosine_contract", "formal_gmt_transformer"),
        default="cosine_contract",
    )
    parser.add_argument("--config-file", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--view-num", type=int, default=2)
    parser.add_argument("--history-limit", type=int)
    parser.add_argument(
        "--max-events",
        type=int,
        help="deterministic protocol/smoke limit; omit for a complete trace",
    )
    parser.add_argument("--video-ids", type=int, nargs="*")
    parser.add_argument("--max-events-per-video", type=int)
    parser.add_argument(
        "--derive-horizons",
        type=int,
        nargs="*",
        help="store cumulative raw outcomes for these horizons in one max-horizon rollout",
    )
    parser.add_argument(
        "--prelock-diagnostic",
        action="store_true",
        help="explicitly quarantine a TEST diagnostic; never an official artifact",
    )
    args = parser.parse_args()
    if args.horizon < 1:
        raise ValueError("horizon must be positive")
    # TEST counterfactual records contain official labels and are not allowed
    # to enter the architecture/policy search pool.  A pre-lock diagnostic
    # process may already exist; this guard applies to every new process and
    # makes the protocol boundary fail closed.
    official_test_generation_authorized = False
    official_test_lock_sha256 = None
    official_test_selection_protocol_sha256 = None
    if args.prelock_diagnostic and args.annotations.name != "test.json":
        raise ValueError("--prelock-diagnostic is only valid with TEST annotations")
    prelock_diagnostic = bool(args.prelock_diagnostic)
    if args.annotations.name == "test.json" and not prelock_diagnostic:
        try:
            lock = json.loads(FINAL_SELECTION_LOCK.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            lock = None
        selection_path = Path(str((lock or {}).get("selection_protocol", "")))
        selection_valid = bool(
            selection_path.is_file()
            and "sha256:" + sha256(selection_path)
            == (lock or {}).get("selection_protocol_sha256")
        ) if isinstance(lock, dict) else False
        allowed = bool(
            isinstance(lock, dict)
            and lock.get("lock_type") == "FINAL_SELECTION_LOCK"
            and lock.get("selection_scope") == "CANONICAL_MODEL20000_ONLY"
            and lock.get("official_test_authority") is True
            and lock.get("canonical_checkpoint_authority") is True
            and lock.get("gmt_checkpoint_sha256")
            == "sha256:" + CANONICAL_CHECKPOINT_SHA256
            and (lock.get("official_test_gate") or {}).get(
                "lock_created_before_official_test"
            ) is True
            and (lock.get("official_test_gate") or {}).get(
                "official_test_read_allowed_after_lock"
            ) is True
            and selection_valid
        )
        if not allowed:
            raise RuntimeError(
                "official TEST counterfactual generation is blocked until the "
                "canonical FINAL_SELECTION_LOCK for model_20000 exists"
            )
        official_test_generation_authorized = True
        official_test_lock_sha256 = "sha256:" + sha256(FINAL_SELECTION_LOCK)
        official_test_selection_protocol_sha256 = (lock or {}).get(
            "selection_protocol_sha256"
        )
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    sentinel = args.output.parent / PRELOCK_SENTINEL
    if sentinel.exists() and not prelock_diagnostic:
        raise RuntimeError(
            f"refusing to read quarantined pre-lock TEST directory: {args.output.parent}"
        )
    trace = args.trace.resolve()
    cache = args.cache.resolve()
    annotations = args.annotations.resolve()
    checkpoint = args.gmt_checkpoint.resolve()
    source_root = Path(__file__).resolve().parents[1]
    config_file = args.config_file.resolve() if args.config_file is not None else None
    if prelock_diagnostic:
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        if not sentinel.exists():
            sentinel.write_text(
                "QUARANTINED_PRELOCK_TEST_DIAGNOSTIC\n"
                "NOT_FOR_SELECTION\n"
                "NOT_FOR_FINAL_REPORT\n",
                encoding="utf-8",
            )
    engine = None
    if args.association_backend == "formal_gmt_transformer":
        if args.config_file is None:
            raise ValueError("--config-file is required for formal_gmt_transformer")
        engine = build_formal_gmt_engine(
            config_file=args.config_file.resolve(),
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
        association_backend=args.association_backend,
        engine=engine,
        max_events=args.max_events,
        video_ids=args.video_ids,
        max_events_per_video=args.max_events_per_video,
        derive_horizons=args.derive_horizons,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    report = {
        "status": "PASS",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "trace": str(trace),
        "trace_sha256": sha256(trace),
        "perception_cache": str(cache),
        "perception_cache_index_sha256": cache_digest(cache),
        "annotations": str(annotations),
        "annotations_sha256": sha256(annotations),
        "gmt_checkpoint": str(checkpoint),
        "gmt_checkpoint_sha256": sha256(checkpoint),
        "horizon": int(args.horizon),
        "derived_horizons": sorted(
            {int(value) for value in (args.derive_horizons or (args.horizon,))}
            | {int(args.horizon)}
        ),
        "records": len(records),
        "max_events": args.max_events,
        "video_ids": args.video_ids,
        "max_events_per_video": args.max_events_per_video,
        "records_by_question": stats,
        "skipped_events": skipped,
        "uses_future_gt": True,
        "counterfactual_engine": ENGINE_VERSION,
        "counterfactual_engine_version": ENGINE_VERSION,
        "state_schema_version": STATE_SCHEMA_VERSION,
        "utility_definition": UTILITY_DEFINITION,
        "source_root": str(source_root),
        "source_commit": source_commit(source_root),
        "config": str(config_file) if config_file is not None else None,
        "config_sha256": sha256(config_file) if config_file is not None else None,
        "cache_sha256": cache_digest(cache),
        "association_backend": args.association_backend,
        "formal_gmt_association_adapter": args.association_backend == "formal_gmt_transformer",
        "prelock": prelock_diagnostic,
        "selection_authority": annotations.name == "train.json" and not prelock_diagnostic,
        "official_result_authority": annotations.name == "test.json" and not prelock_diagnostic,
        "official_test_generation_authorized": official_test_generation_authorized,
        "official_test_lock_sha256": official_test_lock_sha256,
        "official_test_selection_protocol_sha256": official_test_selection_protocol_sha256,
        "review_note": (
            "QUARANTINED_PRELOCK_TEST_DIAGNOSTIC; NOT_FOR_SELECTION; NOT_FOR_FINAL_REPORT"
            if prelock_diagnostic
            else (
            "formal GMT association-transformer replay over frozen perception and "
            "branch-local state"
            if args.association_backend == "formal_gmt_transformer"
            else "cosine_contract is a protocol backend; do not use this manifest as formal GMT causal evidence"
            )
        ),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
