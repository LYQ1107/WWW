"""Run the early pilot on one held-out sequence with mutable GMT replay.

The detector/ReID payloads are loaded from the immutable formal perception
cache.  The repository GMT association transformer is still called for each
frame/view, while the mutable association state, memory and typed actions are
committed online.  This is intentionally a screening replay on the TRAIN
video ``video_08``; it is not an official TEST result.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
from collections import defaultdict
from typing import Any, Dict, Mapping, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
PILOT = Path(
    os.environ.get("JEV_PILOT_ROOT", "/home/liuyeqiang/WWW_jev_full_h8_runtime/pilot")
)
VIDEO_ID = int(os.environ.get("JEV_VIDEO_ID", "8"))
CHECKPOINT = Path("/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth")
CONFIG = ROOT / "configs/VISION_test.yaml"
CACHE = Path("/data1/liuyeqiang/WWW/outputs/research_final_v2/off/perception_cache_train")
TRACE = Path(
    os.environ.get(
        "JEV_TRACE_PATH",
        "/home/liuyeqiang/WWW_jev_full_h8_runtime/partition/trace_by_video/video_08.jsonl",
    )
)
ANNOTATIONS = Path("/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json")
RECORDS = Path(
    os.environ.get("JEV_RECORDS_PATH", str(PILOT / "PILOT_TRACKING_TEST.jsonl"))
)
OFFLINE_ROOT = Path(
    os.environ.get("JEV_OFFLINE_ROOT", str(PILOT / "offline"))
)
SEED = 20261003
METHODS = {
    "gmt_off": None,
    "question_threshold": "offline/question_threshold/model.pth",
    "question_conditioned_mlp": "offline/question_conditioned_mlp/model.pth",
    "jev": "offline/jev/model.pth",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def scale_box(box: Sequence[float], image_size: Sequence[int], image: Mapping[str, Any]):
    model_h, model_w = float(image_size[0]), float(image_size[1])
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    if model_w > 0 and model_h > 0:
        x1 *= float(image["width"]) / model_w
        x2 *= float(image["width"]) / model_w
        y1 *= float(image["height"]) / model_h
        y2 *= float(image["height"]) / model_h
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def load_inputs():
    annotations = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    image_lookup = {}
    subset_images = []
    subset_ids = set()
    for image in annotations["images"]:
        if int(image.get("video_id", -1)) != VIDEO_ID:
            continue
        image_lookup[(VIDEO_ID, int(image["view_id"]), int(image["frame_id"]))] = image
        subset_images.append(image)
        subset_ids.add(int(image["id"]))
    if not subset_images:
        raise RuntimeError("video_08 has no annotation images")
    subset_annotation = {
        "images": subset_images,
        "annotations": [item for item in annotations["annotations"] if int(item["image_id"]) in subset_ids],
        "categories": annotations["categories"],
        "videos": annotations["videos"],
    }
    events = []
    with TRACE.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                event = json.loads(line)
                if int(event.get("context", {}).get("video_id", -1)) == VIDEO_ID:
                    events.append(event)
    events.sort(key=lambda event: int(event.get("context", {}).get("event_order", 0)))
    by_key: Dict[tuple[int, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        context = event.get("context", {})
        if context.get("decision_scope") not in {"match", "memory", "reactivation"}:
            continue
        key = (VIDEO_ID, int(context.get("frame", 0)), int(context.get("view", 0)))
        by_key[key].append(event)
    records = {}
    record_path = RECORDS
    with record_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            context = record["state"]["online_context"]
            if int(context.get("video_id", -1)) != VIDEO_ID:
                continue
            row = context.get("detection_index")
            if row is None:
                continue
            key = (
                VIDEO_ID,
                int(context["frame"]),
                int(context["view"]),
                str(record["question_type"]),
                int(row),
            )
            records[key] = record
    return annotations, subset_annotation, image_lookup, by_key, records


def load_runtime_modules():
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "third_party" / "CenterNet2"))
    sys.path.insert(0, str(ROOT / "reproduction_tools"))
    from build_jev_counterfactual_v2 import build_formal_gmt_engine
    from jev_counterfactual_v2 import MutableGMTState
    from gtr.modeling.jev_perception_cache import FrozenPerceptionCache
    from gtr.modeling.jev_runtime import JEVRuntimePolicy, build_controller_from_checkpoint
    from gtr.modeling.jev_state import (
        association_window_length,
        build_state_features,
        candidate_entropy,
        count_memory_observations,
        count_track_history,
        feature_names,
        legacy_acceptance_threshold,
    )

    return (
        build_formal_gmt_engine,
        MutableGMTState,
        FrozenPerceptionCache,
        JEVRuntimePolicy,
        build_controller_from_checkpoint,
        build_state_features,
        association_window_length,
        candidate_entropy,
        count_memory_observations,
        count_track_history,
        feature_names,
        legacy_acceptance_threshold,
    )


def choose(policy, feature, question, legal, off_action, context=None):
    return policy.decide(
        feature,
        question,
        legal,
        off_action=off_action,
        context=context,
    ).committed_action


def runtime_tracker_context(state) -> Dict[str, Any]:
    """Serialize only mutable online containers for a generated pilot trace."""

    return {
        "active_track_ids": sorted(int(value) for value in state.active_ids),
        "id_count": int(max(state.active_ids or {0})),
        "track_hits": {
            str(int(key)): int(value) for key, value in sorted(state.track_hits.items())
        },
        "memory_lengths": {
            str(int(key)): int(len(value)) for key, value in sorted(state.memory.items())
        },
        "memory_track_ids": sorted(int(value) for value in state.memory),
        "possible_memory_ids": sorted(
            int(value) for value in state.possible_memory_ids
        ),
        "stale_ids": sorted(int(value) for value in state.stale_ids),
        "old_reid_count": int(len(state.reactivation_bank)),
        "old_reid_ids": sorted(int(value) for value in state.reactivation_bank),
        "memory_bank_size": int(state.memory_bank_size),
        "trajectory_rng_policy": "branch_local_explicit_python_random_v1",
        "trajectory_rng_seed": state.trajectory_rng_seed,
        "trajectory_rng_calls": int(state.trajectory_rng_calls),
    }


def ordered_replay_keys(cache, video_id: int, view_num: int):
    """Return the exact production order and the production seed view.

    GMT initializes the first frame from the view with the most detections;
    ``argsort(...).reverse()`` makes the highest view index win ties.  The
    remaining first-frame views are then processed in ascending view order,
    followed by every later frame in ascending view order.
    """

    keys = [key for key in cache.keys() if int(key[0]) == int(video_id)]
    if not keys:
        raise RuntimeError(f"no cached perception keys for video {video_id}")
    frames = sorted({int(key[1]) for key in keys})
    first_keys = sorted(
        [key for key in keys if int(key[1]) == frames[0]],
        key=lambda key: int(key[2]),
    )
    if len(first_keys) != int(view_num):
        raise RuntimeError(
            f"expected {view_num} first-frame views, found {len(first_keys)}"
        )
    first_counts = {
        key: int(torch.as_tensor(cache.load(*key)["pred_boxes"]).shape[0])
        for key in first_keys
    }
    seed_key = max(first_keys, key=lambda key: (first_counts[key], int(key[2])))
    ordered = [seed_key]
    ordered.extend(key for key in first_keys if key != seed_key)
    ordered.extend(
        sorted(
            [key for key in keys if int(key[1]) != frames[0]],
            key=lambda key: (int(key[1]), int(key[2])),
        )
    )
    return ordered, seed_key, first_counts


def image_for(image_lookup, video_id: int, frame: int, view: int):
    """Resolve cache coordinates to the VisionTrack annotation image."""

    candidates = (
        (int(video_id), int(view) + 1, int(frame) + 1),
        (int(video_id), int(view), int(frame)),
        (int(video_id), int(view) + 1, int(frame)),
        (int(video_id), int(view), int(frame) + 1),
    )
    for key in candidates:
        if key in image_lookup:
            return image_lookup[key]
    raise KeyError(f"no annotation image for video={video_id}, frame={frame}, view={view}")


def append_predictions(predictions, payload, image, committed):
    boxes = payload["pred_boxes"]
    scores_tensor = payload["detection_scores"]
    for row in range(len(boxes)):
        predictions.append(
            {
                "image_id": int(image["id"]),
                "category_id": 1,
                "bbox": scale_box(boxes[row].tolist(), payload["image_size"], image),
                "score": float(scores_tensor[row].item()),
                "track_id": int(committed[row]),
            }
        )


def seed_production_state(cache, MutableGMTState, seed_key):
    """Seed the mutable replay exactly as ``sliding_inference_GMT`` does."""
    from jev_counterfactual_v2 import seed_production_state_from_payload

    return seed_production_state_from_payload(cache.load(*seed_key), seed_key)


def trace_action_for(events, question: str, row: int):
    for event in events:
        context = event.get("context", {})
        if str(event.get("question", "")).upper() != question:
            continue
        if context.get("detection_index") is None:
            continue
        if int(context["detection_index"]) != int(row):
            continue
        value = event.get("off_action") or event.get("proposed_action")
        return str(value) if value is not None else None
    return None


def reactivation_candidates(state, *, bank_size: int = 10):
    """Advance and return the native ``poss_ids``/``old_reids`` bank.

    Native GMT promotes a completed online memory bank into ``old_reids`` only
    when the ID leaves the current association window.  The old-reid proposal
    then persists until that ID is successfully reactivated.  Recomputing
    candidates from ``state.memory`` every frame loses this persistence and
    does not reproduce the native score matrix.
    """

    recent_ids = set()
    for item in state.association_history:
        for value in dict(item.get("assignments", {})).values():
            recent_ids.add(int(value))
    state.memory_bank_size = max(1, int(bank_size))
    for track_id, values in state.memory.items():
        track_id = int(track_id)
        if (
            len(values) >= state.memory_bank_size
            and track_id not in state.reactivation_bank
        ):
            state.possible_memory_ids.add(track_id)

    for track_id in sorted(tuple(state.possible_memory_ids)):
        track_id = int(track_id)
        if track_id in recent_ids:
            continue
        values = state.memory.get(track_id, ())
        if len(values) < state.memory_bank_size:
            continue
        recent_values = values[-state.memory_bank_size :]
        state.reactivation_bank[track_id] = torch.stack(
            [torch.as_tensor(value, dtype=torch.float32) for value in recent_values],
            dim=0,
        ).mean(dim=0).detach().cpu().clone()
        state.possible_memory_ids.discard(track_id)

    if os.environ.get("JEV_DEBUG_REACTIVATION") and state.association_history:
        latest = state.association_history[-1]["perception"]
        debug_frame = int(latest.get("frame", -1))
        if 200 <= debug_frame <= 260:
            print(
                json.dumps(
                    {
                        "debug": "reactivation_bank",
                        "frame": debug_frame,
                        "view": int(latest.get("view", -1)),
                        "recent_count": len(recent_ids),
                        "active_ids": sorted(int(value) for value in state.active_ids),
                        "possible_memory_ids": sorted(
                            int(value) for value in state.possible_memory_ids
                        ),
                        "old_reid_ids": sorted(
                            int(value) for value in state.reactivation_bank
                        ),
                        "memory_lengths": {
                            str(int(track_id)): int(len(values))
                            for track_id, values in sorted(state.memory.items())
                            if int(track_id) in {3, 36, 59, 150, 162, 234}
                        },
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    return sorted(int(track_id) for track_id in state.reactivation_bank), recent_ids


def build_reactivation_proposal(engine, payload, state, candidate_ids, proposal):
    """Run the formal GMT stale-bank association on an isolated state clone."""

    if not candidate_ids:
        return None
    probe = state.clone()
    if proposal is not None and proposal.rng_state_after is not None:
        probe.trajectory_rng_state = proposal.rng_state_after
    probe.active_ids = set(int(value) for value in candidate_ids)
    probe.reactivation_bank = {
        int(track_id): state.reactivation_bank[int(track_id)].detach().cpu().clone()
        for track_id in candidate_ids
    }
    probe.reactivation_mode = True
    return engine.propose(payload, probe)


def run_method(
    name: str,
    checkpoint_path: Path | None,
    *,
    image_lookup,
    by_key,
    records,
    build_formal_gmt_engine,
    MutableGMTState,
    FrozenPerceptionCache,
    JEVRuntimePolicy,
    build_controller_from_checkpoint,
    build_state_features,
    association_window_length,
    candidate_entropy,
    count_memory_observations,
    count_track_history,
    legacy_acceptance_threshold,
    feature_source_mode,
    parity_report,
    device="cpu",
    max_frame=None,
    trace_writer=None,
):
    controller = (
        build_controller_from_checkpoint(checkpoint_path, device="cpu")
        if checkpoint_path is not None
        else None
    )
    policy = (
        JEVRuntimePolicy("jev", controller, trace_writer)
        if controller is not None
        else JEVRuntimePolicy("off", trace_writer=trace_writer)
    )
    engine = build_formal_gmt_engine(
        config_file=CONFIG,
        checkpoint=CHECKPOINT,
        device=device,
        view_num=2,
        history_limit=80,
    )
    # DetectionCheckpointer can restore CUDA-indexed embedding buffers even
    # when the model was initially constructed with ``device=cpu``.  This is
    # local to the pilot CPU replay; formal GPU workers are separate
    # processes and are not changed.
    engine.association_fn.model.to(torch.device(device))
    engine.association_fn.model.eval()
    cache = FrozenPerceptionCache(CACHE)
    keys, seed_key, first_counts = ordered_replay_keys(cache, VIDEO_ID, view_num=2)
    if max_frame is not None:
        keys = [key for key in keys if int(key[1]) <= int(max_frame)]
        if not keys or keys[0] != seed_key:
            raise RuntimeError("max_frame smoke limit removed the production seed")
    state, seed_payload = seed_production_state(cache, MutableGMTState, seed_key)
    predictions = []
    decisions = []
    seed_video, seed_frame, seed_view = [int(value) for value in seed_key]
    append_predictions(
        predictions,
        seed_payload,
        image_for(image_lookup, seed_video, seed_frame, seed_view),
        {row: row + 1 for row in range(len(seed_payload["pred_boxes"]))},
    )
    counts = {
        "MATCH_DECISION": 0,
        "MEMORY_DECISION": 0,
        "REACTIVATION_DECISION": 0,
        "ACCEPT_CURRENT": 0,
        "REASSOCIATE": 0,
        "START_NEW": 0,
        "WRITE_MEMORY": 0,
        "SKIP_MEMORY": 0,
        "REACTIVATE_OLD": 0,
        "wrong_commit": 0,
        "known_commit_evaluations": 0,
        "canonical_feature_records": 0,
        "fallback_feature_records": 0,
        "missing_action_outcomes": 0,
        "memory_contamination": 0.0,
        "unsupported_reactivation": 0,
        "false_reactivation": 0.0,
        "reactivation_wrong_commit": 0,
        "reactivation_candidate_evaluations": 0,
        "reactivation_no_candidate": 0,
        "reactivation_action_counts": {
            "REACTIVATE_OLD": 0,
            "START_NEW": 0,
        },
        "off_action_mismatches": 0,
        "trace_action_records": 0,
    }
    runtime_model = engine.association_fn.model
    threshold = float(runtime_model.overlap_thresh)
    can_reassociate = int(runtime_model.jev_max_reassociate) > 0
    memory_enabled = bool(runtime_model.with_bank)
    state.memory_bank_size = int(getattr(runtime_model, "bank_size", 10))
    with_iou = bool(runtime_model.with_iou)
    not_mult_thresh = bool(runtime_model.not_mult_thresh)
    counts["runtime_feature_records"] = 0
    counts["trace_debug_feature_records"] = 0
    counts["feature_parity_records"] = 0
    counts["feature_parity_max_abs_error"] = 0.0

    for key_index, key in enumerate(keys[1:], start=1):
        payload = cache.load(*key)
        video_id, frame, view = [int(value) for value in key]
        required_history = min(frame, 39) * 2 + view
        if required_history < 1:
            required_history = 1
        if len(state.association_history) < required_history:
            raise RuntimeError(
                f"replay history underflow at frame={frame} view={view}: "
                f"{len(state.association_history)} < {required_history}"
            )
        if len(state.association_history) > required_history:
            state.association_history = state.association_history[-required_history:]
        proposal = engine.propose(payload, state)
        track_ids, scores = proposal.track_ids, proposal.scores
        if os.environ.get("JEV_DEBUG_PROPOSALS") and frame <= 6:
            print(
                json.dumps(
                    {
                        "debug": "proposal",
                        "key": [video_id, frame, view],
                        "history": [
                            [
                                int(item["perception"]["frame"]),
                                int(item["perception"]["view"]),
                            ]
                            for item in state.association_history
                        ],
                        "pairs": {str(row): int(col) for row, col in proposal.pairs.items()},
                        "track_ids": [int(value) for value in track_ids],
                        "scores": scores.tolist(),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        actions: Dict[int, str] = {}
        memories: Dict[int, str] = {}
        events_here = by_key.get((video_id, frame, view), ())
        first_frame_secondary_view = frame == seed_frame and view != seed_view
        window_length = association_window_length(
            history_instances=len(state.association_history),
            view_num=2,
            view_index=view,
            first_frame_secondary_view=first_frame_secondary_view,
        )

        def select_feature(record, runtime_feature):
            if record is not None:
                values = record.get("state", {}).get("feature_vector")
                if isinstance(values, list) and len(values) == int(runtime_feature.numel()):
                    trace_feature = torch.tensor(values, dtype=torch.float32)
                    if torch.isfinite(trace_feature).all():
                        error_vector = (trace_feature - runtime_feature.cpu()).abs()
                        error = float(error_vector.max().item())
                        counts["feature_parity_records"] += 1
                        counts["feature_parity_max_abs_error"] = max(
                            float(counts["feature_parity_max_abs_error"]), error
                        )
                        if name == "gmt_off":
                            parity_report["compared_records"] += 1
                            parity_report["max_abs_error"] = max(
                                float(parity_report["max_abs_error"]), error
                            )
                            parity_report["sum_abs_error"] += float(error_vector.sum().item())
                            parity_report["finite_runtime_records"] += int(torch.isfinite(runtime_feature).all())
                            parity_report["seen_record_keys"].add(
                                (VIDEO_ID, frame, view, record["question_type"], int(record["state"]["online_context"]["detection_index"]))
                            )
                            for index, value in enumerate(error_vector.tolist()):
                                parity_report["per_feature_max_abs_error"][index] = max(
                                    float(parity_report["per_feature_max_abs_error"][index]),
                                    float(value),
                                )
                                parity_report["per_feature_sum_abs_error"][index] += float(value)
                                parity_report["per_feature_count"][index] += 1
                        if feature_source_mode == "trace_debug":
                            counts["trace_debug_feature_records"] += 1
                            return trace_feature
                        if parity_report.get("collect_error_values"):
                            parity_report.setdefault("error_values", []).extend(
                                float(value) for value in error_vector.tolist()
                            )
                            per_feature_values = parity_report.setdefault(
                                "per_feature_error_values", [[] for _ in range(64)]
                            )
                            for index, value in enumerate(error_vector.tolist()):
                                per_feature_values[index].append(float(value))
            counts["runtime_feature_records"] += 1
            return runtime_feature

        # Phase 1: MATCH decisions use the same feature-value semantics as
        # production GTRRCNN, but are computed from the current mutable state.
        for row in range(int(scores.shape[0])):
            col = proposal.pairs.get(row)
            if col is None:
                legal = ["START_NEW"]
                off_action = "START_NEW"
                accept_score = reassociate_score = 0.0
                track_id = None
                candidate_entropy_value = 0.0
                track_length = 1
                score_variance = 0.0
                candidate_count = 0
            else:
                order = torch.argsort(scores[row], descending=True).tolist()
                first_col = int(col)
                second_col = next((int(item) for item in order if int(item) != first_col), None)
                first_id = int(track_ids[first_col])
                accept_score = float(scores[row, first_col].item())
                reassociate_score = float(scores[row, second_col].item()) if second_col is not None else 0.0
                candidate_entropy_value = candidate_entropy(scores[row])
                track_length = count_track_history(state.association_history, first_id)
                legacy_threshold = legacy_acceptance_threshold(
                    threshold, track_length, not_mult_thresh
                )
                off_action = "ACCEPT_CURRENT" if accept_score > legacy_threshold else "START_NEW"
                legal = ["ACCEPT_CURRENT", "START_NEW"]
                if can_reassociate and second_col is not None:
                    legal.insert(1, "REASSOCIATE")
                track_id = first_id
                score_variance = float(scores[row].var().item()) if scores.shape[1] > 1 else 0.0
                candidate_count = len(track_ids)

            runtime_feature = build_state_features(
                state_dim=64,
                accept_score=accept_score,
                reassociate_score=reassociate_score,
                threshold=threshold,
                candidate_count=candidate_count,
                candidate_entropy=candidate_entropy_value,
                track_count=len(track_ids),
                track_age=0,
                frame_index=frame,
                window_length=window_length,
                view_index=view,
                can_reassociate=can_reassociate,
                memory_enabled=memory_enabled,
                with_iou=with_iou,
                not_mult_thresh=not_mult_thresh,
                current_is_unmatched=(off_action == "START_NEW"),
                memory_count=0,
                track_score=accept_score,
                track_length=track_length,
                score_variance=score_variance,
            )
            record = records.get((VIDEO_ID, frame, view, "MATCH_DECISION", row))
            feature = select_feature(record, runtime_feature)
            match_context = {
                "video_id": VIDEO_ID,
                "view_num": 2,
                "frame": frame,
                "view": view,
                "decision_scope": "match",
                "detection_index": row,
                "model_image_size": [int(payload["image_size"][0]), int(payload["image_size"][1])],
                "bbox_xyxy": [float(value) for value in payload["pred_boxes"][row].tolist()],
                "detection_score": float(payload["detection_scores"][row].item()),
                "tracker_state_before": runtime_tracker_context(state),
                "association_rng_mode": "branch_local_explicit_python_random_v1",
                "association_rng_seed": state.trajectory_rng_seed,
                "association_rng_calls_before": state.trajectory_rng_calls,
                "association_rng_mapping_digest": proposal.trajectory_slot_mapping_digest,
                "association_transformer_calls": proposal.transformer_calls,
            }
            if col is not None:
                match_context.update(
                    {
                        "candidate_track_ids": [int(track_ids[index]) for index in order],
                        "candidate_scores": [float(scores[row, index].item()) for index in order],
                        "proposal_track_id": int(first_id),
                        "alternate_track_id": (
                            int(track_ids[second_col]) if second_col is not None else None
                        ),
                    }
                )
            else:
                match_context.update(
                    {
                        "candidate_track_ids": [],
                        "candidate_scores": [],
                        "proposal_track_id": None,
                        "alternate_track_id": None,
                    }
                )
            action = choose(
                policy,
                feature,
                "MATCH_DECISION",
                legal,
                off_action,
                context=match_context,
            )
            if action not in legal:
                action = off_action
            trace_action = trace_action_for(events_here, "MATCH_DECISION", row)
            if name == "gmt_off" and trace_action is not None:
                counts["trace_action_records"] += 1
                if action != trace_action:
                    counts["off_action_mismatches"] += 1
            actions[row] = action
            counts["MATCH_DECISION"] += 1
            counts[action] += 1
            if record is not None:
                counts["known_commit_evaluations"] += 1
                if action not in record["best_actions"]:
                    counts["wrong_commit"] += 1
                outcome = record["action_outcomes"].get(action)
                if outcome is None:
                    counts["missing_action_outcomes"] += 1
                else:
                    counts["memory_contamination"] += float(outcome.get("memory_contamination", 0.0))
            decisions.append({"frame": frame, "view": view, "row": row, "question": "MATCH_DECISION", "action": action})

        # Resolve MATCH without mutating state so MEMORY sees the final
        # existing committed identity. START_NEW has no same-step memory gate.
        resolution = engine.resolve_actions(
            payload,
            state,
            actions=actions,
            proposal=proposal,
        )
        final_existing = resolution["existing_track_ids"]

        # Phase 2: MEMORY decisions are computed only for rows committed to an
        # existing identity, matching production GTRRCNN commit order.
        for row in range(int(scores.shape[0])):
            track_id = final_existing.get(row)
            if track_id is None:
                continue
            score = float(scores[row].max().item()) if scores.shape[1] else 0.0
            memory_count = count_memory_observations(state.memory, int(track_id))
            legal = ["WRITE_MEMORY", "SKIP_MEMORY"]
            runtime_feature = build_state_features(
                state_dim=64,
                accept_score=score,
                reassociate_score=0.0,
                threshold=threshold,
                candidate_count=1,
                candidate_entropy=0.0,
                track_count=len(track_ids),
                track_age=memory_count,
                frame_index=frame,
                window_length=window_length,
                view_index=view,
                can_reassociate=can_reassociate,
                memory_enabled=memory_enabled,
                with_iou=with_iou,
                not_mult_thresh=not_mult_thresh,
                memory_count=memory_count,
                track_score=score,
                track_length=max(1, memory_count),
                score_variance=0.0,
            )
            record = records.get((VIDEO_ID, frame, view, "MEMORY_DECISION", row))
            feature = select_feature(record, runtime_feature)
            off_action = "WRITE_MEMORY"
            memory_context = {
                "video_id": VIDEO_ID,
                "view_num": 2,
                "frame": frame,
                "view": view,
                "decision_scope": "memory",
                "detection_index": row,
                "track_id": int(track_id),
                "model_image_size": [int(payload["image_size"][0]), int(payload["image_size"][1])],
                "bbox_xyxy": [float(value) for value in payload["pred_boxes"][row].tolist()],
                "tracker_state_before": runtime_tracker_context(state),
                "association_rng_mode": "branch_local_explicit_python_random_v1",
                "association_rng_seed": state.trajectory_rng_seed,
                "association_rng_calls_before": state.trajectory_rng_calls,
                "association_rng_mapping_digest": proposal.trajectory_slot_mapping_digest,
                "association_transformer_calls": proposal.transformer_calls,
            }
            action = choose(
                policy,
                feature,
                "MEMORY_DECISION",
                legal,
                off_action,
                context=memory_context,
            )
            if action not in legal:
                action = off_action
            trace_action = trace_action_for(events_here, "MEMORY_DECISION", row)
            if name == "gmt_off" and trace_action is not None:
                counts["trace_action_records"] += 1
                if action != trace_action:
                    counts["off_action_mismatches"] += 1
            memories[row] = action
            counts["MEMORY_DECISION"] += 1
            counts[action] += 1
            if record is not None:
                counts["known_commit_evaluations"] += 1
                if action not in record["best_actions"]:
                    counts["wrong_commit"] += 1
                outcome = record["action_outcomes"].get(action)
                if outcome is None:
                    counts["missing_action_outcomes"] += 1
                else:
                    counts["memory_contamination"] += float(outcome.get("memory_contamination", 0.0))
            decisions.append({"frame": frame, "view": view, "row": row, "question": "MEMORY_DECISION", "action": action})

        # Phase 3: reproduce the native memory-bank reactivation boundary.
        # The controller sees only a stale online memory proposal; labels and
        # future annotations remain outside this path.
        reactivation_assignments = {}
        reactivation_proposal = None
        bank_size = int(getattr(runtime_model, "bank_size", 10))
        bank_threshold = float(getattr(runtime_model, "thred_bank", 0.4))
        stale_ids, recent_ids = reactivation_candidates(state, bank_size=bank_size)
        if stale_ids:
            state.stale_ids.update(stale_ids)
            reactivation_proposal = build_reactivation_proposal(
                engine, payload, state, stale_ids, proposal
            )
        for row in range(int(scores.shape[0])):
            if final_existing.get(row) is not None:
                continue
            if reactivation_proposal is None:
                counts["reactivation_no_candidate"] += 1
                continue
            col = reactivation_proposal.pairs.get(row)
            if col is None:
                counts["reactivation_no_candidate"] += 1
                continue
            stale_id = int(reactivation_proposal.track_ids[col])
            score = float(reactivation_proposal.scores[row, col].item())
            memory_count = max(1, len(state.memory.get(stale_id, ())) + 1)
            off_action = (
                "REACTIVATE_OLD"
                if score > bank_threshold
                else "START_NEW"
            )
            legal = ["REACTIVATE_OLD", "START_NEW"]
            runtime_feature = build_state_features(
                state_dim=64,
                accept_score=0.0,
                reassociate_score=score,
                threshold=bank_threshold,
                candidate_count=1,
                candidate_entropy=0.0,
                track_count=len(recent_ids),
                track_age=memory_count,
                frame_index=frame,
                window_length=window_length,
                view_index=view,
                can_reassociate=can_reassociate,
                memory_enabled=memory_enabled,
                with_iou=with_iou,
                not_mult_thresh=not_mult_thresh,
                has_old_track=True,
                current_is_unmatched=True,
                memory_count=memory_count,
                track_score=score,
                track_length=max(1, memory_count),
                score_variance=0.0,
            )
            record = records.get((VIDEO_ID, frame, view, "REACTIVATION_DECISION", row))
            feature = select_feature(record, runtime_feature)
            reactivation_context = {
                "video_id": VIDEO_ID,
                "view_num": 2,
                "frame": frame,
                "view": view,
                "decision_scope": "reactivation",
                "detection_index": row,
                "track_id": stale_id,
                "candidate_track_ids": [int(value) for value in reactivation_proposal.track_ids],
                "candidate_scores": [
                    float(reactivation_proposal.scores[row, index].item())
                    for index in range(len(reactivation_proposal.track_ids))
                ],
                "reactivation_score": score,
                "bank_threshold": bank_threshold,
                "tracker_state_before": runtime_tracker_context(state),
            }
            action = choose(
                policy,
                feature,
                "REACTIVATION_DECISION",
                legal,
                off_action,
                context=reactivation_context,
            )
            if action not in legal:
                action = off_action
            trace_action = trace_action_for(events_here, "REACTIVATION_DECISION", row)
            if name == "gmt_off" and trace_action is not None:
                counts["trace_action_records"] += 1
                if action != trace_action:
                    counts["off_action_mismatches"] += 1
            counts["REACTIVATION_DECISION"] += 1
            counts["reactivation_candidate_evaluations"] += 1
            counts["reactivation_action_counts"][action] = (
                counts["reactivation_action_counts"].get(action, 0) + 1
            )
            counts[action] += 1
            if action == "REACTIVATE_OLD":
                reactivation_assignments[row] = stale_id
            if record is not None:
                counts["known_commit_evaluations"] += 1
                if action not in record["best_actions"]:
                    counts["wrong_commit"] += 1
                    counts["reactivation_wrong_commit"] += 1
                outcome = record["action_outcomes"].get(action)
                if outcome is None:
                    counts["missing_action_outcomes"] += 1
                else:
                    counts["memory_contamination"] += float(outcome.get("memory_contamination", 0.0))
                    counts["false_reactivation"] += float(outcome.get("false_reactivation", 0.0))
            decisions.append(
                {
                    "frame": frame,
                    "view": view,
                    "row": row,
                    "question": "REACTIVATION_DECISION",
                    "action": action,
                    "track_id": stale_id,
                    "score": score,
                    "off_action": off_action,
                    "legal_actions": list(legal),
                    "candidate_track_ids": [
                        int(value) for value in reactivation_proposal.track_ids
                    ],
                    "candidate_scores": [
                        float(reactivation_proposal.scores[row, index].item())
                        for index in range(len(reactivation_proposal.track_ids))
                    ],
                    "candidate_order": "sorted_mutable_stale_ids",
                    "candidate_count": len(reactivation_proposal.track_ids),
                    "bank_threshold": bank_threshold,
                }
            )
        result = engine.step(
            payload,
            state,
            actions=actions,
            memory_actions=memories,
            proposal=proposal,
            reactivation_assignments=reactivation_assignments,
            reactivation_proposal=reactivation_proposal,
        )
        if frame == seed_frame and view != seed_view:
            # The native first-frame path processes the seed view first but
            # stores the completed frame in natural view order before the next
            # frame's global window is built.  Keep that order in the mutable
            # replay history; otherwise the transformer sees view1,view0
            # instead of view0,view1 and the OFF branch diverges later.
            state.association_history.sort(
                key=lambda item: int(item["perception"]["view"])
            )
        append_predictions(
            predictions,
            payload,
            image_for(image_lookup, video_id, frame, view),
            result["committed_track_ids"],
        )
        if key_index and key_index % 100 == 0:
            print(json.dumps({"method": name, "payloads": key_index, "predictions": len(predictions)}), flush=True)
    action_total = sum(counts[name] for name in ("MATCH_DECISION", "MEMORY_DECISION", "REACTIVATION_DECISION"))
    counts["wrong_commit_rate"] = counts["wrong_commit"] / max(1, counts["known_commit_evaluations"])
    counts["action_rates"] = {
        action: counts[action] / max(1, action_total)
        for action in (
            "ACCEPT_CURRENT",
            "REASSOCIATE",
            "START_NEW",
            "WRITE_MEMORY",
            "SKIP_MEMORY",
            "REACTIVATE_OLD",
        )
    }
    out_predictions = PILOT / "tracking_predictions" / f"{name}.json"
    out_decisions = PILOT / "tracking_decisions" / f"{name}.json"
    out_predictions.parent.mkdir(parents=True, exist_ok=True)
    out_decisions.parent.mkdir(parents=True, exist_ok=True)
    out_predictions.write_text(json.dumps(predictions), encoding="utf-8")
    out_decisions.write_text(json.dumps(decisions), encoding="utf-8")
    return {
        "method": name,
        "status": "PASS",
        "predictions": str(out_predictions),
        "decisions": str(out_decisions),
        "payloads": len(keys),
        "prediction_rows": len(predictions),
        "counts": counts,
    }


def prepare_eval_dataset(subset_annotation: Mapping[str, Any]) -> Path:
    root = PILOT / "eval_dataset"
    root.mkdir(parents=True, exist_ok=True)
    annotations = root / "annotations" / "train.json"
    annotations.parent.mkdir(parents=True, exist_ok=True)
    if not annotations.exists():
        annotations.write_text(json.dumps(subset_annotation), encoding="utf-8")
    train_root = root / "train"
    train_root.mkdir(exist_ok=True)
    # The annotation video name (for example ``00021gate``) is the scene
    # name, while TrackEval expects one directory per camera view (for
    # example ``00021gate_View1`` and ``00021gate_View2``).  Link exactly the
    # view sequences present in this held-out annotation subset.
    sequences = sorted(
        {
            str(image["file_name"]).split("/", 1)[0]
            for image in subset_annotation["images"]
        }
    )
    for sequence in sequences:
        destination = train_root / sequence
        source = Path("/data/DATASETS/TRACKING/JDE/VisionTrack/train") / sequence
        if not source.is_dir():
            raise FileNotFoundError(source)
        if not destination.exists():
            destination.symlink_to(source, target_is_directory=True)
    return root


def run_eval(method: str, prediction: Path, dataset_root: Path) -> tuple[Path, Path]:
    # Keep prior failed/fixed screening evaluations immutable.  Each rerun
    # gets a new semantic tag so an old prepared directory can never be
    # mistaken for the current result.
    prepared = PILOT / "tracking_eval_runtime_state" / method / "prepared"
    evaluated = PILOT / "tracking_eval_runtime_state" / method / "evaluation"
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join([str(ROOT), str(ROOT / "third_party/CenterNet2"), str(ROOT / "reproduction_tools"), env.get("PYTHONPATH", "")])
    commands = [
        [sys.executable, str(ROOT / "reproduction_tools/prepare_visiontrack_predictions.py"), "--predictions", str(prediction), "--dataset", str(dataset_root), "--split", "train", "--output", str(prepared)],
        [sys.executable, str(ROOT / "reproduction_tools/evaluate_visiontrack.py"), "--prepared", str(prepared), "--output", str(evaluated), "--allow-duplicate-gt"],
    ]
    for command in commands:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    return prepared, evaluated


def scalar(value: Any) -> float:
    if isinstance(value, list):
        return sum(float(item) for item in value) / len(value) if value else 0.0
    return float(value)


def extract_metrics(evaluated: Path) -> Dict[str, float]:
    combined = json.loads((evaluated / "metrics.json").read_text(encoding="utf-8"))["combined_metrics"]
    return {
        "HOTA": scalar(combined["HOTA"]["HOTA"]) * 100.0,
        "DetA": scalar(combined["HOTA"]["DetA"]) * 100.0,
        "AssA": scalar(combined["HOTA"]["AssA"]) * 100.0,
        "IDF1": scalar(combined["Identity"]["IDF1"]) * 100.0,
        "MOTA": scalar(combined["CLEAR"]["MOTA"]) * 100.0,
        "IDSW": scalar(combined["CLEAR"]["IDSW"]),
        "Frag": scalar(combined["CLEAR"]["Frag"]),
    }


def finalize_feature_parity(
    parity_report: Dict[str, Any],
    records: Mapping[Any, Any],
    off_action_counts: Mapping[str, Any],
) -> Dict[str, Any]:
    """Finalize and serialize-freeze the OFF runtime parity accounting."""

    expected_keys = set(records)
    missing_keys = sorted(expected_keys - parity_report["seen_record_keys"])
    per_feature = []
    for index, feature_name in enumerate(parity_report["feature_names"]):
        count = int(parity_report["per_feature_count"][index])
        per_feature.append(
            {
                "index": index,
                "name": feature_name,
                "count": count,
                "max_abs_error": float(parity_report["per_feature_max_abs_error"][index]),
                "mean_abs_error": (
                    float(parity_report["per_feature_sum_abs_error"][index]) / count
                    if count else None
                ),
            }
        )
    parity_report["missing_record_count"] = len(missing_keys)
    parity_report["missing_record_keys_sample"] = [list(key) for key in missing_keys[:20]]
    parity_report["mean_abs_error"] = (
        float(parity_report["sum_abs_error"])
        / max(1, int(parity_report["compared_records"]) * len(parity_report["feature_names"]))
    )
    parity_report["per_feature"] = per_feature
    parity_report["off_action_mismatches"] = int(off_action_counts["off_action_mismatches"])
    parity_report["pass"] = bool(
        parity_report["expected_record_count"] > 0
        and parity_report["expected_record_count"] == parity_report["compared_records"]
        and parity_report["missing_record_count"] == 0
        and parity_report["finite_runtime_records"] == parity_report["compared_records"]
        and parity_report["max_abs_error"] <= parity_report["tolerance"]
        and parity_report["off_action_mismatches"] == 0
    )
    parity_report["status"] = "PASS" if parity_report["pass"] else "FAIL"
    parity_report.pop("seen_record_keys", None)
    return parity_report


def reactivation_coverage_report(
    records: Mapping[Any, Any],
    counts: Mapping[str, Any],
) -> Dict[str, Any]:
    expected = sum(
        str(record.get("question_type")) == "REACTIVATION_DECISION"
        for record in records.values()
    )
    action_counts = counts.get("reactivation_action_counts", {})
    result = {
        "expected_record_count": int(expected),
        "runtime_decision_count": int(counts.get("REACTIVATION_DECISION", 0)),
        "REACTIVATE_OLD": int(action_counts.get("REACTIVATE_OLD", 0)),
        "START_NEW": int(action_counts.get("START_NEW", 0)),
        "reactivation_candidate_evaluations": int(
            counts.get("reactivation_candidate_evaluations", 0)
        ),
        "reactivation_no_candidate": int(counts.get("reactivation_no_candidate", 0)),
        "reactivation_wrong_commit": int(counts.get("reactivation_wrong_commit", 0)),
        "false_reactivation": float(counts.get("false_reactivation", 0.0)),
        "unsupported_reactivation": int(counts.get("unsupported_reactivation", 0)),
    }
    result["pass"] = bool(
        result["expected_record_count"] > 0
        and result["runtime_decision_count"] > 0
        and result["unsupported_reactivation"] == 0
    )
    result["status"] = "PASS" if result["pass"] else "FAIL_RUNTIME_REACTIVATION_COVERAGE"
    return result


def blocked_runtime_report(
    raw: Mapping[str, Any],
    parity_report: Mapping[str, Any],
    reactivation_gate: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Describe a fail-closed runtime pilot without inventing metrics."""

    off = raw["methods"].get("gmt_off", {})
    report = {
        "status": "BLOCKED",
        "classification": "SCREENING_ONLY_NOT_FOR_FINAL_SELECTION_NOT_FOR_PAPER_RESULT",
        "created_utc": raw["created_utc"],
        "video_id": raw["video_id"],
        "sequence": raw["sequence"],
        "checkpoint": raw["checkpoint"],
        "config": raw["config"],
        "perception_cache": raw["perception_cache"],
        "association_backend": raw["association_backend"],
        "controller_feature_source": raw["controller_feature_source"],
        "official_test_read": False,
        "methods": {"gmt_off": off},
        "runtime_feature_parity_gate": {
            "required_for_runtime_claim": True,
            "report": str(ROOT / "reports" / "JEV_RUNTIME_STATE_V3" / "STATE_FEATURE_PARITY.json"),
            "compared_records": parity_report["compared_records"],
            "expected_records": parity_report["expected_record_count"],
            "missing_records": parity_report["missing_record_count"],
            "max_abs_error": parity_report["max_abs_error"],
            "off_action_mismatches": parity_report["off_action_mismatches"],
            "pass": False,
        },
        "pilot_verdict": "PILOT_FAIL_RUNTIME_STATE_PARITY",
        "blocked_before_learned_controller": True,
        "interpretation": (
            "No learned-controller tracking metrics were produced because the "
            "frozen OFF trace could not be reproduced from the mutated runtime state."
        ),
    }
    if reactivation_gate is not None:
        report["reactivation_coverage_gate"] = dict(reactivation_gate)
        if not bool(reactivation_gate.get("pass")):
            report["pilot_verdict"] = "PILOT_FAIL_RUNTIME_REACTIVATION_COVERAGE"
            report["interpretation"] = (
                "The v4 dataset contains reactivation decisions but the online "
                "runtime did not cover them with valid mutable-state semantics."
            )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default=os.environ.get("JEV_PILOT_DEVICE", "cpu"),
        help="formal replay device; use a visible CUDA device for exact GPU parity",
    )
    parser.add_argument("--feature-source", choices=("runtime", "trace_debug"), default="runtime")
    parser.add_argument(
        "--allow-runtime-with-legacy-trace",
        action="store_true",
        help=(
            "run the mutated-state screening pilot even when the preserved legacy "
            "trace cannot pass strict feature parity; never treats that parity as PASS"
        ),
    )
    parser.add_argument(
        "--methods",
        default="all",
        help="comma-separated subset for CPU smoke/debug runs; default is all methods",
    )
    parser.add_argument("--max-frame", type=int, help="stop after this cache frame; smoke/debug only")
    args = parser.parse_args()
    for path in (CHECKPOINT, CONFIG, CACHE, TRACE, ANNOTATIONS):
        if not path.exists():
            raise FileNotFoundError(path)
    annotations, subset_annotation, image_lookup, by_key, records = load_inputs()
    modules = load_runtime_modules()
    (
        build_formal_gmt_engine,
        MutableGMTState,
        FrozenPerceptionCache,
        JEVRuntimePolicy,
        build_controller_from_checkpoint,
        build_state_features,
        association_window_length,
        candidate_entropy,
        count_memory_observations,
        count_track_history,
        feature_names,
        legacy_acceptance_threshold,
    ) = modules
    parity_report = {
        "schema_version": "jev_runtime_state_contract_v3",
        "source_trace": str(TRACE),
        "tolerance": 1e-6,
        "feature_names": list(feature_names(64)),
        "expected_record_count": len(records),
        "compared_records": 0,
        "finite_runtime_records": 0,
        "max_abs_error": 0.0,
        "sum_abs_error": 0.0,
        "per_feature_max_abs_error": [0.0] * 64,
        "per_feature_sum_abs_error": [0.0] * 64,
        "per_feature_count": [0] * 64,
        "seen_record_keys": set(),
    }
    raw = {
        "status": "RUNNING",
        "classification": "SCREENING_ONLY_NOT_FOR_FINAL_SELECTION_NOT_FOR_PAPER_RESULT",
        "created_utc": utc_now(),
        "video_id": VIDEO_ID,
        "sequence": str(subset_annotation["videos"][VIDEO_ID - 1]["file_name"]),
        "checkpoint": str(CHECKPOINT),
        "config": str(CONFIG),
        "perception_cache": str(CACHE),
        "association_backend": "formal_gmt_transformer",
        "trajectory_rng_policy": "branch_local_explicit_python_random_v1",
        "trajectory_rng_master_seed": 20261006,
        "proposal_reused_across_legal_actions": True,
        "reassociate_reuses_score_matrix": True,
        "second_transformer_call_for_reassociate": False,
        "device": args.device,
        "controller_feature_source": args.feature_source,
        "allow_runtime_with_legacy_trace": bool(args.allow_runtime_with_legacy_trace),
        "methods": {},
    }
    selected_names = list(METHODS)
    if args.methods != "all":
        selected_names = [name.strip() for name in args.methods.split(",") if name.strip()]
        unknown = sorted(set(selected_names) - set(METHODS))
        if unknown:
            raise ValueError(f"unknown pilot methods: {unknown}")
        if args.max_frame is None:
            raise ValueError("--methods subsets require --max-frame smoke mode")
    selected_methods = {name: METHODS[name] for name in selected_names}
    reactivation_gate = None
    def execute_method(name: str) -> None:
        relative = selected_methods[name]
        checkpoint_path = (
            None
            if relative is None
            else OFFLINE_ROOT / name / "model.pth"
        )
        if checkpoint_path is not None and not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        print(json.dumps({"starting": name}), flush=True)
        raw["methods"][name] = run_method(
            name,
            checkpoint_path,
            image_lookup=image_lookup,
            by_key=by_key,
            records=records,
            build_formal_gmt_engine=build_formal_gmt_engine,
            MutableGMTState=MutableGMTState,
            FrozenPerceptionCache=FrozenPerceptionCache,
            JEVRuntimePolicy=JEVRuntimePolicy,
            build_controller_from_checkpoint=build_controller_from_checkpoint,
            build_state_features=build_state_features,
            association_window_length=association_window_length,
            candidate_entropy=candidate_entropy,
            count_memory_observations=count_memory_observations,
            count_track_history=count_track_history,
            legacy_acceptance_threshold=legacy_acceptance_threshold,
            feature_source_mode=args.feature_source,
            parity_report=parity_report,
            device=args.device,
            max_frame=args.max_frame,
        )
        json_write(PILOT / "PILOT_TRACKING_RAW.json", raw)

    # In runtime mode, the OFF replay is the only method allowed to run
    # before the parity gate.  Learned controllers must never get a chance to
    # mutate state when the frozen formal trace itself is not reproducible.
    if args.max_frame is None and args.feature_source == "runtime":
        if selected_names != list(METHODS) or "gmt_off" not in selected_methods:
            raise ValueError("full runtime mode requires the complete method set including gmt_off")
        execute_method("gmt_off")
        parity_report = finalize_feature_parity(
            parity_report,
            records,
            raw["methods"]["gmt_off"]["counts"],
        )
        reactivation_gate = reactivation_coverage_report(
            records, raw["methods"]["gmt_off"]["counts"]
        )
        raw["reactivation_coverage_gate"] = reactivation_gate
        json_write(ROOT / "reports" / "JEV_RUNTIME_STATE_V3" / "STATE_FEATURE_PARITY.json", parity_report)
        if (
            (not parity_report["pass"] or not reactivation_gate["pass"])
            and not args.allow_runtime_with_legacy_trace
        ):
            raw["status"] = "BLOCKED_RUNTIME_STATE_PARITY"
            raw["runtime_feature_parity_gate"] = parity_report
            json_write(PILOT / "PILOT_TRACKING_RAW.json", raw)
            report = blocked_runtime_report(raw, parity_report, reactivation_gate)
            json_write(
                ROOT / "reports" / "JEV_RUNTIME_STATE_V3" / "RUNTIME_STATE_TRACKING.json",
                report,
            )
            json_write(PILOT / "PILOT_TRACKING_THREE_WAY.json", report)
            print(json.dumps(report, indent=2))
            raise SystemExit("runtime pilot blocked by failed OFF state-feature parity gate")
        if not parity_report["pass"]:
            raw["runtime_feature_parity_override"] = (
                "SCREENING_ONLY: preserved legacy trace lacks strict RNG/state provenance; "
                "controllers still receive current mutated-state features"
            )
        for name in selected_names:
            if name != "gmt_off":
                execute_method(name)
    else:
        for name in selected_names:
            execute_method(name)
    if args.max_frame is not None:
        if "gmt_off" in raw["methods"]:
            parity_report = finalize_feature_parity(
                parity_report,
                records,
                raw["methods"]["gmt_off"]["counts"],
            )
            parity_report["scope"] = "bounded_smoke_probe"
            parity_report["max_frame"] = int(args.max_frame)
            reactivation_gate = reactivation_coverage_report(
                records, raw["methods"]["gmt_off"]["counts"]
            )
            raw["reactivation_coverage_gate"] = reactivation_gate
            json_write(
                ROOT / "reports" / "JEV_RUNTIME_STATE_V3" / "STATE_FEATURE_PARITY.json",
                parity_report,
            )
            raw["runtime_feature_parity_gate"] = parity_report
        raw["status"] = "SMOKE_ONLY"
        raw["max_frame"] = int(args.max_frame)
        json_write(PILOT / "PILOT_TRACKING_RAW_SMOKE.json", raw)
        print(json.dumps(raw, indent=2))
        return

    dataset_root = prepare_eval_dataset(subset_annotation)
    evaluations = {}
    for name in METHODS:
        prepared, evaluated = run_eval(name, Path(raw["methods"][name]["predictions"]), dataset_root)
        evaluations[name] = {
            "prepared": str(prepared),
            "evaluation": str(evaluated),
            "metrics": extract_metrics(evaluated),
        }
    baseline = evaluations["gmt_off"]["metrics"]
    methods = {}
    for name, item in raw["methods"].items():
        metrics = evaluations[name]["metrics"]
        delta = {
            key: float(metrics[key] - baseline[key])
            for key in ("HOTA", "AssA", "IDF1", "MOTA", "IDSW", "Frag")
        }
        methods[name] = {
            "metrics": metrics,
            "delta_vs_gmt_off": {
                "ΔHOTA": delta["HOTA"],
                "ΔAssA": delta["AssA"],
                "ΔIDF1": delta["IDF1"],
                "ΔMOTA": delta["MOTA"],
                "ΔIDSW": delta["IDSW"],
                "ΔFrag": delta["Frag"],
            },
            "action_counts": item["counts"],
            "evaluation": evaluations[name],
        }
    # Runtime mode was finalized and gated above.  Trace-debug mode reaches
    # here only to produce a clearly labelled diagnostic tracking report.
    if args.feature_source == "trace_debug":
        parity_report = finalize_feature_parity(
            parity_report,
            records,
            raw["methods"]["gmt_off"]["counts"],
        )
        json_write(ROOT / "reports" / "JEV_RUNTIME_STATE_V3" / "STATE_FEATURE_PARITY.json", parity_report)
    if reactivation_gate is None and "gmt_off" in raw["methods"]:
        reactivation_gate = reactivation_coverage_report(
            records, raw["methods"]["gmt_off"]["counts"]
        )
    json_write(ROOT / "reports" / "JEV_RUNTIME_STATE_V3" / "STATE_FEATURE_PARITY.json", parity_report)
    report = {
        "status": "PASS",
        "classification": "SCREENING_ONLY_NOT_FOR_FINAL_SELECTION_NOT_FOR_PAPER_RESULT",
        "created_utc": raw["created_utc"],
        "video_id": VIDEO_ID,
        "sequence": raw["sequence"],
        "checkpoint": raw["checkpoint"],
        "config": raw["config"],
        "perception_cache": raw["perception_cache"],
        "association_backend": raw["association_backend"],
        "closed_loop_definition": "frozen detector/ReID payloads + formal GMT association transformer + mutable branch-local association/memory state + online typed controller commits; runtime mode recomputes controller features from the mutated state through the canonical shared builder",
        "controller_feature_source": args.feature_source,
        "trace_feature_policy": "trace_debug is parity-only and must not be interpreted as a closed-loop scientific result",
        "official_test_read": False,
        "methods": methods,
        "gmt_off_baseline": baseline,
        "runtime_feature_parity_gate": {
            "required_for_runtime_claim": args.feature_source == "runtime",
            "report": str(ROOT / "reports" / "JEV_RUNTIME_STATE_V3" / "STATE_FEATURE_PARITY.json"),
            "compared_records": parity_report["compared_records"],
            "expected_records": parity_report["expected_record_count"],
            "missing_records": parity_report["missing_record_count"],
            "max_abs_error": parity_report["max_abs_error"],
            "off_action_mismatches": methods["gmt_off"]["action_counts"]["off_action_mismatches"],
            "pass": parity_report["pass"],
        },
        "reactivation_coverage_gate": reactivation_gate,
        "pilot_verdict": (
            "TRACE_DEBUG_ONLY"
            if args.feature_source == "trace_debug"
            else (
                "PILOT_SCREENING_RUNTIME_STATE_WITH_LEGACY_TRACE_PARITY_BLOCK"
                if not parity_report["pass"]
                else (
                "PILOT_FAIL_RUNTIME_REACTIVATION_COVERAGE"
                if not (reactivation_gate and reactivation_gate["pass"])
                else (
                    "PILOT_GO_FOR_FULL_H8_CONTINUATION"
                    if (
                        methods["jev"]["metrics"]["AssA"] >= baseline["AssA"]
                        and methods["jev"]["metrics"]["IDSW"] <= baseline["IDSW"]
                    )
                    else "PILOT_FAIL_RUNTIME_STATE_PARITY_OR_TRACKING"
                )
                )
            )
        ),
        "previous_manual_feature_result": "SUPERSEDED_INVALID_FEATURE_SCHEMA",
        "interpretation": "Pilot-only matched-sequence evidence; not a final paper result and not a substitute for full VISION_test evaluation.",
    }
    json_write(
        ROOT / "reports" / "JEV_RUNTIME_STATE_V3" / (
            "TRACE_DEBUG_TRACKING.json" if args.feature_source == "trace_debug" else "RUNTIME_STATE_TRACKING.json"
        ),
        report,
    )
    json_write(PILOT / "PILOT_TRACKING_THREE_WAY.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
