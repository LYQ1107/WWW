"""Validate a no-intervention v2 simulator replay against the real OFF trace.

This is the second OFF gate.  The trace-contract validator only checks that
the trace says OFF consistently; this validator actually replays frozen
perception through the formal GMT association adapter and compares proposal,
assignment, IDs, memory and state transitions at every available decision
boundary. Annotation JSON is used only to map cached detections to the real
OFF prediction image IDs; no GT/evaluator field is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from build_jev_counterfactual_dataset import iou, normalize_events
from build_jev_counterfactual_v2 import build_formal_gmt_engine, event_maps
from gtr.modeling.jev_perception_cache import FrozenPerceptionCache
from jev_counterfactual_v2 import CachedPerceptionMutableAssociationV2, MutableGMTState


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def load_image_index(path: Path):
    """Read only image geometry/index metadata; never parse GT annotations."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    images = {}
    image_meta = {}
    for image in payload.get("images", ()):
        image_id = int(image["id"])
        images[(int(image["video_id"]), int(image["view_id"]), int(image["frame_id"]))] = image_id
        image_meta[image_id] = image
    return images, image_meta


def _state_from_snapshot(snapshot: Mapping[str, Any], feature_dim: int) -> MutableGMTState:
    active = {int(value) for value in snapshot.get("active_track_ids", ())}
    hits = {int(key): int(value) for key, value in (snapshot.get("track_hits") or {}).items()}
    memory_lengths = {
        int(key): int(value) for key, value in (snapshot.get("memory_lengths") or {}).items()
    }
    zeros = torch.zeros((max(0, int(feature_dim)),), dtype=torch.float32)
    return MutableGMTState(
        next_id=int(snapshot.get("id_count", max(active or {0}))),
        active_ids=active,
        stale_ids={int(value) for value in snapshot.get("stale_ids", ())},
        track_embeddings={track_id: zeros.clone() for track_id in active},
        track_hits=hits,
        memory={track_id: [zeros.clone() for _ in range(length)] for track_id, length in memory_lengths.items()},
    )


def _state_projection(state: MutableGMTState) -> Mapping[str, Any]:
    return {
        "id_count": int(state.next_id),
        "active_track_ids": sorted(int(value) for value in state.active_ids),
        "track_hits": {str(key): int(value) for key, value in sorted(state.track_hits.items())},
        "memory_lengths": {str(key): len(value) for key, value in sorted(state.memory.items())},
        "memory_track_ids": sorted(int(value) for value in state.memory),
        "possible_memory_ids": [],
        "stale_ids": sorted(int(value) for value in state.stale_ids),
    }


def _state_mismatch(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> int:
    fields = (
        "id_count",
        "active_track_ids",
        "track_hits",
        "memory_lengths",
        "memory_track_ids",
        "possible_memory_ids",
        "stale_ids",
    )
    return sum(int(expected.get(field, [] if field.endswith("ids") else {}) != actual.get(field)) for field in fields)


def _expected_id(event: Mapping[str, Any], action: str):
    context = event.get("context") or {}
    if action == "ACCEPT_CURRENT":
        return context.get("proposal_track_id")
    if action == "REASSOCIATE":
        return context.get("alternate_track_id")
    if action == "REACTIVATE_OLD":
        return context.get("track_id")
    return None


def _reference_rows(
    payload: Mapping[str, Any],
    *,
    images: Mapping[tuple[int, int, int], int],
    image_meta: Mapping[int, Mapping[str, Any]],
    predictions: Mapping[int, list[Mapping[str, Any]]],
):
    video_id = int(payload["video_id"])
    frame = int((payload.get("metadata") or {}).get("dataset_frame", payload["frame"]))
    view = int(payload["view"])
    image_id = images.get((video_id, view + 1, frame)) or images.get((video_id, view, frame))
    if image_id is None:
        return []
    image = image_meta[image_id]
    image_h, image_w = (float(payload["image_size"][0]), float(payload["image_size"][1]))
    scale_x = float(image["width"]) / image_w if image_w else 1.0
    scale_y = float(image["height"]) / image_h if image_h else 1.0
    rows = []
    for row, box in enumerate(torch.as_tensor(payload["pred_boxes"]).tolist()):
        xyxy = (box[0] * scale_x, box[1] * scale_y, box[2] * scale_x, box[3] * scale_y)
        rows.append((row, xyxy))
    return image_id, rows, predictions.get(int(image_id), [])


def replay(
    *,
    trace: Path,
    cache_root: Path,
    checkpoint: Path,
    config: Path,
    device: str,
    view_num: int,
    history_limit: int,
    annotations: Path,
    reference_predictions: Path,
) -> Mapping[str, Any]:
    grouped = normalize_events(trace)
    cache = FrozenPerceptionCache(cache_root)
    images, image_meta = load_image_index(annotations)
    reference_payload = json.loads(reference_predictions.read_text(encoding="utf-8"))
    reference_by_image: dict[int, list[Mapping[str, Any]]] = {}
    for item in reference_payload:
        reference_by_image.setdefault(int(item["image_id"]), []).append(item)
    engine = build_formal_gmt_engine(
        config_file=config,
        checkpoint=checkpoint,
        device=device,
        view_num=view_num,
        history_limit=history_limit,
    )
    counts = {
        "events": 0,
        "proposal_mismatches": 0,
        "hungarian_assignment_mismatches": 0,
        "track_id_mismatches": 0,
        "memory_write_mismatches": 0,
        "memory_length_mismatches": 0,
        "stale_bank_mismatches": 0,
        "reactivation_mismatches": 0,
        "new_id_mismatches": 0,
        "state_boundary_mismatches": 0,
        "future_gt_access": 0,
        "reference_detections": 0,
        "reference_assignment_mismatches": 0,
        "reference_unmatched_detections": 0,
    }
    trajectory_hasher = hashlib.sha256()
    trajectory_count = 0
    trajectory_sample = []
    for video_id, events in sorted(grouped.items()):
        actions, memories, by_key = event_maps(events)
        keys = sorted((key for key in cache.keys() if key[0] == int(video_id)), key=lambda key: (key[1], key[2]))
        if not keys:
            raise ValueError(f"no frozen perception cache for video {video_id}")
        first_snapshot = (events[0].get("context") or {}).get("tracker_state_before") or {}
        first_payload = cache.load(*keys[0])
        feature_dim = int(torch.as_tensor(first_payload["reid_features"]).shape[1])
        state = _state_from_snapshot(first_snapshot, feature_dim)
        for key in keys:
            payload = cache.load(*key)
            current_events = by_key.get(key, ())
            if current_events:
                expected_before = (current_events[0].get("context") or {}).get("tracker_state_before")
                if expected_before:
                    projection = _state_projection(state)
                    mismatch = _state_mismatch(expected_before, projection)
                    counts["state_boundary_mismatches"] += mismatch
                    counts["memory_length_mismatches"] += int(
                        expected_before.get("memory_lengths", {}) != projection["memory_lengths"]
                    )
                    counts["stale_bank_mismatches"] += int(
                        expected_before.get("stale_ids", []) != projection["stale_ids"]
                    )
            previous_next_id = int(state.next_id)
            previous_active_ids = set(state.active_ids)
            previous_memory_entries = sum(len(values) for values in state.memory.values())
            result = engine.step(
                payload,
                state,
                actions=actions.get(key),
                memory_actions=memories.get(key),
            )
            initial_track_ids = list(result.get("initial_track_ids", ()))
            final_track_ids = list(result.get("final_track_ids", ()))
            reference = _reference_rows(
                payload,
                images=images,
                image_meta=image_meta,
                predictions=reference_by_image,
            )
            if reference:
                _image_id, reference_boxes, reference_rows = reference
                used_reference = set()
                for row, simulator_box in reference_boxes:
                    best_index = None
                    best_overlap = 0.0
                    for index, reference_row in enumerate(reference_rows):
                        if index in used_reference:
                            continue
                        box = reference_row.get("bbox") or ()
                        if len(box) != 4:
                            continue
                        reference_box = (
                            float(box[0]),
                            float(box[1]),
                            float(box[0]) + float(box[2]),
                            float(box[1]) + float(box[3]),
                        )
                        overlap = iou(simulator_box, reference_box)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_index = index
                    if best_index is None or best_overlap < 0.99:
                        counts["reference_unmatched_detections"] += 1
                        continue
                    used_reference.add(best_index)
                    counts["reference_detections"] += 1
                    simulator_id = result.get("committed_track_ids", {}).get(row)
                    reference_id = reference_rows[best_index].get("track_id")
                    if simulator_id is None or int(simulator_id) != int(reference_id):
                        counts["reference_assignment_mismatches"] += 1
            for event in current_events:
                counts["events"] += 1
                if event.get("future_gt_access") is not False:
                    counts["future_gt_access"] += 1
                row = (event.get("context") or {}).get("detection_index")
                if row is None:
                    continue
                row = int(row)
                action = str(event.get("off_action") or event.get("proposed_action"))
                expected = _expected_id(event, action)
                pair = result.get("initial_pairs", {}).get(row)
                proposal_id = initial_track_ids[int(pair)] if pair is not None and int(pair) < len(initial_track_ids) else None
                final_pair = result.get("final_pairs", {}).get(row)
                final_id = result.get("committed_track_ids", {}).get(row)
                if expected is not None and proposal_id != int(expected):
                    counts["proposal_mismatches"] += 1
                if expected is not None and action in {"ACCEPT_CURRENT", "REASSOCIATE", "REACTIVATE_OLD"} and final_id != int(expected):
                    counts["track_id_mismatches"] += 1
                if action == "REASSOCIATE" and expected is not None and final_id != int(expected):
                    counts["reactivation_mismatches"] += 1
                if action == "START_NEW" and (
                    final_id is None
                    or int(final_id) <= previous_next_id
                    or int(final_id) in previous_active_ids
                ):
                    counts["new_id_mismatches"] += 1
                if pair is not None and final_pair is None and action != "START_NEW":
                    counts["hungarian_assignment_mismatches"] += 1
                trajectory_item = {
                    "video_id": int(video_id),
                    "frame": int(key[1]),
                    "view": int(key[2]),
                    "detection_index": row,
                    "action": action,
                    "proposal_track_id": proposal_id,
                    "committed_track_id": final_id,
                }
                encoded = json.dumps(trajectory_item, sort_keys=True, separators=(",", ":")).encode()
                trajectory_hasher.update(encoded + b"\n")
                trajectory_count += 1
                if len(trajectory_sample) < 20:
                    trajectory_sample.append(trajectory_item)
                elif trajectory_count % 1000 == 0:
                    trajectory_sample[-1] = trajectory_item
            expected_memory_writes = sum(
                int(action == "WRITE_MEMORY") for action in memories.get(key, {}).values()
            )
            actual_step_writes = sum(len(values) for values in state.memory.values()) - previous_memory_entries
            if actual_step_writes != expected_memory_writes:
                counts["memory_write_mismatches"] += 1
    counts["final_trajectory_events"] = trajectory_count
    failures = sum(
        value for key, value in counts.items()
        if key != "events" and key != "final_trajectory_events" and isinstance(value, int)
    )
    return {
        "status": "PASS" if counts["events"] and not failures else "FAIL",
        "equivalence_mode": "v2_simulator_off_replay",
        "gate_type": "V2_SIMULATOR_OFF_REPLAY_GATE",
        "gate_authority": "FORMAL_COUNTERFACTUAL_PREREQUISITE",
        "official_selection_authority": False,
        "official_test_authority": False,
        "trace": str(trace.resolve()),
        "trace_sha256": sha256(trace),
        "perception_cache": str(cache_root.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "config": str(config.resolve()),
        "config_sha256": sha256(config),
        "annotations": str(annotations.resolve()),
        "annotations_sha256": sha256(annotations),
        "reference_predictions": str(reference_predictions.resolve()),
        "reference_predictions_sha256": sha256(reference_predictions),
        "counts": counts,
        "final_trajectory_sha256": "sha256:" + trajectory_hasher.hexdigest(),
        "final_trajectory_sample": trajectory_sample,
        "future_gt_access": False,
        "review_note": "PASS is required before formal policy labels are authoritative",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--gmt-checkpoint", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--view-num", type=int, default=2)
    parser.add_argument("--history-limit", type=int, default=80)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    report = replay(
        trace=args.trace.resolve(),
        cache_root=args.cache.resolve(),
        checkpoint=args.gmt_checkpoint.resolve(),
        config=args.config_file.resolve(),
        device=args.device,
        view_num=args.view_num,
        history_limit=args.history_limit,
        annotations=args.annotations.resolve(),
        reference_predictions=args.reference_predictions.resolve(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
