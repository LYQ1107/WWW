"""Build offline JEV records from an online GMT decision trace.

The trace contains only causal state/evidence and the GMT proposal.  This
labeler is allowed to read future GT and uses it only to score isolated action
branches.  It is deliberately a proxy rollout over the frozen GMT decision
trace, not an online evaluator and not a candidate-reranking model.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from jev_dataset_contract import validate_record
from jev_dataset_tools import make_record
from jev_mutable_branch import FrozenEvidenceGMTBranchRunner, TraceTrackerState, is_main_decision


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def xywh_to_xyxy(box: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, width, height = (float(value) for value in box[:4])
    return x, y, x + width, y + height


def iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def load_gt(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    videos = {int(video["id"]): video for video in payload["videos"]}
    images = {}
    for image in payload["images"]:
        key = (int(image["video_id"]), int(image["view_id"]), int(image["frame_id"]))
        images[key] = int(image["id"])
    gt_by_image: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = {}
    for annotation in payload["annotations"]:
        gt_by_image.setdefault(int(annotation["image_id"]), []).append(
            (int(annotation["instance_id"]), xywh_to_xyxy(annotation["bbox"]))
        )
    image_meta = {int(image["id"]): image for image in payload["images"]}
    return videos, images, gt_by_image, image_meta


def event_key(event: Mapping[str, Any], order: int) -> Tuple[int, int, int]:
    context = event.get("context", {})
    return (int(context.get("frame", 0)), int(context.get("view", 0)), order)


def normalize_events(
    path: Path, *, video_ids: Optional[Sequence[int]] = None
) -> Dict[int, List[Dict[str, Any]]]:
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    wanted = {int(value) for value in video_ids} if video_ids is not None else None
    with path.open(encoding="utf-8") as handle:
        for order, line in enumerate(handle):
            if not line.strip():
                continue
            event = json.loads(line)
            context = dict(event.get("context", {}))
            video_id = int(context.get("video_id", -1))
            if video_id < 0:
                raise ValueError("trace event is missing context.video_id")
            if wanted is not None and video_id not in wanted:
                continue
            vector = event.get("state_feature_vector")
            if not isinstance(vector, list) or not vector:
                raise ValueError("trace event is missing state_feature_vector")
            event["context"] = context
            event["_order"] = order
            event["_video_id"] = video_id
            event["_frame"] = int(context.get("frame", 0))
            event["_view"] = int(context.get("view", 0))
            grouped.setdefault(video_id, []).append(event)
    for events in grouped.values():
        events.sort(key=lambda item: event_key(item, int(item["_order"])))
    return grouped


def observed_gt(
    event: Mapping[str, Any],
    images: Mapping[Tuple[int, int, int], int],
    gt_by_image: Mapping[int, Sequence[Tuple[int, Sequence[float]]]],
    image_meta: Mapping[int, Mapping[str, Any]],
) -> Optional[int]:
    context = event["context"]
    bbox = context.get("bbox_xyxy")
    if not bbox:
        return None
    view = int(context.get("view", 0)) + 1
    image_id = images.get((int(event["_video_id"]), view, int(event["_frame"])))
    if image_id is None:
        # First-view traces can be zero-indexed or omit the view.  Try both
        # deterministic interpretations before treating the observation as
        # unmatched.
        image_id = images.get((int(event["_video_id"]), int(context.get("view", 0)), int(event["_frame"])))
    if image_id is None:
        return None
    candidates = gt_by_image.get(image_id, ())
    if not candidates:
        return None
    image_info = image_meta[image_id]
    model_size = context.get("model_image_size")
    if model_size and len(model_size) == 2:
        model_h, model_w = float(model_size[0]), float(model_size[1])
        if model_h > 0 and model_w > 0:
            bbox = [
                float(bbox[0]) * float(image_info["width"]) / model_w,
                float(bbox[1]) * float(image_info["height"]) / model_h,
                float(bbox[2]) * float(image_info["width"]) / model_w,
                float(bbox[3]) * float(image_info["height"]) / model_h,
            ]
    best_id, best_overlap = max(
        ((int(instance_id), iou(bbox, gt_box)) for instance_id, gt_box in candidates),
        key=lambda item: item[1],
    )
    return best_id if best_overlap >= 0.5 else None


def synthetic_id(event: Mapping[str, Any]) -> str:
    return "new:{}:{}:{}:{}".format(
        event["_video_id"], event["_frame"], event["_view"], event["_order"]
    )


def proposed_id(event: Mapping[str, Any], action: str) -> Optional[Any]:
    context = event["context"]
    if action == "ACCEPT_CURRENT":
        return context.get("proposal_track_id")
    if action == "REASSOCIATE":
        return context.get("alternate_track_id")
    if action == "REACTIVATE_OLD":
        return context.get("track_id")
    if action == "START_NEW":
        return synthetic_id(event)
    return None


def assign(mapping: MutableMapping[Any, int], identifier: Optional[Any], target: Optional[int]) -> float:
    if identifier is None or target is None:
        return 0.0
    previous = mapping.get(identifier)
    score = 1.0 if previous is None or previous == target else -1.0
    mapping[identifier] = int(target)
    return score


def future_identity_utility(
    events: Sequence[Mapping[str, Any]],
    position: int,
    horizon: int,
    mapping: Mapping[Any, int],
    current_event: Mapping[str, Any],
    action: str,
    images,
    gt_by_image,
    image_meta,
) -> Dict[str, float]:
    branch_mapping: Dict[Any, int] = dict(mapping)
    target = observed_gt(current_event, images, gt_by_image, image_meta)
    immediate = assign(branch_mapping, proposed_id(current_event, action), target)
    correct = 0
    total = 0
    switches = 0
    fragments = 0
    previous_target_by_gt: Dict[int, Any] = {}
    for future in events[position + 1 :]:
        frame = int(future["_frame"])
        if frame > int(current_event["_frame"]) + horizon:
            break
        if future.get("question") not in {"MATCH_DECISION", "REACTIVATION_DECISION"}:
            continue
        future_target = observed_gt(future, images, gt_by_image, image_meta)
        if future_target is None:
            continue
        base_action = str(future.get("off_action") or future.get("proposed_action"))
        identifier = proposed_id(future, base_action)
        previous = branch_mapping.get(identifier)
        if previous is not None and previous != future_target:
            switches += 1
        if previous_target_by_gt.get(future_target) is not None and previous_target_by_gt[future_target] != identifier:
            fragments += 1
        assign(branch_mapping, identifier, future_target)
        previous_target_by_gt[future_target] = identifier
        total += 1
        correct += int(branch_mapping.get(identifier) == future_target)
    consistency = correct / total if total else 0.0
    utility = immediate + consistency - 0.5 * switches - 0.25 * fragments
    return {
        "utility": float(utility),
        "immediate_identity": float(immediate),
        "future_identity_consistency": float(consistency),
        "future_identity_switches": float(switches),
        "future_fragments": float(fragments),
        "future_events": float(total),
    }


def memory_utility(
    events: Sequence[Mapping[str, Any]],
    position: int,
    horizon: int,
    memory: Mapping[Any, int],
    event: Mapping[str, Any],
    action: str,
    images,
    gt_by_image,
    image_meta,
) -> Dict[str, float]:
    branch_memory = dict(memory)
    track_id = event["context"].get("track_id")
    target = observed_gt(event, images, gt_by_image, image_meta)
    contamination = 0.0
    if action == "WRITE_MEMORY" and track_id is not None and target is not None:
        if track_id in branch_memory and branch_memory[track_id] != target:
            contamination = 1.0
        branch_memory[track_id] = target
    consistent = 0
    total = 0
    for future in events[position + 1 :]:
        if int(future["_frame"]) > int(event["_frame"]) + horizon:
            break
        if future.get("question") != "MATCH_DECISION":
            continue
        future_target = observed_gt(future, images, gt_by_image, image_meta)
        if future_target is None or track_id is None:
            continue
        context = future["context"]
        if track_id not in {context.get("proposal_track_id"), context.get("alternate_track_id")}:
            continue
        total += 1
        consistent += int(branch_memory.get(track_id) in {None, future_target})
    ratio = consistent / total if total else 0.0
    return {
        "utility": float(ratio - contamination),
        "future_memory_consistency": float(ratio),
        "memory_contamination": float(contamination),
        "future_events": float(total),
    }


def make_records(trace: Path, annotations: Path, checkpoint_hash: str, horizon: int):
    videos, images, gt_by_image, image_meta = load_gt(annotations)
    grouped = normalize_events(trace)
    records = []
    stats: Dict[str, int] = {}
    for video_id, events in grouped.items():
        sequence = str(videos.get(video_id, {}).get("file_name", video_id))
        if not events:
            continue
        base_state = TraceTrackerState.from_event(events[0])
        runner = FrozenEvidenceGMTBranchRunner(
            events,
            lambda item: observed_gt(item, images, gt_by_image, image_meta),
        )
        for position, event in enumerate(events):
            question = str(event.get("question"))
            if question not in {"MATCH_DECISION", "MEMORY_DECISION", "REACTIVATION_DECISION"}:
                continue
            if not is_main_decision(event):
                continue
            legal = list(event.get("legal_actions", ()))
            outcomes = runner.run(base_state, position, event, legal, horizon)
            state = {
                "feature_vector": [float(value) for value in event["state_feature_vector"]],
                "online_context": {
                    "video_id": int(event["_video_id"]),
                    "frame": int(event["_frame"]),
                    "view": int(event["_view"]),
                    "event_order": int(event["_order"]),
                    "detection_index": event["context"].get("detection_index"),
                    "model_image_size": event["context"].get("model_image_size"),
                    "decision_scope": event["context"].get("decision_scope"),
                    "proposal_track_id": event["context"].get("proposal_track_id"),
                    "alternate_track_id": event["context"].get("alternate_track_id"),
                    "track_id": event["context"].get("track_id"),
                    "detection_score": event["context"].get("detection_score"),
                },
                "tracker_state_snapshot": event["context"].get("tracker_state_before"),
                "counterfactual_engine": "frozen_evidence_mutable_gmt_state_v1",
            }
            record = make_record(
                dataset="VisionTrack",
                sequence=sequence,
                frame=int(event["_frame"]),
                view=int(event["_view"]),
                question_type=question,
                state=state,
                legal_actions=legal,
                action_outcomes=outcomes,
                gmt_checkpoint_sha256=checkpoint_hash,
                horizon=horizon,
            )
            validate_record(record, allow_future_gt=True)
            records.append(record)
            stats[question] = stats.get(question, 0) + 1

            # Advance the observed OFF branch only after all counterfactual
            # branches have been scored from the same pre-decision state.
            off_action = str(event.get("off_action") or event.get("proposed_action"))
            target = observed_gt(event, images, gt_by_image, image_meta)
            runner.apply(
                base_state,
                event,
                off_action,
                target=target,
                position=position,
            )
    return records, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gmt-checkpoint", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=32)
    args = parser.parse_args()
    if args.horizon < 1:
        raise ValueError("horizon must be positive")
    if args.output.exists():
        raise RuntimeError(f"refusing to overwrite {args.output}")
    trace = args.trace.resolve()
    annotations = args.annotations.resolve()
    checkpoint = args.gmt_checkpoint.resolve()
    records, stats = make_records(trace, annotations, sha256(checkpoint), args.horizon)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    report = {
        "status": "PASS",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "trace": str(trace),
        "trace_sha256": sha256(trace),
        "annotations": str(annotations),
        "annotations_sha256": sha256(annotations),
        "gmt_checkpoint": str(checkpoint),
        "gmt_checkpoint_sha256": sha256(checkpoint),
        "horizon": args.horizon,
        "records": len(records),
        "records_by_question": stats,
        "uses_future_gt": True,
        "online_model_features": "trace state_feature_vector only",
        "labeling_note": "offline frozen-evidence rollout over deep-copied mutable GMT tracker containers; detector/association outputs are frozen and GT is used only for branch utility",
        "counterfactual_engine": "frozen_evidence_mutable_gmt_state_v1",
    }
    report_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
