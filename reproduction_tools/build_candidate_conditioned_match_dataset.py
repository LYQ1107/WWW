"""Build candidate-level MATCH returns from a frozen native GMT trace.

This labeler is deliberately separate from the action-level JEV dataset.  It
evaluates every legal native candidate with an isolated frozen-evidence branch
rollout and never treats the native ACCEPT/REASSOCIATE action as a candidate
label.  Absolute track IDs are retained only as runtime binding metadata; the
feature vector contains candidate-local evidence and no ID value.

The output is a screening/provenance artifact.  It is not a Full-H8
authorization and it is not an online-consumable dataset because it contains
future-GT returns.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from jev_mutable_branch import FrozenEvidenceGMTBranchRunner, TraceTrackerState, is_main_decision


SCHEMA_VERSION = "jev_candidate_match_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite {name}: {value!r}")
    return result


def xywh_to_xyxy(box: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, width, height = (float(value) for value in box[:4])
    return x, y, x + width, y + height


def iou(left: Sequence[float], right: Sequence[float]) -> float:
    ix1, iy1 = max(float(left[0]), float(right[0])), max(float(left[1]), float(right[1]))
    ix2, iy2 = min(float(left[2]), float(right[2])), min(float(left[3]), float(right[3]))
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def load_gt(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    videos = {int(video["id"]): video for video in payload["videos"]}
    images = {
        (int(image["video_id"]), int(image["view_id"]), int(image["frame_id"])): int(image["id"])
        for image in payload["images"]
    }
    image_meta = {int(image["id"]): image for image in payload["images"]}
    gt_by_image: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = {}
    for annotation in payload["annotations"]:
        gt_by_image.setdefault(int(annotation["image_id"]), []).append(
            (int(annotation["instance_id"]), xywh_to_xyxy(annotation["bbox"]))
        )
    return videos, images, gt_by_image, image_meta


def normalize_events(path: Path, video_id: int) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if not line.strip():
                continue
            event = json.loads(line)
            context = dict(event.get("context", {}))
            if int(context.get("video_id", -1)) != int(video_id):
                continue
            vector = event.get("state_feature_vector")
            if not isinstance(vector, list) or not vector:
                raise ValueError(f"trace line {line_number + 1} has no state_feature_vector")
            event["context"] = context
            event["_order"] = int(context.get("event_order", line_number))
            event["_video_id"] = int(video_id)
            event["_frame"] = int(context.get("frame", 0))
            event["_view"] = int(context.get("view", 0))
            events.append(event)
    events.sort(key=lambda item: (int(item["_order"]), int(item["_frame"]), int(item["_view"])))
    return events


def observed_gt(
    event: Mapping[str, Any],
    images: Mapping[Tuple[int, int, int], int],
    gt_by_image: Mapping[int, Sequence[Tuple[int, Sequence[float]]]],
    image_meta: Mapping[int, Mapping[str, Any]],
) -> int | None:
    context = event["context"]
    bbox = context.get("bbox_xyxy")
    if not bbox:
        return None
    video_id = int(event["_video_id"])
    frame = int(event["_frame"])
    view = int(context.get("view", 0)) + 1
    image_id = images.get((video_id, view, frame))
    if image_id is None:
        image_id = images.get((video_id, int(context.get("view", 0)), frame))
    if image_id is None or not gt_by_image.get(image_id):
        return None
    image_info = image_meta[image_id]
    model_size = context.get("model_image_size")
    scaled = [float(value) for value in bbox[:4]]
    if isinstance(model_size, list) and len(model_size) == 2:
        scaled = [
            scaled[0] * float(image_info["width"]) / float(model_size[1]),
            scaled[1] * float(image_info["height"]) / float(model_size[0]),
            scaled[2] * float(image_info["width"]) / float(model_size[1]),
            scaled[3] * float(image_info["height"]) / float(model_size[0]),
        ]
    best_id, best_overlap = max(
        ((int(instance_id), iou(scaled, gt_box)) for instance_id, gt_box in gt_by_image[image_id]),
        key=lambda item: item[1],
    )
    return best_id if best_overlap >= 0.5 else None


def utility_target(labels: Sequence[str], utilities: Mapping[str, float]) -> Tuple[List[str], List[float]]:
    values = [finite(utilities[label], name="candidate utility") for label in labels]
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    normalizer = sum(weights)
    probabilities = [weight / normalizer for weight in weights]
    best_value = max(values)
    best = [label for label, value in zip(labels, values) if best_value - value <= 1e-8]
    return best, probabilities


def candidate_features(event: Mapping[str, Any], candidate_index: int) -> List[float]:
    """Return ID-free evidence for one legal candidate row."""

    context = event["context"]
    scores = [finite(value, name="candidate score") for value in context["candidate_scores"]]
    if len(scores) != len(context["candidate_track_ids"]):
        raise ValueError("candidate score/ID length mismatch")
    if not 0 <= candidate_index < len(scores):
        raise ValueError("candidate index out of range")
    ordered = sorted(scores, reverse=True)
    score = scores[candidate_index]
    proposal_id = context.get("proposal_track_id")
    alternate_id = context.get("alternate_track_id")
    candidate_id = int(context["candidate_track_ids"][candidate_index])
    rank = 1 + sum(other > score for other in scores)
    top = ordered[0] if ordered else 0.0
    second = ordered[1] if len(ordered) > 1 else top
    detection_score = finite(context.get("detection_score", 0.0), name="detection score")
    return [
        score,
        top - score,
        score - second if rank == 1 else score - top,
        float(rank) / float(max(1, len(scores))),
        float(len(scores)),
        detection_score,
        float(candidate_id == int(proposal_id)) if proposal_id is not None else 0.0,
        float(candidate_id == int(alternate_id)) if alternate_id is not None else 0.0,
    ]


def start_new_features(event: Mapping[str, Any]) -> List[float]:
    """Return a fixed-size sentinel row with no absolute-ID feature."""

    context = event["context"]
    scores = [finite(value, name="candidate score") for value in context["candidate_scores"]]
    top = max(scores) if scores else 0.0
    return [0.0, top, -top, 1.0, float(len(scores)), finite(context.get("detection_score", 0.0), name="detection score"), 0.0, 0.0]


def candidate_label(
    runner: FrozenEvidenceGMTBranchRunner,
    base_state: TraceTrackerState,
    position: int,
    event: Mapping[str, Any],
    horizon: int,
) -> Dict[str, Any]:
    context = event["context"]
    native_ids = [int(value) for value in context.get("candidate_track_ids", ())]
    outcomes = runner.run_candidate_rollouts(
        base_state,
        position,
        event,
        native_ids,
        horizon,
        include_start_new=True,
    )
    labels = list(outcomes)
    utilities = {label: float(outcomes[label]["utility"]) for label in labels}
    best, probabilities = utility_target(labels, utilities)
    common_weight = min(float(outcomes[label].get("sample_weight", 1.0)) for label in labels)
    if any(abs(float(outcomes[label].get("sample_weight", 1.0)) - common_weight) > 1e-8 for label in labels):
        raise ValueError("candidate outcomes do not share a sample weight")
    rows = []
    for index, candidate_id in enumerate(native_ids):
        label = f"track:{candidate_id}"
        rows.append({
            "candidate_key": label,
            "candidate_track_id": candidate_id,
            "features": candidate_features(event, index),
            "outcome": outcomes[label],
        })
    rows.append({
        "candidate_key": "START_NEW",
        "candidate_track_id": None,
        "features": start_new_features(event),
        "outcome": outcomes["START_NEW"],
    })
    return {
        "candidates": rows,
        "best_candidates": list(best),
        "target_probs": {label: float(probability) for label, probability in zip(labels, probabilities)},
        "sample_weight": float(common_weight),
    }


def validate_record(record: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "dataset", "sequence", "video_id", "frame", "view",
        "event_order", "detection_index", "question_type", "state_features",
        "candidates", "best_candidates", "target_probs", "sample_weight",
        "horizon", "uses_future_gt", "native_off_action", "provenance",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"candidate record missing fields: {sorted(missing)}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unexpected candidate schema")
    if record["question_type"] != "MATCH_DECISION":
        raise ValueError("candidate dataset must contain MATCH_DECISION only")
    if not record["uses_future_gt"]:
        raise ValueError("candidate labels must declare future GT use")
    candidates = list(record["candidates"])
    keys = [str(row["candidate_key"]) for row in candidates]
    if len(keys) != len(set(keys)) or "START_NEW" not in keys:
        raise ValueError("candidate keys must be unique and include START_NEW")
    if set(record["best_candidates"]) - set(keys):
        raise ValueError("best candidate is outside legal candidate set")
    probabilities = record["target_probs"]
    if set(probabilities) != set(keys):
        raise ValueError("target probabilities do not align with candidates")
    if abs(sum(float(value) for value in probabilities.values()) - 1.0) > 1e-5:
        raise ValueError("candidate target probabilities do not sum to one")
    state_features = list(record["state_features"])
    if not state_features or any(not math.isfinite(float(value)) for value in state_features):
        raise ValueError("state features must be finite")
    for row in candidates:
        features = list(row["features"])
        if len(features) != 8 or any(not math.isfinite(float(value)) for value in features):
            raise ValueError("candidate features must be eight finite ID-free values")
        outcome = row["outcome"]
        if not math.isfinite(float(outcome["utility"])):
            raise ValueError("candidate utility must be finite")


def build(
    trace: Path,
    annotations: Path,
    checkpoint: Path,
    output: Path,
    *,
    video_id: int,
    source_commit: str,
    expected_source_commit: str,
    horizon: int,
    max_records: int | None,
) -> Dict[str, Any]:
    if source_commit != expected_source_commit:
        raise RuntimeError(
            f"source commit mismatch: trace={source_commit} expected={expected_source_commit}"
        )
    videos, images, gt_by_image, image_meta = load_gt(annotations)
    events = normalize_events(trace, int(video_id))
    if not events:
        raise ValueError(f"no events for video {video_id}")
    base_state = TraceTrackerState.from_event(events[0])
    runner = FrozenEvidenceGMTBranchRunner(
        events,
        lambda item: observed_gt(item, images, gt_by_image, image_meta),
    )
    trace_hash = sha256(trace)
    annotations_hash = sha256(annotations)
    checkpoint_hash = sha256(checkpoint)
    sequence = str(videos.get(int(video_id), {}).get("file_name", video_id))
    records: List[Dict[str, Any]] = []
    count_by_question: Dict[str, int] = {}
    for position, event in enumerate(events):
        question = str(event.get("question"))
        if is_main_decision(event) and question == "MATCH_DECISION":
            label = candidate_label(runner, base_state, position, event, horizon)
            record = {
                "schema_version": SCHEMA_VERSION,
                "dataset": "VisionTrack",
                "sequence": sequence,
                "video_id": int(video_id),
                "frame": int(event["_frame"]),
                "view": int(event["_view"]),
                "event_order": int(event["_order"]),
                "detection_index": event["context"].get("detection_index"),
                "question_type": "MATCH_DECISION",
                "state_features": [finite(value, name="state feature") for value in event["state_feature_vector"]],
                "candidates": label["candidates"],
                "best_candidates": label["best_candidates"],
                "target_probs": label["target_probs"],
                "sample_weight": label["sample_weight"],
                "horizon": int(horizon),
                "uses_future_gt": True,
                "native_off_action": str(event.get("off_action") or event.get("proposed_action")),
                "native_candidate_key": (
                    "START_NEW"
                    if str(event.get("off_action") or event.get("proposed_action")) == "START_NEW"
                    else f"track:{int(event['context']['proposal_track_id'])}"
                    if str(event.get("off_action") or event.get("proposed_action")) == "ACCEPT_CURRENT"
                    else f"track:{int(event['context']['alternate_track_id'])}"
                    if str(event.get("off_action") or event.get("proposed_action")) == "REASSOCIATE"
                    else None
                ),
                "native_proposal_track_id": event["context"].get("proposal_track_id"),
                "native_alternate_track_id": event["context"].get("alternate_track_id"),
                "provenance": {
                    "trace": str(trace.resolve()),
                    "trace_sha256": trace_hash,
                    "annotations": str(annotations.resolve()),
                    "annotations_sha256": annotations_hash,
                    "gmt_checkpoint": str(checkpoint.resolve()),
                    "gmt_checkpoint_sha256": checkpoint_hash,
                    "source_commit": source_commit,
                    "candidate_utility": "frozen_evidence_mutable_gmt_state_v1",
                    "candidate_ids_are_metadata_only": True,
                    "native_action_label_copied": False,
                },
            }
            validate_record(record)
            records.append(record)
            count_by_question[question] = count_by_question.get(question, 0) + 1
            if max_records is not None and len(records) >= max_records:
                break
        if is_main_decision(event):
            off_action = str(event.get("off_action") or event.get("proposed_action"))
            runner.apply(
                base_state,
                event,
                off_action,
                target=observed_gt(event, images, gt_by_image, image_meta),
                position=position,
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return {
        "status": "PASS",
        "schema_version": SCHEMA_VERSION,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "records": len(records),
        "records_by_question": count_by_question,
        "video_id": int(video_id),
        "sequence": sequence,
        "trace": str(trace.resolve()),
        "trace_sha256": trace_hash,
        "annotations": str(annotations.resolve()),
        "annotations_sha256": annotations_hash,
        "gmt_checkpoint": str(checkpoint.resolve()),
        "gmt_checkpoint_sha256": checkpoint_hash,
        "source_commit": source_commit,
        "expected_source_commit": expected_source_commit,
        "horizon": int(horizon),
        "screening_only": True,
        "full_h8_authorized": False,
        "native_action_label_copied": False,
        "output_sha256": sha256(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--gmt-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-id", type=int, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()
    if args.horizon < 1:
        raise ValueError("horizon must be positive")
    result = build(
        args.trace.resolve(),
        args.annotations.resolve(),
        args.gmt_checkpoint.resolve(),
        args.output.resolve(),
        video_id=int(args.video_id),
        source_commit=str(args.source_commit),
        expected_source_commit=str(args.expected_source_commit),
        horizon=int(args.horizon),
        max_records=args.max_records,
    )
    report_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
