"""Read-only audit of candidate-to-GT identity feasibility.

The audit reconstructs native OFF track identity evidence from the immutable
trace and frame-zero seed payload. It never creates training labels and never
copies ACCEPT/REASSOCIATE/START_NEW action labels into candidate rows.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Mapping, Sequence, Tuple

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def xywh_to_xyxy(box: Sequence[float]) -> Tuple[float, float, float, float]:
    x, y, width, height = (float(value) for value in box[:4])
    return x, y, x + width, y + height


def iou(left: Sequence[float], right: Sequence[float]) -> float:
    ix1 = max(float(left[0]), float(right[0]))
    iy1 = max(float(left[1]), float(right[1]))
    ix2 = min(float(left[2]), float(right[2]))
    iy2 = min(float(left[3]), float(right[3]))
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    right_area = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


def load_annotations(path: Path, video_id: int):
    payload = json.loads(path.read_text(encoding="utf-8"))
    image_by_key: Dict[Tuple[int, int, int], Mapping[str, Any]] = {}
    selected = set()
    for image in payload["images"]:
        if int(image.get("video_id", -1)) != video_id:
            continue
        image_by_key[(video_id, int(image["frame_id"]), int(image["view_id"]))] = image
        selected.add(int(image["id"]))
    gt_by_image: DefaultDict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        image_id = int(annotation["image_id"])
        if image_id in selected:
            gt_by_image[image_id].append((int(annotation["instance_id"]), xywh_to_xyxy(annotation["bbox"])))
    if not image_by_key:
        raise ValueError(f"no images for video {video_id}")
    return image_by_key, gt_by_image


def load_cache_index(path: Path, video_id: int):
    result: Dict[Tuple[int, int, int], Mapping[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if int(item.get("video_id", -1)) != video_id:
                continue
            image_size = item.get("image_size")
            if not isinstance(image_size, list) or len(image_size) != 2:
                raise ValueError(f"cache index line {line_number} has no image_size")
            record_path = Path(str(item["record"]))
            if not record_path.is_absolute():
                record_path = (path.parent / record_path).resolve()
            result[(video_id, int(item["frame"]), int(item["view"]))] = {
                "image_size": (int(image_size[0]), int(image_size[1])),
                "detection_count": int(item.get("detection_count", 0)),
                "record": str(record_path),
            }
    if not result:
        raise ValueError(f"no cache entries for video {video_id}")
    return result


def load_events(path: Path, video_id: int):
    events = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            event = json.loads(line)
            if int(event.get("context", {}).get("video_id", -1)) != video_id:
                continue
            event = dict(event)
            event["_line"] = line_number
            event["_order"] = int(event["context"].get("event_order", line_number))
            events.append(event)
    events.sort(key=lambda item: (int(item["_order"]), int(item["_line"])))
    if not events:
        raise ValueError(f"no trace events for video {video_id}")
    return events


def image_for_event(event, image_by_key):
    context = event["context"]
    key = (int(context["video_id"]), int(context["frame"]) + 1, int(context["view"]) + 1)
    try:
        return image_by_key[key]
    except KeyError as exc:
        raise KeyError(f"missing annotation image for {key}") from exc


def match_box_to_gt(box, image, model_size, gt_rows, threshold):
    scale_x = float(image["width"]) / float(model_size[1])
    scale_y = float(image["height"]) / float(model_size[0])
    scaled = (float(box[0]) * scale_x, float(box[1]) * scale_y, float(box[2]) * scale_x, float(box[3]) * scale_y)
    best_id, best_iou = max(
        ((int(instance_id), iou(scaled, gt_box)) for instance_id, gt_box in gt_rows),
        key=lambda item: item[1],
        default=(None, 0.0),
    )
    return {
        "instance_id": int(best_id) if best_id is not None and best_iou >= threshold else None,
        "raw_best_instance_id": int(best_id) if best_id is not None else None,
        "iou": float(best_iou),
    }


def event_target(event, image_by_key, gt_by_image, cache_index, threshold):
    image = image_for_event(event, image_by_key)
    context = event["context"]
    cache_key = (int(context["video_id"]), int(context["frame"]), int(context["view"]))
    box = context.get("bbox_xyxy")
    if not isinstance(box, list) or len(box) < 4:
        return {"instance_id": None, "raw_best_instance_id": None, "iou": 0.0, "image_id": int(image["id"])}
    result = match_box_to_gt(box, image, cache_index[cache_key]["image_size"], gt_by_image.get(int(image["id"]), ()), threshold)
    result["image_id"] = int(image["id"])
    return result


def assigned_track(event) -> int | None:
    question = str(event.get("question", ""))
    action = str(event.get("committed_action") or event.get("off_action") or "")
    context = event["context"]
    if question == "MATCH_DECISION":
        value = context.get("proposal_track_id") if action == "ACCEPT_CURRENT" else context.get("alternate_track_id") if action == "REASSOCIATE" else None
    elif question == "REACTIVATION_DECISION" and action == "REACTIVATE_OLD":
        value = context.get("track_id")
    else:
        value = None
    return int(value) if value is not None else None


def add_seed_observations(observations, cache_index, image_by_key, gt_by_image, video_id, threshold):
    candidates = [(key, value) for key, value in cache_index.items() if int(key[1]) == 0]
    seed_key, seed_meta = max(candidates, key=lambda item: (int(item[1]["detection_count"]), int(item[0][2])))
    record_path = Path(str(seed_meta["record"]))
    payload = torch.load(str(record_path), map_location="cpu")
    image = image_by_key[(video_id, 1, int(seed_key[2]) + 1)]
    boxes = payload.get("pred_boxes")
    if boxes is None:
        raise ValueError(f"seed payload has no pred_boxes: {record_path}")
    for row, box in enumerate(boxes):
        result = match_box_to_gt(box.tolist(), image, seed_meta["image_size"], gt_by_image.get(int(image["id"]), ()), threshold)
        observations[row + 1].append({"order": -1, "frame": 0, "view": int(seed_key[2]), **result})
    return {"video_id": int(video_id), "frame": 0, "view": int(seed_key[2]), "detection_count": int(seed_meta["detection_count"]), "record": str(record_path)}


def candidate_identity(observations, track_id, before_order, consistency_threshold):
    prior = [item for item in observations.get(int(track_id), ()) if int(item["order"]) < int(before_order) and item.get("instance_id") is not None]
    counts = Counter(int(item["instance_id"]) for item in prior)
    if not counts:
        return {"track_id": int(track_id), "instance_id": None, "observations": 0, "identity_counts": {}, "consistency": 0.0, "status": "UNRESOLVED_NO_PRIOR_GT_OBSERVATION"}
    identity, count = counts.most_common(1)[0]
    total = sum(counts.values())
    consistency = float(count) / float(total)
    return {"track_id": int(track_id), "instance_id": int(identity) if consistency >= consistency_threshold else None, "observations": int(total), "identity_counts": {str(key): int(value) for key, value in sorted(counts.items())}, "consistency": consistency, "status": "PASS" if consistency >= consistency_threshold else "AMBIGUOUS_ID_SWITCH_HISTORY"}


def audit(trace: Path, annotations: Path, cache_index_path: Path, video_id: int, iou_threshold: float, consistency_threshold: float):
    image_by_key, gt_by_image = load_annotations(annotations, video_id)
    cache_index = load_cache_index(cache_index_path, video_id)
    events = load_events(trace, video_id)
    observations: DefaultDict[int, List[Mapping[str, Any]]] = defaultdict(list)
    seed = add_seed_observations(observations, cache_index, image_by_key, gt_by_image, video_id, iou_threshold)
    for event in events:
        if str(event.get("question")) not in {"MATCH_DECISION", "REACTIVATION_DECISION"}:
            continue
        track_id = assigned_track(event)
        if track_id is None:
            continue
        target = event_target(event, image_by_key, gt_by_image, cache_index, iou_threshold)
        observations[int(track_id)].append({"order": int(event["_order"]), "frame": int(event["context"]["frame"]), "view": int(event["context"]["view"]), **target})

    by_question = {}
    all_rows = []
    for question in ("MATCH_DECISION", "REACTIVATION_DECISION"):
        question_events = [event for event in events if str(event.get("question")) == question]
        rows = []
        mapped_current = 0
        recall_events = 0
        ranks = []
        unresolved_rows = 0
        ambiguous_rows = 0
        ambiguous_events = 0
        for event in question_events:
            context = event["context"]
            current = event_target(event, image_by_key, gt_by_image, cache_index, iou_threshold)
            current_id = current.get("instance_id")
            mapped_current += int(current_id is not None)
            id_field = "candidate_track_ids" if question == "MATCH_DECISION" else "native_candidate_track_ids"
            score_field = "candidate_scores" if question == "MATCH_DECISION" else "native_candidate_scores"
            candidate_ids = list(context.get(id_field) or ())
            candidate_scores = list(context.get(score_field) or ())
            if len(candidate_ids) != len(candidate_scores):
                raise ValueError(f"candidate ID/score length mismatch at trace line {event['_line']}")
            candidates = [
                {**candidate_identity(observations, int(track_id), int(event["_order"]), consistency_threshold), "score": float(score)}
                for track_id, score in zip(candidate_ids, candidate_scores)
            ]
            unresolved_rows += sum(item["status"] == "UNRESOLVED_NO_PRIOR_GT_OBSERVATION" for item in candidates)
            ambiguous_rows += sum(item["status"] == "AMBIGUOUS_ID_SWITCH_HISTORY" for item in candidates)
            ambiguous_events += int(any(item["status"] == "AMBIGUOUS_ID_SWITCH_HISTORY" for item in candidates))
            ranked = sorted(candidates, key=lambda item: item["score"], reverse=True)
            matching = [index + 1 for index, item in enumerate(ranked) if current_id is not None and item.get("instance_id") == current_id]
            if matching:
                recall_events += 1
                ranks.append(min(matching))
            rows.append({"trace_line": int(event["_line"]), "event_order": int(event["_order"]), "frame": int(context["frame"]), "view": int(context["view"]), "current_gt_instance_id": current_id, "current_bbox_iou": float(current.get("iou", 0.0)), "candidate_count": len(candidates), "candidates_ranked_by_score": ranked, "candidate_contains_current_gt": bool(matching), "current_gt_rank": min(matching) if matching else None})
        rank_counts = Counter(str(value) for value in ranks)
        denominator = mapped_current
        by_question[question] = {
            "events": len(question_events),
            "mapped_current_detections": int(mapped_current),
            "unmapped_current_detections": int(len(question_events) - mapped_current),
            "candidate_rows": int(sum(len(row["candidates_ranked_by_score"]) for row in rows)),
            "candidate_rows_with_prior_identity": int(sum(len(row["candidates_ranked_by_score"]) for row in rows) - unresolved_rows - ambiguous_rows),
            "unresolved_new_candidate_rows": int(unresolved_rows),
            "ambiguous_candidate_rows": int(ambiguous_rows),
            "unresolved_or_ambiguous_candidate_rows": int(unresolved_rows + ambiguous_rows),
            "events_with_ambiguous_or_unresolved_candidate": int(ambiguous_events),
            "candidate_set_recall": float(recall_events) / float(denominator) if denominator else None,
            "events_with_current_gt_candidate": int(recall_events),
            "rank_distribution": dict(sorted(rank_counts.items(), key=lambda item: int(item[0]))),
            "top1_rate_when_current_mapped": float(sum(value == 1 for value in ranks)) / float(denominator) if denominator else None,
            "mean_rank_when_present": float(sum(ranks)) / len(ranks) if ranks else None,
        }
        all_rows.extend({"question": question, **row} for row in rows)

    return {
        "status": "PASS" if all(item["unmapped_current_detections"] == 0 and item["unresolved_or_ambiguous_candidate_rows"] == 0 for item in by_question.values()) else "PASS_WITH_LIMITATIONS" if all(item["ambiguous_candidate_rows"] == 0 for item in by_question.values()) else "BLOCKED_AMBIGUOUS_IDENTITY_MAPPING",
        "schema_version": "jev_candidate_identity_mapping_audit_v2",
        "classification": "READ_ONLY_OFFLINE_AUDIT_NOT_TRAINING_LABELS_NOT_PAPER_RESULT",
        "video_id": int(video_id),
        "trace": str(trace.resolve()),
        "trace_sha256": sha256(trace),
        "annotations": str(annotations.resolve()),
        "annotations_sha256": sha256(annotations),
        "cache_index": str(cache_index_path.resolve()),
        "cache_index_sha256": sha256(cache_index_path),
        "iou_threshold": float(iou_threshold),
        "identity_consistency_threshold": float(consistency_threshold),
        "seed_initialization": seed,
        "by_question": by_question,
        "reactivation_events": int(by_question["REACTIVATION_DECISION"]["events"]),
        "current_detection_gt_mapping": {"mapped_events": int(by_question["REACTIVATION_DECISION"]["mapped_current_detections"]), "unmapped_events": int(by_question["REACTIVATION_DECISION"]["unmapped_current_detections"])},
        "candidate_identity_mapping": dict(by_question["REACTIVATION_DECISION"]),
        "candidate_level_return": {"available": False, "reason": "Candidate IDs and scores are available, but no per-candidate long-horizon return is present in the native trace. Identity mapping is an audit signal only."},
        "policy": {"action_labels_copied_to_candidate_supervision": False, "gt_used_only_for_offline_audit": True, "authorizes_training": False, "authorizes_full_h8": False},
        "events": all_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--video-id", type=int, default=1)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--identity-consistency-threshold", type=float, default=0.8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 < args.iou_threshold <= 1.0 or not 0.0 < args.identity_consistency_threshold <= 1.0:
        raise ValueError("thresholds must be in (0, 1]")
    report = audit(args.trace.resolve(), args.annotations.resolve(), args.cache_index.resolve(), int(args.video_id), float(args.iou_threshold), float(args.identity_consistency_threshold))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "seed_initialization", "by_question", "candidate_level_return")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
