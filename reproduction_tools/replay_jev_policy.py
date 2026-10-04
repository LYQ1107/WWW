"""Apply a trained online JEV/baseline policy to a frozen GMT trace.

This is a proposal-replay evaluation: detector and GMT proposals remain the
same as the OFF run, and only typed commit actions are changed.  It is useful
for a cheap matched-policy comparison before the full GPU JEV inference.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

from gtr.modeling.jev_runtime import JEVRuntimePolicy, build_controller_from_checkpoint


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def xywh_to_xyxy(box: Sequence[float]):
    x, y, w, h = (float(value) for value in box[:4])
    return x, y, x + w, y + h


def iou(left, right):
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    la = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    ra = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = la + ra - intersection
    return intersection / union if union else 0.0


def load_images(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    lookup = {}
    for image in payload["images"]:
        lookup[(int(image["video_id"]), int(image["view_id"]), int(image["frame_id"]))] = image
    return lookup


def event_image(event, image_lookup):
    context = event.get("context", {})
    video_id = int(context.get("video_id", -1))
    frame = int(context.get("frame", 0))
    view = int(context.get("view", 0)) + 1
    image = image_lookup.get((video_id, view, frame))
    if image is None:
        image = image_lookup.get((video_id, int(context.get("view", 0)), frame))
    return image


def scaled_event_box(event, image):
    box = event.get("context", {}).get("bbox_xyxy")
    if not box or image is None:
        return None
    model_size = event.get("context", {}).get("model_image_size")
    if model_size and len(model_size) == 2:
        model_h, model_w = float(model_size[0]), float(model_size[1])
        return [
            float(box[0]) * float(image["width"]) / model_w,
            float(box[1]) * float(image["height"]) / model_h,
            float(box[2]) * float(image["width"]) / model_w,
            float(box[3]) * float(image["height"]) / model_h,
        ]
    return [float(value) for value in box]


def best_prediction(predictions, target_box, proposal_id=None):
    if target_box is None or not predictions:
        return None
    ranked = []
    for index, prediction in enumerate(predictions):
        overlap = iou(target_box, xywh_to_xyxy(prediction["bbox"]))
        exact = int(proposal_id is not None and int(prediction.get("track_id", -1)) == int(proposal_id))
        ranked.append((overlap, exact, index))
    overlap, _exact, index = max(ranked)
    return index if overlap >= 0.05 else None


def load_oracle_actions(path: Path) -> Dict[Tuple[int, int, str], str]:
    """Load offline branch winners keyed by the immutable trace event identity."""
    actions: Dict[Tuple[int, int, str], str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            state = record.get("state", {})
            context = state.get("online_context", {})
            if not isinstance(context, Mapping):
                raise ValueError(f"oracle record {line_number} has no online_context")
            if "event_order" not in context or "video_id" not in context:
                raise ValueError(f"oracle record {line_number} has no event identity")
            best_actions = record.get("best_actions")
            if not isinstance(best_actions, list) or not best_actions:
                raise ValueError(f"oracle record {line_number} has no best action")
            key = (
                int(context["video_id"]),
                int(context["event_order"]),
                str(record["question_type"]),
            )
            action = str(best_actions[0])
            previous = actions.get(key)
            if previous is not None and previous != action:
                raise ValueError(f"conflicting oracle actions for event {key}")
            actions[key] = action
    if not actions:
        raise ValueError(f"oracle dataset is empty: {path}")
    return actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--controller", type=Path)
    parser.add_argument(
        "--oracle-dataset",
        type=Path,
        help="offline counterfactual JSONL; uses best_actions and is never valid online",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    predictions_path = args.predictions.resolve()
    trace_path = args.trace.resolve()
    annotations_path = args.annotations.resolve()
    controller_path = args.controller.resolve() if args.controller else None
    oracle_dataset_path = args.oracle_dataset.resolve() if args.oracle_dataset else None
    if (controller_path is None) == (oracle_dataset_path is None):
        raise ValueError("provide exactly one of --controller or --oracle-dataset")
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise ValueError("predictions must be a JSON list")
    image_lookup = load_images(annotations_path)
    by_image: Dict[int, List[Dict[str, Any]]] = {}
    for prediction in predictions:
        by_image.setdefault(int(prediction["image_id"]), []).append(prediction)
    maximum_id = max([int(item.get("track_id", 0)) for item in predictions] + [0])
    next_id = maximum_id + 1
    controller = (
        build_controller_from_checkpoint(controller_path, device="cpu")
        if controller_path is not None
        else None
    )
    policy = JEVRuntimePolicy("jev", controller) if controller is not None else None
    oracle_actions = load_oracle_actions(oracle_dataset_path) if oracle_dataset_path else None
    counts = {"events": 0, "changed": 0, "unmatched": 0, "oracle_missing": 0}
    with trace_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            question = event.get("question")
            if question not in {"MATCH_DECISION", "REACTIVATION_DECISION"}:
                continue
            image = event_image(event, image_lookup)
            if image is None:
                counts["unmatched"] += 1
                continue
            image_predictions = by_image.get(int(image["id"]), [])
            context = event.get("context", {})
            proposal_id = context.get("proposal_track_id")
            if question == "REACTIVATION_DECISION":
                proposal_id = context.get("track_id")
            index = best_prediction(image_predictions, scaled_event_box(event, image), proposal_id)
            if index is None:
                counts["unmatched"] += 1
                continue
            legal = list(event["legal_actions"])
            off_action = event.get("off_action") or event.get("proposed_action")
            if oracle_actions is not None:
                key = (
                    int(context.get("video_id", -1)),
                    int(context.get("event_order", -1)),
                    str(question),
                )
                action = oracle_actions.get(key)
                if action is None:
                    counts["oracle_missing"] += 1
                    raise ValueError(f"oracle dataset has no action for trace event {key}")
                if action not in legal:
                    raise ValueError(f"oracle action {action} is not legal for event {key}")
            else:
                assert policy is not None
                decision = policy.decide(
                    torch.tensor(event["state_feature_vector"], dtype=torch.float32),
                    question,
                    legal,
                    off_action=off_action,
                    context=context,
                )
                action = decision.committed_action
            if action == "ACCEPT_CURRENT":
                target_id = context.get("proposal_track_id")
            elif action == "REASSOCIATE":
                target_id = context.get("alternate_track_id")
            elif action == "REACTIVATE_OLD":
                target_id = context.get("track_id")
            else:
                target_id = None
            if target_id is None or action == "START_NEW":
                target_id = next_id
                next_id += 1
            target_id = int(target_id)
            current_id = int(image_predictions[index].get("track_id", -1))
            if current_id != target_id:
                image_predictions[index]["track_id"] = target_id
                counts["changed"] += 1
            counts["events"] += 1
    result = []
    for prediction in predictions:
        result.append(prediction)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result), encoding="utf-8")
    manifest = {
        "status": "PASS",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_predictions": str(predictions_path),
        "base_predictions_sha256": sha256(predictions_path),
        "trace": str(trace_path),
        "trace_sha256": sha256(trace_path),
        "annotations": str(annotations_path),
        "annotations_sha256": sha256(annotations_path),
        "controller": str(controller_path) if controller_path else None,
        "controller_sha256": sha256(controller_path) if controller_path else None,
        "oracle_dataset": str(oracle_dataset_path) if oracle_dataset_path else None,
        "oracle_dataset_sha256": sha256(oracle_dataset_path) if oracle_dataset_path else None,
        "counts": counts,
        "policy_mode": "offline_oracle" if oracle_actions is not None else "jev",
        "future_gt_access": oracle_actions is not None,
        "evaluation_policy": (
            "offline best-action replay from frozen-trace counterfactual branches; never online"
            if oracle_actions is not None
            else "frozen GMT proposal replay; no detector or GMT forward recomputed"
        ),
    }
    output.with_suffix(output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
