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
PILOT = Path("/home/liuyeqiang/WWW_jev_full_h8_runtime/pilot")
VIDEO_ID = 8
CHECKPOINT = Path("/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth")
CONFIG = ROOT / "configs/VISION_test.yaml"
CACHE = Path("/data1/liuyeqiang/WWW/outputs/research_final_v2/off/perception_cache_train")
TRACE = Path("/home/liuyeqiang/WWW_jev_full_h8_runtime/partition/trace_by_video/video_08.jsonl")
ANNOTATIONS = Path("/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json")
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
    record_path = PILOT / "PILOT_TRACKING_TEST.jsonl"
    with record_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            context = record["state"]["online_context"]
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
    from gtr.modeling.jev_state import build_state_features

    return (
        build_formal_gmt_engine,
        MutableGMTState,
        FrozenPerceptionCache,
        JEVRuntimePolicy,
        build_controller_from_checkpoint,
        build_state_features,
    )


def history_track_length(state, track_id: int | None) -> int:
    """Count one track's occurrences in the bounded mutable association history."""
    if track_id is None:
        return 1
    count = 0
    for item in state.association_history:
        assignments = item.get("assignments", {})
        count += sum(int(value) == int(track_id) for value in assignments.values())
    return max(1, int(count))


def formal_memory_count(state, track_id: int) -> int:
    """Mirror GTRRCNN id_reid_dict length: initial observation plus writes."""
    return max(1, 1 + len(state.memory.get(int(track_id), ())))

def choose(policy, feature, question, legal, off_action):
    return policy.decide(feature, question, legal, off_action=off_action).committed_action


def canonical_feature_or_fallback(record, fallback: torch.Tensor):
    """Use the exact formal trace feature vector when this event is present.

    The H=8 policy dataset is trained on the state vector emitted by the
    formal GMT trace.  Reconstructing that vector from a replay proposal is
    unsafe: the native model's window length, candidate entropy, track length,
    and memory counters depend on the batching/commit boundary.  The pilot
    record already carries that online-only vector, so use it for parity and
    expose a fallback counter for any event not represented in the snapshot.
    """

    if record is not None:
        values = record.get("state", {}).get("feature_vector")
        if isinstance(values, list) and len(values) == int(fallback.numel()):
            tensor = torch.tensor(values, dtype=torch.float32)
            if torch.isfinite(tensor).all():
                return tensor, "CANONICAL_TRACE_RECORD"
    return fallback, "RECONSTRUCTED_FALLBACK"


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
    feature_source_mode,
):
    controller = (
        build_controller_from_checkpoint(checkpoint_path, device="cpu")
        if checkpoint_path is not None
        else None
    )
    policy = JEVRuntimePolicy("jev", controller) if controller is not None else JEVRuntimePolicy("off")
    engine = build_formal_gmt_engine(
        config_file=CONFIG,
        checkpoint=CHECKPOINT,
        device="cpu",
        view_num=2,
        history_limit=80,
    )
    # DetectionCheckpointer can restore CUDA-indexed embedding buffers even
    # when the model was initially constructed with ``device=cpu``.  This is
    # local to the pilot CPU replay; formal GPU workers are separate
    # processes and are not changed.
    engine.association_fn.model.cpu()
    engine.association_fn.model.eval()
    cache = FrozenPerceptionCache(CACHE)
    keys = [key for key in cache.keys() if int(key[0]) == VIDEO_ID]
    keys.sort(key=lambda key: (int(key[1]), int(key[2])))
    state = MutableGMTState()
    predictions = []
    decisions = []
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
    }
    runtime_model = engine.association_fn.model
    threshold = float(runtime_model.overlap_thresh)
    can_reassociate = int(runtime_model.jev_max_reassociate) > 0
    memory_enabled = bool(runtime_model.with_bank)
    with_iou = bool(runtime_model.with_iou)
    not_mult_thresh = bool(runtime_model.not_mult_thresh)
    counts["runtime_feature_records"] = 0
    counts["trace_debug_feature_records"] = 0
    counts["feature_parity_records"] = 0
    counts["feature_parity_max_abs_error"] = 0.0

    for key_index, key in enumerate(keys):
        payload = cache.load(*key)
        video_id, frame, view = [int(value) for value in key]
        proposal = engine.propose(payload, state)
        track_ids, scores = proposal.track_ids, proposal.scores
        actions: Dict[int, str] = {}
        memories: Dict[int, str] = {}
        events_here = by_key.get((video_id, frame, view), ())
        trace_off = {
            (str(event.get("question")), int(event.get("context", {}).get("detection_index", -1))): str(
                event.get("off_action") or event.get("proposed_action")
            )
            for event in events_here
            if event.get("context", {}).get("detection_index") is not None
        }
        window_length = max(1, len(state.association_history) + 1)

        def select_feature(record, runtime_feature):
            if record is not None:
                values = record.get("state", {}).get("feature_vector")
                if isinstance(values, list) and len(values) == int(runtime_feature.numel()):
                    trace_feature = torch.tensor(values, dtype=torch.float32)
                    if torch.isfinite(trace_feature).all():
                        error = float((trace_feature - runtime_feature.cpu()).abs().max().item())
                        counts["feature_parity_records"] += 1
                        counts["feature_parity_max_abs_error"] = max(
                            float(counts["feature_parity_max_abs_error"]), error
                        )
                        if feature_source_mode == "trace_debug":
                            counts["trace_debug_feature_records"] += 1
                            return trace_feature
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
                candidate_entropy = 0.0
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
                probabilities = torch.softmax(scores[row], dim=0)
                candidate_entropy = float(
                    (-(probabilities * probabilities.clamp_min(1e-8).log()).sum()).item()
                )
                track_length = history_track_length(state, first_id)
                legacy_threshold = threshold if not_mult_thresh else threshold * float(track_length)
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
                candidate_entropy=candidate_entropy,
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
            action = trace_off.get(("MATCH_DECISION", row), off_action) if name == "gmt_off" else choose(
                policy, feature, "MATCH_DECISION", legal, off_action
            )
            if action not in legal:
                action = off_action
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
        resolution = engine.resolve_actions(payload, state, actions=actions)
        final_existing = resolution["existing_track_ids"]

        # Phase 2: MEMORY decisions are computed only for rows committed to an
        # existing identity, matching production GTRRCNN commit order.
        for row in range(int(scores.shape[0])):
            track_id = final_existing.get(row)
            if track_id is None:
                continue
            score = float(scores[row].max().item()) if scores.shape[1] else 0.0
            memory_count = formal_memory_count(state, int(track_id))
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
            action = trace_off.get(("MEMORY_DECISION", row), off_action) if name == "gmt_off" else choose(
                policy, feature, "MEMORY_DECISION", legal, off_action
            )
            if action not in legal:
                action = off_action
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
        result = engine.step(payload, state, actions=actions, memory_actions=memories)
        image = image_lookup[(VIDEO_ID, view + 1, frame + 1)]
        committed = result["committed_track_ids"]
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--feature-source", choices=("runtime", "trace_debug"), default="runtime")
    args = parser.parse_args()
    for path in (CHECKPOINT, CONFIG, CACHE, TRACE, ANNOTATIONS):
        if not path.exists():
            raise FileNotFoundError(path)
    annotations, subset_annotation, image_lookup, by_key, records = load_inputs()
    modules = load_runtime_modules()
    build_formal_gmt_engine, MutableGMTState, FrozenPerceptionCache, JEVRuntimePolicy, build_controller_from_checkpoint, build_state_features = modules
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
        "device": args.device,
        "methods": {},
    }
    for name, relative in METHODS.items():
        checkpoint_path = None if relative is None else PILOT / relative
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
            feature_source_mode=args.feature_source,
        )
        json_write(PILOT / "PILOT_TRACKING_RAW.json", raw)
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
            "gmt_off_compared_records": methods["gmt_off"]["action_counts"]["feature_parity_records"],
            "gmt_off_max_abs_error": methods["gmt_off"]["action_counts"]["feature_parity_max_abs_error"],
            "pass": (
                methods["gmt_off"]["action_counts"]["feature_parity_records"] > 0
                and methods["gmt_off"]["action_counts"]["feature_parity_max_abs_error"] <= 1e-6
            ),
        },
        "pilot_verdict": (
            "TRACE_DEBUG_ONLY"
            if args.feature_source == "trace_debug"
            else (
                "PILOT_GO_FOR_FULL_H8_CONTINUATION"
                if (
                    methods["gmt_off"]["action_counts"]["feature_parity_records"] > 0
                    and methods["gmt_off"]["action_counts"]["feature_parity_max_abs_error"] <= 1e-6
                    and methods["jev"]["metrics"]["AssA"] >= baseline["AssA"]
                    and methods["jev"]["metrics"]["IDSW"] <= baseline["IDSW"]
                )
                else "PILOT_FAIL_RUNTIME_STATE_PARITY_OR_TRACKING"
            )
        ),
        "previous_manual_feature_result": "SUPERSEDED_INVALID_FEATURE_SCHEMA",
        "interpretation": "Pilot-only matched-sequence evidence; not a final paper result and not a substitute for full VISION_test evaluation.",
    }
    json_write(PILOT / "PILOT_TRACKING_THREE_WAY.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()