#!/usr/bin/env python3
"""Run the dev-only GMT baseline and an oracle current-choice upper bound.

The oracle hook is attached to GMT's existing audit-only pre-commit callback.
It can replace the current assignment for this decision, but it never edits
history, relocates an earlier observation, reads a future frame, or performs a
state action.  Ground truth is used only to choose the current dev decision.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.modeling import build_model

from centernet.config import add_centernet_config
from gtr.config import add_gtr_config
from gtr.data.custom_build_augmentation import build_custom_augmentation
from gtr.data.gtr_dataset_dataloader import build_gtr_test_loader
from gtr.data.gtr_dataset_mapper import GMTDatasetMapper
from gtr.data.datasets.mot import _get_builtin_metadata, register_mot_instances


def scene_name(file_name: str) -> str:
    match = re.search(r"([^/\\]+)_View\d+", str(file_name))
    if not match:
        raise ValueError(f"cannot derive scene from {file_name!r}")
    return match.group(1)


def image_scene_view(file_name: str) -> tuple[str, int]:
    match = re.search(r"([^/\\]+)_View(\d+)", str(file_name))
    if not match:
        raise ValueError(f"cannot derive scene/view from {file_name!r}")
    return match.group(1), int(match.group(2)) - 1


def load_split(path: Path) -> dict:
    return json.loads(path.read_text())


def write_dev_annotation(source_path: Path, split: dict, output: Path) -> tuple[dict, set[str]]:
    source = json.loads(source_path.read_text())
    dev_scenes = set(split["scenes_by_split"]["JEV-dev"])
    images = [row for row in source["images"] if scene_name(row["file_name"]) in dev_scenes]
    image_ids = {int(row["id"]) for row in images}
    video_ids = {int(row["video_id"]) for row in images}
    payload = {
        "images": images,
        "annotations": [row for row in source["annotations"] if int(row["image_id"]) in image_ids],
        "categories": source["categories"],
        # ``load_video_divo_json`` in this release indexes the video list by
        # ``video_id - 1``.  Keep the complete metadata table (no images or
        # annotations are added) so sparse dev video IDs remain valid.
        "videos": source["videos"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload))
    return payload, dev_scenes


def load_labels(root: Path, dev_scenes: set[str]) -> dict[tuple[str, int, int, int], dict]:
    labels = {}
    for path in sorted((root / "labeled_records").glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        scene = str(payload["scene"])
        if scene not in dev_scenes:
            continue
        for row in payload["records"]:
            labels[(scene, int(row["frame_index"]), int(row["view_index"]), int(row["detection_index"]))] = row
    return labels


def build_cfg(config_file: Path, weight: Path, output: Path):
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_gtr_config(cfg)
    cfg.merge_from_file(str(config_file))
    cfg.merge_from_list([
        "DATASETS.TEST", "('VISION_decision_dev',)",
        "MODEL.WEIGHTS", str(weight),
        "OUTPUT_DIR", str(output),
    ])
    cfg.freeze()
    return cfg


def install_context_and_policy(model, labels, enabled: bool, k: int):
    original = model.run_global_tracker_plus
    state = {"scene": None, "call_count": 0, "view_num": None, "items": {}}

    def oracle_action(*, context, instances, track_ids, unique_ids, traj_score, support, id_count_dict):
        scene = str(context["scene"])
        frame = int(context["frame_index"])
        view = int(context["view_index"])
        replacement = track_ids.clone()
        desired = {}
        candidate_set_by_det = {}
        for det in range(len(track_ids)):
            row = labels.get((scene, frame, view, det))
            if row is None or row.get("gt_id_for_supervision") is None:
                continue
            ordered = sorted(
                row["candidates"],
                key=lambda item: (-float(item["traj_score"]), int(item["global_id"])),
            )
            top_ids = {int(item["global_id"]) for item in ordered[:k]}
            target_type = str(row.get("target_type"))
            target = row.get("target_global_id")
            if target_type == "NEW":
                desired[det] = None  # NEW is represented by -1 pre-commit.
            elif target is not None and int(target) in top_ids:
                desired[det] = int(target)
            candidate_set_by_det[det] = top_ids

        # Apply current-view choices in detection order while preserving the
        # one-to-one existing-ID invariant.  Unsupervised or candidate-miss
        # rows retain the released assignment.
        used: set[int] = set()
        for det in range(len(track_ids)):
            if det in desired:
                target = desired[det]
                if target is None:
                    replacement[det] = -1
                elif target not in used:
                    replacement[det] = target
                    used.add(target)
                else:
                    replacement[det] = -1
            else:
                original_id = int(track_ids[det].item())
                if original_id >= 0 and original_id not in used:
                    used.add(original_id)
                elif original_id >= 0 and original_id in used:
                    replacement[det] = -1
        return {"track_ids": replacement}

    def wrapped(bound_self, view_num, instances, asso_output, pred_boxes, k_index,
                id_count, id_count_dict, id_reid_dict, instances_old, view):
        if state["scene"] is None:
            raise RuntimeError("oracle context was not initialized for the scene")
        frame = 1 + state["call_count"] // max(int(view_num), 1)
        view_index = state["call_count"] % max(int(view_num), 1)
        state["call_count"] += 1
        item = state["items"].get((frame, view_index))
        if item is None:
            raise RuntimeError(f"missing replay context for {state['scene']} frame {frame} view {view_index}")
        model._gmt_audit_context = {
            "scene": state["scene"],
            "video_id": int(item.get("video_id", -1)),
            "image_id": int(item.get("image_id", -1)),
            "frame": int(item.get("frame_id", frame)),
            "frame_index": int(frame),
            "view": int(item.get("view_id", view_index + 1)),
            "view_index": int(view_index),
            "width": int(item.get("width", 0)),
            "height": int(item.get("height", 0)),
        }
        model._gmt_replay_action = oracle_action if enabled else None
        try:
            return original(view_num, instances, asso_output, pred_boxes, k_index,
                            id_count, id_count_dict, id_reid_dict, instances_old, view)
        finally:
            model._gmt_replay_action = None

    model.run_global_tracker_plus = types.MethodType(wrapped, model)

    def begin_scene(batch):
        scene = scene_name(batch[0]["file_name"])
        view_num = int(batch[0]["view_num"])
        view_frames = len(batch) // view_num
        state["scene"] = scene
        state["view_num"] = view_num
        state["call_count"] = 0
        state["items"] = {
            (frame, view): batch[view * view_frames + frame]
            for frame in range(view_frames)
            for view in range(view_num)
        }
        model._gmt_audit_context = None
        model._gmt_replay_action = None
        return scene

    return begin_scene


def prepare_eval_ground_truth(annotation: dict, scenes: set[str], root: Path) -> tuple[list[str], dict[str, int]]:
    seq_images: dict[str, list[dict]] = collections.defaultdict(list)
    for image in annotation["images"]:
        scene, view = image_scene_view(image["file_name"])
        if scene in scenes:
            seq_images[f"{scene}_View{view + 1}"].append(image)
    anns_by_image = collections.defaultdict(list)
    for ann in annotation["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)
    gt_root = root / "gt"
    seq_names = sorted(seq_images)
    lengths = {}
    for seq, images in seq_images.items():
        images = sorted(images, key=lambda row: int(row["frame_id"]))
        lengths[seq] = max(int(row["frame_id"]) for row in images)
        seq_dir = gt_root / seq
        (seq_dir / "gt").mkdir(parents=True, exist_ok=True)
        info = "\n".join([
            "[Sequence]", "name = " + seq, "imDir = img1", "frameRate = 30",
            "seqLength = " + str(lengths[seq]), "imWidth = " + str(images[0]["width"]),
            "imHeight = " + str(images[0]["height"]), "imExt = .jpg", "",
        ])
        (seq_dir / "seqinfo.ini").write_text(info)
        lines = []
        for image in images:
            frame = int(image["frame_id"])
            for ann in anns_by_image[int(image["id"])]:
                x, y, w, h = [float(value) for value in ann["bbox"]]
                gid = int(ann.get("instance_id", -1))
                if gid < 0:
                    continue
                lines.append(f"{frame},{gid},{x:.6f},{y:.6f},{w:.6f},{h:.6f},1,-1,-1,-1")
        (seq_dir / "gt" / "gt.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
    seqmap = root / "seqmap.txt"
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqmap.write_text("name\n" + "\n".join(seq_names) + "\n")
    return seq_names, lengths


def write_tracker_outputs(predictions: dict[str, list[tuple[int, dict, object]]], root: Path, name: str, lengths: dict[str, int]):
    tracker_root = root / "trackers" / name / "data"
    tracker_root.mkdir(parents=True, exist_ok=True)
    grouped = collections.defaultdict(list)
    for seq, rows in predictions.items():
        for frame, item, instance in rows:
            boxes = instance.pred_boxes.tensor.detach().cpu().tolist()
            scores = instance.scores.detach().cpu().tolist()
            ids = instance.track_ids.detach().cpu().tolist()
            for box, score, track_id in zip(boxes, scores, ids):
                x1, y1, x2, y2 = [float(value) for value in box]
                grouped[seq].append(
                    f"{frame},{int(track_id)},{x1:.6f},{y1:.6f},{x2-x1:.6f},{y2-y1:.6f},{float(score):.6f},-1,-1,-1"
                )
    for seq in lengths:
        lines = sorted(grouped.get(seq, []), key=lambda row: (int(row.split(",", 1)[0]), int(row.split(",")[1])))
        (tracker_root / f"{seq}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def evaluate_tracker(root: Path, tracker_name: str):
    sys.path.insert(0, str(Path("TrackEval").resolve()))
    import trackeval

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({
        "USE_PARALLEL": False,
        "PRINT_ONLY_COMBINED": True,
        "DISPLAY_LESS_PROGRESS": True,
        "TIME_PROGRESS": False,
        "PLOT_CURVES": False,
        "OUTPUT_SUMMARY": False,
        "OUTPUT_DETAILED": False,
    })
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": str(root / "gt"),
        "TRACKERS_FOLDER": str(root / "trackers"),
        "TRACKERS_TO_EVAL": [tracker_name],
        "BENCHMARK": "vision",
        "SPLIT_TO_EVAL": "dev",
        "SKIP_SPLIT_FOL": True,
        "SEQMAP_FILE": str(root / "seqmap.txt"),
        "DO_PREPROC": False,
        "OUTPUT_FOLDER": str(root / "results"),
    })
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    result, messages = trackeval.Evaluator(eval_config).evaluate([dataset], metrics)
    combined = result[dataset.get_name()][tracker_name]["COMBINED_SEQ"]["pedestrian"]
    return {
        "HOTA": float(np.mean(combined["HOTA"]["HOTA"]) * 100.0),
        "AssA": float(np.mean(combined["HOTA"]["AssA"]) * 100.0),
        "MOTA": float(combined["CLEAR"]["MOTA"] * 100.0),
        "IDF1": float(combined["Identity"]["IDF1"] * 100.0),
        "raw_message": messages[dataset.get_name()][tracker_name],
    }


def run_pass(cfg, loader, model, begin_scene, output_root: Path, name: str, lengths: dict[str, int]):
    predictions = collections.defaultdict(list)
    model.eval()
    with torch.no_grad():
        for index, batch in enumerate(loader):
            scene = begin_scene(batch)
            outputs, view_num = model(batch)
            view_num = int(view_num)
            view_frames = len(batch) // view_num
            # GMT returns postprocessed outputs in frame-major order while
            # the loader supplies the scene in view-major order.
            ordered_items = [
                batch[view * view_frames + frame]
                for frame in range(view_frames)
                for view in range(view_num)
            ]
            for item, output in zip(ordered_items, outputs):
                seq = f"{scene}_View{int(item.get('view_id', 1))}"
                predictions[seq].append((int(item.get("frame_id", 0)), item, output["instances"].to("cpu")))
            del outputs
            print(json.dumps({"pass": name, "batch": index, "scene": scene}, sort_keys=True), flush=True)
    write_tracker_outputs(predictions, output_root, name, lengths)
    return evaluate_tracker(output_root, name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/VISION_test.yaml"))
    parser.add_argument("--weight", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="5")
    parser.add_argument("--k", type=int, required=True)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["GMT_ASSOC_REPLAY_LOAD"] = "1"
    os.environ.pop("GMT_ASSOC_REPLAY_DUMP", None)
    os.environ["GMT_ASSOC_REPLAY_DIR"] = str((args.records / "observation_cache").resolve())
    split = load_split(args.split)
    annotation, dev_scenes = write_dev_annotation(args.annotation, split, args.output / "dev_annotation.json")
    register_mot_instances(
        "VISION_decision_dev", _get_builtin_metadata(),
        str((args.output / "dev_annotation.json").resolve()),
        "datasets/VisionTrack/images/train",
    )
    labels = load_labels(args.records, dev_scenes)
    if not labels:
        raise RuntimeError("no labeled dev records found; collect and label the full train set first")
    cfg = build_cfg(args.config, args.weight, args.output)
    model = build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(str(args.weight), resume=False)
    mapper = GMTDatasetMapper(cfg, False, augmentations=build_custom_augmentation(cfg, False))
    loader = build_gtr_test_loader(cfg, "VISION_decision_dev", mapper)
    seq_names, lengths = prepare_eval_ground_truth(annotation, dev_scenes, args.output / "eval")

    begin_baseline = install_context_and_policy(model, labels, enabled=False, k=args.k)
    baseline = run_pass(cfg, loader, model, begin_baseline, args.output / "eval", "gmt_baseline", lengths)

    # The loader is deterministic and can be iterated again.  Reinitialize
    # mutable class-level GMT state by rebuilding the model before oracle pass.
    model = build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(str(args.weight), resume=False)
    loader = build_gtr_test_loader(cfg, "VISION_decision_dev", mapper)
    begin_oracle = install_context_and_policy(model, labels, enabled=True, k=args.k)
    oracle = run_pass(cfg, loader, model, begin_oracle, args.output / "eval", "oracle_choice", lengths)

    payload = {
        "format": "gmt-oracle-choice-headroom-v1",
        "split": str(args.split.resolve()),
        "annotation": str(args.annotation.resolve()),
        "scenes": sorted(dev_scenes),
        "K": int(args.k),
        "visiontrack_test_used": False,
        "baseline": baseline,
        "oracle_choice": oracle,
        "delta_oracle_minus_baseline": {key: oracle[key] - baseline[key] for key in ("HOTA", "AssA", "MOTA", "IDF1")},
        "gate": {
            "AssA_plus_0.5_or_IDF1_plus_0.5": bool(
                oracle["AssA"] >= baseline["AssA"] + 0.5 or oracle["IDF1"] >= baseline["IDF1"] + 0.5
            ),
            "MOTA_drop_at_most_1": bool(oracle["MOTA"] >= baseline["MOTA"] - 1.0),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "ORACLE_CHOICE_HEADROOM.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Oracle Choice headroom (JEV-dev)", "",
        "This is an upper bound for a current-choice layer.  It uses train GT only",
        "to replace the current choice when the target is in frozen Top-K; it does",
        "not repair history, quarantine, commit, delay, or read future frames.", "",
        f"* frozen K: `{args.k}`", f"* scenes: `{', '.join(sorted(dev_scenes))}`", "",
        "| metric | GMT baseline | Oracle Choice | delta |", "|---|---:|---:|---:|",
    ]
    for key in ("HOTA", "AssA", "IDF1", "MOTA"):
        lines.append(f"| {key} | {baseline[key]:.6f} | {oracle[key]:.6f} | {oracle[key]-baseline[key]:+.6f} |")
    lines += ["", f"* practical headroom gate: `{payload['gate']['AssA_plus_0.5_or_IDF1_plus_0.5']}`",
              f"* MOTA guard: `{payload['gate']['MOTA_drop_at_most_1']}`", ""]
    (args.output / "ORACLE_CHOICE_HEADROOM.md").write_text("\n".join(lines))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
