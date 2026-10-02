#!/usr/bin/env python3
"""Collect strictly-online GMT candidate states and build offline labels.

The collector wraps ``run_global_tracker_plus`` without changing its decision.
It records the association state immediately before the released Hungarian /
threshold choice and attaches the chosen ID only after the original method
returns.  No GT or future frame is visible to the collector.  The ``label``
subcommand uses only VisionTrack *train* annotations after collection to add
supervision targets and candidate-coverage metadata.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import itertools
import json
import math
import os
import re
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.modeling import build_model

from centernet.config import add_centernet_config
from gtr.config import add_gtr_config
from gtr.data.custom_build_augmentation import build_custom_augmentation
from gtr.data.gtr_dataset_dataloader import build_gtr_test_loader
from gtr.data.gtr_dataset_mapper import GMTDatasetMapper
from gtr.data.datasets.mot import _get_builtin_metadata, register_mot_instances
from scipy.optimize import linear_sum_assignment


HISTORY_FEATURE_NAMES = (
    "normalized_score", "support", "max_asso", "mean_asso", "std_asso",
    "observation_count", "distinct_view_count", "time_since_last_seen",
    "same_view_time_since_last_seen", "age", "current_to_all_history_cosine",
    "current_to_recent_history_cosine", "history_feature_variance",
    "detector_score_mean", "detector_score_std",
)


def scene_name(file_name: str) -> str:
    match = re.search(r"([^/\\]+)_View\d+", str(file_name))
    if not match:
        raise ValueError(f"cannot derive scene from {file_name!r}")
    return match.group(1)


def bbox_norm(inst, index: int) -> list[float]:
    box = inst.pred_boxes.tensor[index].detach().float().cpu()
    width = max(float(inst.image_size[1]), 1.0)
    height = max(float(inst.image_size[0]), 1.0)
    x1, y1, x2, y2 = [float(x) for x in box]
    return [x1 / width, y1 / height, max(0.0, x2 - x1) / width,
            max(0.0, y2 - y1) / height]


def cosine_stats(current: torch.Tensor, history: torch.Tensor) -> tuple[float, float, float]:
    if history.numel() == 0:
        return 0.0, 0.0, 0.0
    current = current.float().reshape(1, -1)
    history = history.float()
    sims = F.cosine_similarity(history, current.expand_as(history), dim=1, eps=1e-8)
    return float(sims.mean()), float(sims[-min(5, len(sims)):].mean()), float(history.var(dim=0, unbiased=False).mean())


class OnlineDecisionRecorder:
    """Capture pre-choice candidate states while preserving released GMT."""

    def __init__(self, output_root: Path, save_observations: bool = True):
        self.output_root = Path(output_root)
        self.raw_root = self.output_root / "raw_records"
        self.raw_root.mkdir(parents=True, exist_ok=True)
        self.save_observations = bool(save_observations)
        self.current_scene: str | None = None
        self.current_items: dict[tuple[int, int], dict] = {}
        self._scene_call_count = 0
        self.records: list[dict] = []
        self._original = None
        self._model = None

    def install(self, model) -> None:
        self._model = model
        self._original = model.run_global_tracker_plus

        def wrapped(bound_self, view_num, instances, asso_output, pred_boxes, k,
                    id_count, id_count_dict, id_reid_dict, instances_old, view):
            if self.current_scene is None:
                raise RuntimeError("recorder scene must be set before model inference")
            # ``sliding_inference_GMT`` calls this method in frame-major order
            # for all views, beginning at frame 1 (frame 0 is initialized by
            # ``run_first_tracker_plus``).  Keep this context outside the
            # model tensors so labels cannot influence the online decision.
            frame_index = 1 + self._scene_call_count // max(int(view_num), 1)
            view_index = self._scene_call_count % max(int(view_num), 1)
            self._scene_call_count += 1
            pre = self._pre_choice(
                bound_self, view_num, instances, asso_output, k, view,
                frame_index=frame_index, view_index=view_index,
            )
            result = self._original(view_num, instances, asso_output, pred_boxes, k,
                                    id_count, id_count_dict, id_reid_dict,
                                    instances_old, view)
            if pre:
                current = result[0][k]
                for item in pre:
                    det = item.pop("_det_index")
                    item["gmt_choice_id"] = int(current.track_ids[det].item())
                    item["current_global_id_after_choice"] = int(current.track_ids[det].item())
                    self.records.append(item)
            return result

        model.run_global_tracker_plus = types.MethodType(wrapped, model)

    def _pre_choice(self, model, view_num, instances, asso_output, k, view,
                    frame_index: int, view_index: int):
        if self.current_scene is None:
            raise RuntimeError("recorder scene must be set before model inference")
        n_t = [len(x) for x in instances]
        if not n_t or k >= len(n_t):
            return []
        n_k = n_t[k]
        n_p = sum(n_t) - n_k
        if n_p <= 0:
            return []
        split = asso_output[-1].split(n_t[:-1], dim=1)
        activated = model.roi_heads._activate_asso(split)
        asso_nonk = torch.cat(activated, dim=1)
        ids = torch.cat([x.track_ids for t, x in enumerate(instances) if t != k], dim=0).view(n_p)
        unique_ids = torch.unique(ids)
        if not len(unique_ids):
            return []
        id_inds = (unique_ids[None, :] == ids[:, None]).float()
        traj = torch.mm(asso_nonk, id_inds)
        support = id_inds.sum(dim=0)
        normalized = traj / support.clamp_min(1e-6)[None, :]
        n_candidates = int(len(unique_ids))
        option_order = torch.argsort(normalized, dim=1, descending=True)
        # The released code uses the raw traj score for Hungarian.  Record the
        # exact candidate set and score order, then let the original method
        # perform the actual choice.
        rows = []
        current_frame = int(frame_index)
        current_view = int(view_index)
        item = self.current_items.get((current_frame, current_view))
        if item is None:
            raise RuntimeError(
                f"missing batch metadata for {self.current_scene} "
                f"frame={current_frame} view={current_view}"
            )
        for det in range(n_k):
            current_feature = instances[k].reid_features[det]
            current_score = float(instances[k].scores[det].detach().cpu())
            current_candidates = []
            for j, pid_tensor in enumerate(unique_ids):
                pid = int(pid_tensor.item())
                columns = torch.where(id_inds[:, j] > 0)[0]
                history_features = []
                history_scores = []
                history_positions = []
                history_views = []
                for col in columns.tolist():
                    # Map a flattened non-current index back to its source row.
                    flat = col
                    row_index = 0
                    for t, count in enumerate(n_t):
                        if t == k:
                            continue
                        if flat < count:
                            row_index = t
                            det_index = flat
                            break
                        flat -= count
                    source = instances[row_index]
                    history_features.append(source.reid_features[det_index])
                    history_scores.append(float(source.scores[det_index].detach().cpu()))
                    history_positions.append(row_index)
                    history_views.append(row_index % max(int(view_num), 1))
                feats = torch.stack(history_features) if history_features else current_feature.new_zeros((0, current_feature.numel()))
                all_cos, recent_cos, variance = cosine_stats(current_feature, feats)
                frame_positions = [p // max(int(view_num), 1) for p in history_positions]
                same_positions = [p // max(int(view_num), 1) for p, v in zip(history_positions, history_views) if v == current_view]
                last_frame = max(frame_positions) if frame_positions else -1
                last_same = max(same_positions) if same_positions else -1
                first_frame = min(frame_positions) if frame_positions else current_frame
                evidence = asso_nonk[det, columns] if len(columns) else asso_nonk.new_zeros((0,))
                current_candidates.append({
                    "global_id": pid,
                    "traj_score": float(traj[det, j].detach().cpu()),
                    "support": float(support[j].detach().cpu()),
                    "normalized_score": float(normalized[det, j].detach().cpu()),
                    "max_asso": float(evidence.max().detach().cpu()) if len(evidence) else 0.0,
                    "mean_asso": float(evidence.mean().detach().cpu()) if len(evidence) else 0.0,
                    "std_asso": float(evidence.std(unbiased=False).detach().cpu()) if len(evidence) else 0.0,
                    "observation_count": len(history_positions),
                    "distinct_view_count": len(set(history_views)),
                    "time_since_last_seen": max(0, current_frame - last_frame),
                    "same_view_time_since_last_seen": (max(0, current_frame - last_same) if last_same >= 0 else current_frame + 1),
                    "age": max(0, current_frame - first_frame),
                    "current_to_all_history_cosine": all_cos,
                    "current_to_recent_history_cosine": recent_cos,
                    "history_feature_variance": variance,
                    "detector_score_mean": float(np.mean(history_scores)) if history_scores else 0.0,
                    "detector_score_std": float(np.std(history_scores)) if history_scores else 0.0,
                })
            order = option_order[det].tolist()
            ordered_scores = normalized[det, option_order[det]].detach().float()
            probs = torch.softmax(ordered_scores, dim=0)
            entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum() / math.log(max(n_candidates, 2))).cpu())
            top1 = float(ordered_scores[0].cpu()) if n_candidates else 0.0
            top2 = float(ordered_scores[1].cpu()) if n_candidates > 1 else top1
            rows.append({
                "scene": self.current_scene,
                "frame_index": int(current_frame),
                "view_index": current_view,
                "detection_index": int(det),
                "candidate_ids": [int(x["global_id"]) for x in current_candidates],
                "candidates": current_candidates,
                "candidate_count": n_candidates,
                "top1_normalized_score": top1,
                "top2_normalized_score": top2,
                "top1_top2_margin": top1 - top2,
                "candidate_entropy": entropy,
                "current_score": current_score,
                "bbox_xywh_normalized": bbox_norm(instances[k], det),
                "image_id": int(item.get("image_id", -1)),
                "video_id": int(item.get("video_id", -1)),
                "frame_id": int(item.get("frame_id", current_frame)),
                "view_id": int(item.get("view_id", current_view + 1)),
                "width": int(item.get("width", instances[k].image_size[1])),
                "height": int(item.get("height", instances[k].image_size[0])),
                "bbox_xyxy": [
                    float(x) for x in instances[k].pred_boxes.tensor[det]
                    .detach().float().cpu().tolist()
                ],
                "current_fused_reid": instances[k].reid_features[det].detach().cpu().half(),
                "_det_index": int(det),
                "record_format": "gmt-online-decision-v1",
            })
        return rows

    def flush_scene(self, scene: str) -> Path:
        if self.current_scene != scene:
            raise RuntimeError(f"flush scene mismatch: {self.current_scene!r} vs {scene!r}")
        ordered = sorted(self.records, key=lambda x: (x["frame_index"], x["view_index"], x["detection_index"]))
        out = self.raw_root / f"{scene}.pt"
        torch.save({"format": "gmt-online-decision-v1", "scene": scene,
                    "record_count": len(ordered), "records": ordered}, out)
        self.records = []
        return out


def build_cfg(config_file: Path, opts: list[str]):
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_gtr_config(cfg)
    cfg.merge_from_file(str(config_file))
    cfg.merge_from_list(opts)
    cfg.freeze()
    return cfg


def register_scene_subset(dataset_name: str, scenes: set[str], output_root: Path) -> str:
    """Register a train-only scene subset for parallel collection workers."""
    source_path = Path("datasets/VisionTrack/annotations/train.json")
    source = json.loads(source_path.read_text())
    selected_images = [
        image for image in source["images"] if scene_name(image["file_name"]) in scenes
    ]
    image_ids = {int(image["id"]) for image in selected_images}
    payload = {
        "images": selected_images,
        "annotations": [
            ann for ann in source["annotations"] if int(ann["image_id"]) in image_ids
        ],
        "categories": source["categories"],
        # The released loader indexes this table with video_id - 1.  Retain
        # metadata for every video while selecting no images outside scenes.
        "videos": source["videos"],
    }
    path = output_root / "dataset_subsets" / f"{dataset_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    register_mot_instances(
        dataset_name, _get_builtin_metadata(), str(path.resolve()),
        "datasets/VisionTrack/images/train",
    )
    return dataset_name


def collect(args) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    replay_load = os.environ.get("GMT_DECISION_RECORDS_REPLAY_LOAD") == "1"
    if replay_load:
        os.environ["GMT_ASSOC_REPLAY_LOAD"] = "1"
        os.environ.pop("GMT_ASSOC_REPLAY_DUMP", None)
    else:
        os.environ.pop("GMT_ASSOC_REPLAY_LOAD", None)
        os.environ["GMT_ASSOC_REPLAY_DUMP"] = "1"
    os.environ["GMT_ASSOC_REPLAY_DIR"] = str((args.output / "observation_cache").resolve())
    dataset_name = "VISION_train"
    if args.scenes:
        dataset_name = register_scene_subset(
            f"VISION_decision_collect_{args.gpu}", set(args.scenes), args.output
        )
    cfg = build_cfg(args.config, ["DATASETS.TEST", f"('{dataset_name}',)",
                                  "MODEL.WEIGHTS", str(args.weight),
                                  "OUTPUT_DIR", str(args.output)])
    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).resume_or_load(str(args.weight), resume=False)
    mapper = GMTDatasetMapper(cfg, False, augmentations=build_custom_augmentation(cfg, False))
    loader = build_gtr_test_loader(cfg, dataset_name, mapper)
    recorder = OnlineDecisionRecorder(args.output)
    recorder.install(model)
    limit = args.scene_limit
    batches = itertools.islice(loader, limit) if limit is not None else loader
    for index, batch in enumerate(batches):
        scene = scene_name(batch[0]["file_name"])
        recorder.current_scene = scene
        recorder._scene_call_count = 0
        view_num = int(batch[0]["view_num"])
        view_frames = len(batch) // view_num
        # The data loader is view-major while GMT's internal ``instances``
        # list is frame-major.  Freeze the metadata mapping explicitly so
        # every recorded detection can be matched to train annotations later.
        recorder.current_items = {
            (frame, view): batch[view * view_frames + frame]
            for frame in range(view_frames)
            for view in range(view_num)
        }
        with torch.no_grad():
            model(batch)
        path = recorder.flush_scene(scene)
        print(json.dumps({"scene": scene, "records": int(torch.load(path, map_location="cpu", weights_only=False)["record_count"]), "path": str(path)}, sort_keys=True), flush=True)


def xyxy_iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (aa + ab - inter) if aa + ab - inter > 0 else 0.0


def image_boxes(annotation: dict):
    by_image = collections.defaultdict(list)
    for ann in annotation["annotations"]:
        x, y, w, h = [float(v) for v in ann["bbox"]]
        by_image[int(ann["image_id"])].append((int(ann.get("instance_id", -1)), (x, y, x + w, y + h)))
    return by_image


def match_group_to_gt(rows: list[dict], anns: list[tuple[int, tuple[float, ...]]]) -> list[int | None]:
    """One-to-one IoU matching for one current frame/view.

    The supervision protocol specifies Hungarian IoU matching.  Keeping this
    assignment at image level prevents two detections from receiving the same
    GT identity merely because both overlap its box.
    """
    if not rows or not anns:
        return [None] * len(rows)
    matrix = np.asarray([
        [xyxy_iou(row.get("bbox_xyxy", (0.0, 0.0, 0.0, 0.0)), gtbox) for _, gtbox in anns]
        for row in rows
    ], dtype=np.float64)
    row_indices, col_indices = linear_sum_assignment(-matrix)
    matched: list[int | None] = [None] * len(rows)
    for row_index, col_index in zip(row_indices.tolist(), col_indices.tolist()):
        if matrix[row_index, col_index] >= 0.5:
            matched[row_index] = int(anns[col_index][0])
    return matched


def label(args) -> None:
    split = json.loads(args.split.read_text())
    annotations = json.loads(args.annotation.read_text())
    images = {int(row["id"]): row for row in annotations["images"]}
    anns_by_image = image_boxes(annotations)
    split_by_scene = {scene: part for part, scenes in split["scenes_by_split"].items() for scene in scenes}
    rows_by_scene = {}
    for path in sorted((args.input / "raw_records").glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        scene = str(payload["scene"])
        if scene not in split_by_scene:
            raise ValueError(f"raw scene not in frozen split: {scene}")
        rows_by_scene[scene] = payload["records"]
    out_root = args.output / "labeled_records"
    out_root.mkdir(parents=True, exist_ok=True)
    for scene, rows in sorted(rows_by_scene.items()):
        counters = collections.defaultdict(collections.Counter)
        labeled = []
        ordered_rows = sorted(rows, key=lambda x: (x["frame_index"], x["view_index"], x["detection_index"]))
        # All detections in one camera view see the same previous-view
        # history.  Label the complete group first, then commit its matched
        # baseline choices to the helper history.
        for _, group_iter in itertools.groupby(
            ordered_rows, key=lambda x: (x["frame_index"], x["view_index"])
        ):
            group = list(group_iter)
            image_ids = {int(row.get("image_id", -1)) for row in group}
            if len(image_ids) != 1:
                raise ValueError(f"frame/view group contains multiple image IDs: {image_ids}")
            image_id = next(iter(image_ids))
            if image_id not in images:
                raise ValueError(f"record has no annotation image_id: {image_id}")
            matched_gids = match_group_to_gt(group, anns_by_image.get(image_id, []))
            for row, gid in zip(group, matched_gids):
                candidate_support = []
                for candidate in row["candidates"]:
                    pid = int(candidate["global_id"])
                    counts = counters[pid]
                    total = int(sum(counts.values()))
                    canonical = None
                    canonical_purity = 0.0
                    if counts:
                        canonical = max(
                            counts,
                            key=lambda gt: (
                                int(counts[gt]),
                                float(counts[gt]) / max(total, 1),
                                -int(gt),
                            ),
                        )
                        canonical_purity = float(counts[canonical]) / max(total, 1)
                    candidate_support.append((
                        pid, canonical, total, canonical_purity,
                        float(candidate["normalized_score"]),
                    ))
                correct = [x for x in candidate_support if gid is not None and x[1] == gid]
                if not correct:
                    target_type, target_id = "NEW", None
                else:
                    # Frozen rule from the protocol: past GT support, then
                    # historical purity, current normalized GMT score, lower
                    # numeric ID.  This is label construction only.
                    target_type, target_id = "EXISTING", max(
                        correct, key=lambda x: (x[2], x[3], x[4], -x[0])
                    )[0]
                row["partition"] = split_by_scene[scene]
                row["gt_id_for_supervision"] = gid
                row["target_type"] = target_type
                row["target_global_id"] = target_id
                row["candidate_miss"] = bool(
                    target_id is not None and target_id not in row["candidate_ids"]
                )
                row["baseline_choice_correct"] = bool(
                    (target_type == "NEW" and row["gmt_choice_id"] not in row["candidate_ids"])
                    or row["gmt_choice_id"] == target_id
                )
                row["baseline_is_new"] = bool(row["gmt_choice_id"] not in row["candidate_ids"])
                row["candidate_support_snapshot"] = {
                    str(pid): {str(k): int(v) for k, v in counters[pid].items()}
                    for pid in row["candidate_ids"]
                }
                labeled.append(row)
            # No current-view label affects another current-view decision.
            for row in group:
                if row["gt_id_for_supervision"] is not None:
                    counters[int(row["gmt_choice_id"])][int(row["gt_id_for_supervision"])] += 1
        torch.save({"format": "gmt-labeled-decision-v1", "scene": scene,
                    "partition": split_by_scene[scene], "records": labeled}, out_root / f"{scene}.pt")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    collect_ap = sub.add_parser("collect")
    collect_ap.add_argument("--config", type=Path, default=Path("configs/VISION_test.yaml"))
    collect_ap.add_argument("--weight", type=Path, required=True)
    collect_ap.add_argument("--output", type=Path, required=True)
    collect_ap.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "5"))
    collect_ap.add_argument("--scene-limit", type=int, default=None)
    collect_ap.add_argument("--scenes", nargs="+", default=None,
                            help="optional train scene names for a parallel worker")
    collect_ap.set_defaults(func=collect)
    label_ap = sub.add_parser("label")
    label_ap.add_argument("--input", type=Path, required=True)
    label_ap.add_argument("--output", type=Path, required=True)
    label_ap.add_argument("--split", type=Path, required=True)
    label_ap.add_argument("--annotation", type=Path, required=True)
    label_ap.set_defaults(func=label)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
