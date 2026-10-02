#!/usr/bin/env python3
"""Materialize a fixed-K decision dataset from labeled online records."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import torch


CANDIDATE_FEATURE_NAMES = (
    "traj_score", "support", "normalized_score", "max_asso", "mean_asso",
    "std_asso", "observation_count", "distinct_view_count",
    "time_since_last_seen", "same_view_time_since_last_seen", "age",
    "current_to_all_history_cosine", "current_to_recent_history_cosine",
    "history_feature_variance", "detector_score_mean", "detector_score_std",
)
QUERY_EXTRA_NAMES = (
    "current_score", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "candidate_count", "top1_normalized_score", "top2_normalized_score",
    "top1_top2_margin", "candidate_entropy", "view_index_normalized",
)


def candidate_vector(candidate: dict, *, is_new: bool = False) -> list[float]:
    return [float(candidate.get(name, 0.0)) for name in CANDIDATE_FEATURE_NAMES] + [
        1.0 if is_new else 0.0
    ]


def query_vector(row: dict) -> list[float]:
    reid = row["current_fused_reid"].float().reshape(-1).tolist()
    bbox = [float(x) for x in row["bbox_xywh_normalized"]]
    extra = [
        float(row.get(name, 0.0)) for name in QUERY_EXTRA_NAMES[:1]
    ] + bbox + [
        float(row.get(name, 0.0)) for name in QUERY_EXTRA_NAMES[5:-1]
    ] + [float(row["view_index"]) / 8.0]
    return reid + extra


def ranked_candidates(row: dict) -> list[dict]:
    return sorted(
        row["candidates"],
        key=lambda item: (-float(item["traj_score"]), int(item["global_id"])),
    )


def load_rows(records_root: Path, split: dict) -> dict[str, list[dict]]:
    split_by_scene = {
        scene: part for part, scenes in split["scenes_by_split"].items() for scene in scenes
    }
    rows_by_partition: dict[str, list[dict]] = collections.defaultdict(list)
    for path in sorted((records_root / "labeled_records").glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        scene = str(payload["scene"])
        if scene not in split_by_scene:
            raise ValueError(f"scene not in frozen split: {scene}")
        rows_by_partition[split_by_scene[scene]].extend(payload["records"])
    return rows_by_partition


def materialize(rows: list[dict], k: int, template: dict | None = None) -> dict:
    queries, options, masks = [], [], []
    targets, baselines, supervised, hard = [], [], [], []
    extras, extra_masks, extra_ids = [], [], []
    metadata = []
    max_extra = 4
    for row in sorted(rows, key=lambda x: (x["scene"], x["frame_index"], x["view_index"], x["detection_index"])):
        candidates = ranked_candidates(row)
        top = candidates[:k]
        tail = candidates[k:k + max_extra]
        q = query_vector(row)
        opt = [candidate_vector(candidate) for candidate in top]
        mask = [True] * len(opt)
        while len(opt) < k:
            opt.append([0.0] * (len(CANDIDATE_FEATURE_NAMES) + 1))
            mask.append(False)
        opt.append(candidate_vector({}, is_new=True))
        mask.append(True)
        target = -1
        target_type = str(row.get("target_type", "UNKNOWN"))
        target_id = row.get("target_global_id")
        top_ids = [int(candidate["global_id"]) for candidate in top]
        if row.get("gt_id_for_supervision") is not None:
            if target_type == "NEW":
                target = k
            elif target_id is not None and int(target_id) in top_ids:
                target = top_ids.index(int(target_id))
        baseline_id = int(row["gmt_choice_id"])
        baseline = top_ids.index(baseline_id) if baseline_id in top_ids else k
        extra = [candidate_vector(candidate) for candidate in tail]
        extra_id = [int(candidate["global_id"]) for candidate in tail]
        while len(extra) < max_extra:
            extra.append([0.0] * (len(CANDIDATE_FEATURE_NAMES) + 1))
            extra_id.append(-1)
        extras.append(extra)
        extra_masks.append([index < len(tail) for index in range(max_extra)])
        extra_ids.append(extra_id)
        queries.append(q)
        options.append(opt)
        masks.append(mask)
        targets.append(target)
        baselines.append(baseline)
        is_supervised = target >= 0
        supervised.append(is_supervised)
        hard.append(bool(is_supervised and baseline != target and target_type == "EXISTING"))
        metadata.append({
            "scene": str(row["scene"]), "frame_index": int(row["frame_index"]),
            "view_index": int(row["view_index"]), "detection_index": int(row["detection_index"]),
            "image_id": int(row["image_id"]), "target_type": target_type,
            "target_global_id": (int(target_id) if target_id is not None else None),
            "gt_id_for_supervision": (
                int(row["gt_id_for_supervision"]) if row.get("gt_id_for_supervision") is not None else None
            ),
            "candidate_miss": bool(target_type == "EXISTING" and target < 0 and row.get("gt_id_for_supervision") is not None),
            "baseline_choice_correct": bool(row.get("baseline_choice_correct", False)),
            "baseline_is_new": bool(row.get("baseline_is_new", False)),
            "top_candidate_ids": top_ids,
            "all_candidate_count": len(candidates),
        })
    if not queries:
        if template is None:
            raise RuntimeError("no records to cache and no template is available")
        query_dim = len(query_vector(template))
        option_dim = len(CANDIDATE_FEATURE_NAMES) + 1
        return {
            "format": "gmt-decision-cache-v1", "K": int(k),
            "query": torch.empty((0, query_dim), dtype=torch.float32),
            "option": torch.empty((0, k + 1, option_dim), dtype=torch.float32),
            "option_mask": torch.empty((0, k + 1), dtype=torch.bool),
            "target": torch.empty((0,), dtype=torch.long), "baseline": torch.empty((0,), dtype=torch.long),
            "supervised": torch.empty((0,), dtype=torch.bool), "hard_recoverable": torch.empty((0,), dtype=torch.bool),
            "extra_option": torch.empty((0, 4, option_dim), dtype=torch.float32),
            "extra_mask": torch.empty((0, 4), dtype=torch.bool), "extra_ids": torch.empty((0, 4), dtype=torch.long),
            "metadata": [], "feature_names": {"query_extra": list(QUERY_EXTRA_NAMES), "candidate": list(CANDIDATE_FEATURE_NAMES) + ["is_new"]},
        }
    return {
        "format": "gmt-decision-cache-v1", "K": int(k),
        "query": torch.tensor(queries, dtype=torch.float32),
        "option": torch.tensor(options, dtype=torch.float32),
        "option_mask": torch.tensor(masks, dtype=torch.bool),
        "target": torch.tensor(targets, dtype=torch.long),
        "baseline": torch.tensor(baselines, dtype=torch.long),
        "supervised": torch.tensor(supervised, dtype=torch.bool),
        "hard_recoverable": torch.tensor(hard, dtype=torch.bool),
        "extra_option": torch.tensor(extras, dtype=torch.float32),
        "extra_mask": torch.tensor(extra_masks, dtype=torch.bool),
        "extra_ids": torch.tensor(extra_ids, dtype=torch.long),
        "metadata": metadata,
        "feature_names": {
            "query_extra": list(QUERY_EXTRA_NAMES),
            "candidate": list(CANDIDATE_FEATURE_NAMES) + ["is_new"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    args = parser.parse_args()
    split = json.loads(args.split.read_text())
    rows = load_rows(args.records, split)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"K": args.k, "partitions": {}}
    template = next((row for partition_rows in rows.values() for row in partition_rows), None)
    for partition in ("JEV-train", "JEV-calibration", "JEV-dev"):
        payload = materialize(rows.get(partition, []), args.k, template=template)
        path = args.output / {
            "JEV-train": "train.pt", "JEV-calibration": "calibration.pt", "JEV-dev": "dev.pt"
        }[partition]
        torch.save(payload, path)
        summary["partitions"][partition] = {
            "records": int(len(payload["target"])),
            "supervised": int(payload["supervised"].sum()),
            "hard_recoverable": int(payload["hard_recoverable"].sum()),
            "candidate_miss": int(sum(item["candidate_miss"] for item in payload["metadata"])),
        }
    (args.output / "CACHE_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
