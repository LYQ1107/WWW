#!/usr/bin/env python
"""Build the registered WorldMOT detection-order ablation artifacts.

The script consumes only saved causal JSONL and the original VisionTrack
annotations.  It does not alter predictions or manufacture GT.  All rows are
tagged with the source system so the unavailable Omega 3-D person head cannot
silently enter the numerical comparison.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worldmot.data.visiontrack import VisionTrack
from worldmot.models.omega_3d_query_detector import (
    PERSON_3D_HEAD_REQUIRES_TRAINING,
    VGGT_DET_COMMIT,
    VGGT_DET_CONFIG,
    VGGT_DET_HEAD,
    VGGT_DET_REFERENCE,
)


ROOT = Path(__file__).resolve().parents[1]
SCENES = (8, 9, 24)


def _q(values: list[float], p: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=np.float64), p, method="nearest"))


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return float(inter / union) if union > 0 else 0.0


def _greedy(preds: list[dict], anns: list[dict], threshold: float = 0.3):
    pairs = []
    for pi, p in enumerate(preds):
        box = p.get("bbox_xywh") or p.get("bbox")
        if box is None:
            continue
        for ai, a in enumerate(anns):
            score = _iou(box, a["bbox"])
            if score >= threshold:
                pairs.append((score, pi, ai))
    out, used_p, used_a = [], set(), set()
    for score, pi, ai in sorted(pairs, reverse=True):
        if pi in used_p or ai in used_a:
            continue
        used_p.add(pi); used_a.add(ai); out.append((pi, ai, score))
    return out


def _rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _raw_detection_metrics(dataset: VisionTrack, rows: list[dict], threshold: float = 0.3) -> dict:
    gt_total = pred_total = matched = 0
    ious = []
    for row in rows:
        for view, image_id in row.get("image_ids", {}).items():
            anns = dataset.ground_truth(int(image_id))
            preds = row.get("detections_by_camera", {}).get(str(view), [])
            gt_total += len(anns); pred_total += len(preds)
            pairs = _greedy(preds, anns, threshold)
            matched += len(pairs); ious.extend(x[2] for x in pairs)
    return {
        "gt_boxes": gt_total, "predictions": pred_total, "matched": matched,
        "false_positives": max(0, pred_total - matched),
        "false_negatives": max(0, gt_total - matched),
        "precision": float(matched / max(pred_total, 1)),
        "recall": float(matched / max(gt_total, 1)),
        "mean_iou": float(np.mean(ious)) if ious else 0.0,
        "iou_threshold": threshold,
        "AP": None,
        "AP_reason": "single saved operating point; detector-native COCO AP requires a score sweep",
    }


def _fusion_stats(rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    candidate = accepted = rejected = abstained = 0
    reliable = total_obs = 0
    for row in rows:
        for obs in row.get("world_observations", []):
            status = str(obs.get("fusion_status", "UNKNOWN"))
            counts[status] = counts.get(status, 0) + 1
            total_obs += 1
            reliable += int(bool(obs.get("geometry_reliable", False)))
        f = row.get("fusion", {}) or {}
        candidate += int(f.get("candidate_pairs", 0) or 0)
        accepted += int(f.get("accepted_pairs", 0) or 0)
        rejected += int(f.get("rejected_pairs", 0) or 0)
        abstained += int(f.get("abstained", 0) or 0)
    return {
        "observation_status_counts": counts,
        "fused_two_view": counts.get("FUSED_TWO_VIEW", 0),
        "unpaired_single_view": counts.get("UNPAIRED_SINGLE_VIEW", 0),
        "rejected_geometry": counts.get("REJECTED", 0),
        "geometry_reliable_ratio": float(reliable / max(total_obs, 1)),
        "candidate_pairs": candidate, "accepted_pairs": accepted,
        "rejected_pairs": rejected, "abstained": abstained,
        "fusion_acceptance": float(accepted / max(candidate, 1)),
        "cannot_determine_ratio": float(abstained / max(candidate + abstained, 1)),
    }


def _frame_timings(rows: list[dict], system: str) -> dict[str, list[float]]:
    """Return per-synchronized-time stage samples, with legacy fallback."""
    out: dict[str, list[float]] = {}
    previous_totals: dict[str, float] = {}
    for row in rows:
        runtime = row.get("runtime", {}) or {}
        frame = runtime.get("frame_stages") or {}
        if frame:
            values = {str(k): float(v) for k, v in frame.items() if isinstance(v, (int, float))}
        else:
            values = {}
            cumulative = runtime.get("stages", {}) or {}
            for name, rec in cumulative.items():
                if not isinstance(rec, dict) or "total_s" not in rec:
                    continue
                total = float(rec["total_s"])
                values[name] = max(0.0, total - previous_totals.get(name, 0.0))
                previous_totals[name] = total
        if "rgb_read_s" in row:
            values["rgb_read"] = float(row["rgb_read_s"])
        if "end_to_end_s" in row:
            values["end_to_end"] = float(row["end_to_end_s"])
        elif "total" in values:
            values["end_to_end"] = values["total"] + values.get("rgb_read", 0.0)
        for name, value in values.items():
            out.setdefault(name, []).append(float(value))
    return out


def _fps_rows(rows: list[dict], system: str, scene: int) -> list[dict]:
    samples = _frame_timings(rows, system)
    out = []
    for stage, values in sorted(samples.items()):
        mean = _mean(values); p50 = _q(values, 0.50); p95 = _q(values, 0.95)
        out.append({
            "system": system, "scene": scene, "stage": stage,
            "samples": len(values), "mean_s": mean, "p50_s": p50, "p95_s": p95,
            "pair_fps_from_mean": (1.0 / mean if mean and mean > 0 else None),
            "pair_fps_from_p50": (1.0 / p50 if p50 and p50 > 0 else None),
            "timing_unit": "one synchronized two-camera timestamp",
        })
    return out


def _metric_row(dataset: VisionTrack, rows: list[dict], metrics: dict, system: str, scene: int) -> dict:
    fusion = _fusion_stats(rows)
    raw = _raw_detection_metrics(dataset, rows) if system == "A" else {
        "gt_boxes": None, "predictions": None, "matched": None,
        "false_positives": None, "false_negatives": None,
        "precision": None, "recall": None, "mean_iou": None,
        "AP": None, "AP_reason": "B0 has no independent 2-D person detector",
    }
    geometry_ok = sum(str(r.get("geometry_status")) == "OK" for r in rows)
    return {
        "system": system, "scene": scene, "frames": metrics.get("frames", len(rows)),
        "gt_boxes": metrics.get("gt_boxes"), "visible_predictions": metrics.get("visible_predictions"),
        "matched_boxes": metrics.get("matched_boxes"), "false_positives": metrics.get("false_positives"),
        "false_negatives": metrics.get("false_negatives"), "precision": metrics.get("box_precision"),
        "recall": metrics.get("box_recall"), "AP": metrics.get("AP"),
        "visibility_accuracy": metrics.get("visibility_accuracy"),
        "mean_projection_iou": metrics.get("matched_iou_mean"),
        "IDSW": metrics.get("id_switches"), "fragments": metrics.get("fragments"),
        "HOTA": metrics.get("HOTA"), "AssA": metrics.get("AssA"),
        "IDF1": metrics.get("IDF1"), "CVIDF1": metrics.get("CVIDF1"),
        "cross_view_pairs": metrics.get("cross_view_pairs"),
        "cross_view_same_id": metrics.get("cross_view_same_global_id"),
        "same_person_fusion_success": metrics.get("cross_view_accuracy"),
        "different_person_merge_errors": len(metrics.get("cross_view_conflicts_sample", [])),
        "raw_detector_precision": raw.get("precision"), "raw_detector_recall": raw.get("recall"),
        "raw_detector_mean_iou": raw.get("mean_iou"), "raw_detector_fp": raw.get("false_positives"),
        "raw_detector_fn": raw.get("false_negatives"),
        "fused_two_view": fusion["fused_two_view"], "unpaired_single_view": fusion["unpaired_single_view"],
        "fusion_candidate_pairs": fusion["candidate_pairs"], "fusion_accepted_pairs": fusion["accepted_pairs"],
        "fusion_abstained": fusion["abstained"], "cannot_determine_ratio": fusion["cannot_determine_ratio"],
        "geometry_reliable_ratio": fusion["geometry_reliable_ratio"] if system == "A" else float(geometry_ok / max(len(rows), 1)),
        "geometry_ok_frames": geometry_ok,
        "protocol": metrics.get("protocol", "original VisionTrack GT; fixed IoU; complete supplied sequence"),
    }


def _failure_cases(dataset: VisionTrack, rows: list[dict], system: str, scene: int) -> list[dict]:
    failures = []
    for row in rows:
        frame = int(row["timestamp"])
        if row.get("geometry_status") not in (None, "OK"):
            failures.append({"system": system, "scene": scene, "frame": frame,
                             "stage": "geometry_reconstruction", "reason": row.get("geometry_status")})
        if not (row.get("alignment") or {}).get("success", True):
            failures.append({"system": system, "scene": scene, "frame": frame,
                             "stage": "world_alignment", "reason": (row.get("alignment") or {}).get("reason")})
        fusion = row.get("fusion", {}) or {}
        if int(fusion.get("abstained", 0) or 0) > 0:
            failures.append({"system": system, "scene": scene, "frame": frame,
                             "stage": "cross_view_fusion", "reason": "abstained_unpaired_observations",
                             "abstained": int(fusion.get("abstained", 0))})
        if system == "A":
            for view, image_id in row.get("image_ids", {}).items():
                anns = dataset.ground_truth(int(image_id))
                preds = row.get("detections_by_camera", {}).get(str(view), [])
                pairs = _greedy(preds, anns)
                if len(pairs) < len(anns):
                    failures.append({"system": system, "scene": scene, "frame": frame,
                                     "view": int(view), "stage": "person_detection",
                                     "reason": "raw_detector_missed_ground_truth",
                                     "missed": len(anns) - len(pairs)})
                if len(preds) > len(pairs):
                    failures.append({"system": system, "scene": scene, "frame": frame,
                                     "view": int(view), "stage": "person_detection",
                                     "reason": "raw_detector_false_positive",
                                     "false_positives": len(preds) - len(pairs)})
        # A current-target output with no matching GT is a projection/identity
        # diagnostic; it is kept separately from raw detector failures.
        for view, image_id in row.get("image_ids", {}).items():
            anns = dataset.ground_truth(int(image_id))
            preds = row.get("tracks_2d_by_camera", {}).get(str(view), [])
            eligible = [p for p in preds if p.get("evaluation_eligible", p.get("visible", False))]
            if eligible and not _greedy(eligible, anns):
                failures.append({"system": system, "scene": scene, "frame": frame,
                                 "view": int(view), "stage": "projection_or_identity",
                                 "reason": "no_iou_match_for_current_observation",
                                 "predictions": len(eligible), "gt": len(anns)})
    return failures[:500]


def main() -> int:
    dataset = VisionTrack("/data/DATASETS/TRACKING/JDE/VisionTrack")
    metrics_rows: list[dict] = []
    fps_rows: list[dict] = []
    failures: list[dict] = []
    scene_records = {}
    for system in ("A", "B0"):
        for scene in SCENES:
            jsonl = ROOT / "results" / "detection_order" / system / f"scene_{scene:02d}" / "tracks.jsonl"
            metric_path = jsonl.with_name("metrics.json")
            if not jsonl.is_file():
                continue
            rows = _rows(jsonl)
            metrics = json.loads(metric_path.read_text(encoding="utf-8")) if metric_path.is_file() else {}
            metrics_rows.append(_metric_row(dataset, rows, metrics, system, scene))
            fps_rows.extend(_fps_rows(rows, system, scene))
            case = _failure_cases(dataset, rows, system, scene)
            failures.extend(case)
            scene_records[f"{system}:scene_{scene:02d}"] = {
                "jsonl": str(jsonl.relative_to(ROOT)), "frames": len(rows),
                "jsonl_gz": (str(jsonl.with_suffix(jsonl.suffix + ".gz").relative_to(ROOT))
                             if jsonl.with_suffix(jsonl.suffix + ".gz").is_file() else None),
                "metrics": metrics, "fusion": _fusion_stats(rows),
            }
    # The public VGGT-Det reference checkpoint is ScanNet-only and has no
    # person class.  It is registered in the artifact, but excluded from
    # numerical ranking until a legal person-supervised checkpoint exists.
    b1 = {
        "system": "B1", "state": PERSON_3D_HEAD_REQUIRES_TRAINING,
        "reference_repository": VGGT_DET_REFERENCE, "reference_commit": VGGT_DET_COMMIT,
        "reference_config": VGGT_DET_CONFIG, "reference_head": VGGT_DET_HEAD,
        "reason": "public ScanNet 18-class head contains no person category; no VisionTrack-person checkpoint was found",
        "formal_metrics_included": False,
    }
    control_path = ROOT / "results" / "detection_order" / "schedule_control.json"
    schedule_control = {
        "state": "REGISTERED_CONTROL_PENDING",
        "description": "same complete RGB pair, detector and Omega execution order swapped; no model/input change",
        "formal_metrics_included": False,
        "reason": "a finite ordering smoke has not completed",
    }
    if control_path.is_file():
        control = json.loads(control_path.read_text(encoding="utf-8"))
        schedule_control.update({"state": control.get("state", "UNKNOWN"),
                                 "records": control.get("records", []),
                                 "output": str(control_path),
                                 "reason": "same outputs within the registered finite control; excluded from A/B0 ranking"})
    cadence = {
        "full_frame_baseline": "MEASURED",
        "geometry_update_every_5": "NOT_ADMITTED_NO_CURRENT_3D_INFERENCE",
        "geometry_update_every_10": "NOT_ADMITTED_NO_CURRENT_3D_INFERENCE",
        "reason": "intermediate frames cannot reuse stale person depth; no legal current-frame pose/3-D inference was counted",
    }
    ablation = {
        "schema_version": 1,
        "independent_of_gmt": True,
        "baseline_commit": "846a73004a0c1f1514324e8596234614d4644fab",
        "branch": "research/worldmot-detection-reconstruction-ablation-20261022",
        "systems": {
            "A": {"name": "RTMDet-Ins-tiny + VGGT-Omega-512 + 3-D fusion + WorldIdentityMemory",
                  "input": "two complete synchronized RGB images",
                  "detector_checkpoint_sha256": "ec670f7ee9e20bd7931e15f15b7016f7fe531baaab81f2e6153382d046111885",
                  "omega_checkpoint_sha256": "c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934",
                  "formal_comparison": True},
            "B0": {"name": "PanSt3R native joint instances + WorldIdentityMemory",
                    "formal_comparison": True},
            "B1": b1,
        },
        "scenes": scene_records,
        "metrics": metrics_rows,
        "fps": fps_rows,
        "schedule_only_control": schedule_control,
        "cadence_speed_experiments": cadence,
        "limitations": [
            "AP is null because the saved run is one operating point; raw detector precision/recall are reported.",
            "Scene 24 is a 20-pair diagnostic sequence, not a complete-video scientific claim.",
            "B1 is excluded until a person-supervised 3-D query checkpoint is available.",
        ],
    }
    out = ROOT / "results"
    (out / "worldmot_detection_order_ablation.json").write_text(json.dumps(ablation, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = sorted({k for row in metrics_rows for k in row})
    with (out / "worldmot_detection_order_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(metrics_rows)
    fields = sorted({k for row in fps_rows for k in row})
    with (out / "worldmot_detection_order_fps.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(fps_rows)
    (out / "worldmot_detection_order_failure_cases.json").write_text(
        json.dumps({"schema_version": 1, "cases": failures, "case_count": len(failures)}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(json.dumps({"metrics_rows": len(metrics_rows), "fps_rows": len(fps_rows), "failure_cases": len(failures),
                      "b1": b1["state"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
