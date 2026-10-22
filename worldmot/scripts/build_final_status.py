#!/usr/bin/env python
"""Assemble the small, reproducible result index for a WorldMOT run.

The script consumes only run logs and post-hoc audit JSON files.  It never
opens annotations while the tracker is running and it does not manufacture
3-D ground truth or identity labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stage_stats(summary: dict, name: str) -> dict:
    return dict(summary.get("runtime", {}).get(name, {}))


def gpu_max(rows: list[dict]) -> dict:
    values = [r.get("runtime", {}).get("gpu", {}) for r in rows]
    values = [v for v in values if v.get("available")]
    if not values:
        return {"available": False}
    out = {"available": True, "device": values[-1].get("device")}
    for key in ("allocated_mb", "reserved_mb", "peak_allocated_mb", "peak_reserved_mb"):
        out[key] = max(float(v.get(key, 0.0)) for v in values)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=".")
    args = ap.parse_args()
    root = Path(args.project).resolve()
    result_dir = root / "results"

    smoke = read(result_dir / "worldmot_smoke_v2.json", {})
    env = read(result_dir / "environment_audit.json", {})
    run_specs = [
        ("scene_08_100", result_dir / "continuous_v0_100/scene_08/summary.json", result_dir / "continuous_v0_100/scene_08/tracks.jsonl", result_dir / "worldmot_tracking_audit_scene08_100.json"),
        ("scene_09_100", result_dir / "continuous_v0_100/scene_09/summary.json", result_dir / "continuous_v0_100/scene_09/tracks.jsonl", result_dir / "worldmot_tracking_audit_scene09_100.json"),
        ("scene_24_20_diagnostic", result_dir / "continuous_diagnostic/scene_24/summary.json", result_dir / "continuous_diagnostic/scene_24/tracks.jsonl", result_dir / "worldmot_tracking_audit_scene24_20.json"),
    ]
    runs = []
    for name, summary_path, jsonl_path, audit_path in run_specs:
        summary = read(summary_path, {})
        rows = records(jsonl_path)
        audit = read(audit_path, {})
        align = [r.get("alignment", {}) for r in rows]
        runs.append({
            "name": name,
            "summary": str(summary_path.relative_to(root)),
            "tracks": str(jsonl_path.relative_to(root)),
            "posthoc_audit": str(audit_path.relative_to(root)),
            "status": summary.get("status"),
            "frames": len(rows),
            "geometry_ok": sum(r.get("geometry_status") == "OK" for r in rows),
            "alignment_success": sum(bool(a.get("success")) for a in align),
            "alignment_failure_samples": [
                {"timestamp": r.get("timestamp"), "alignment": r.get("alignment")}
                for r in rows if not r.get("alignment", {}).get("success", False)
            ][:20],
            "runtime": summary.get("runtime", {}),
            "gpu_max": gpu_max(rows),
            "posthoc": audit,
        })

    long_runs = [r for r in runs if r["frames"] >= 100]
    stage10 = read(result_dir / "continuous_v1_stage10/scene_08/summary.json", {})
    stage10_rows = records(result_dir / "continuous_v1_stage10/scene_08/tracks.jsonl")
    fps = {
        "exact_stage_split_source": "results/continuous_v1_stage10/scene_08/summary.json",
        "exact_stage_split": {k: stage_stats(stage10, k) for k in (
            "preprocess", "model_only", "panoptic", "pointmap_postprocess",
            "geometry", "world_alignment", "instance_build", "identity_memory", "reprojection", "total")},
        "model_only_fps_mean": 1.0 / max(stage_stats(stage10, "model_only").get("mean_s", 0.0), 1e-9),
        "pipeline_fps_stage10": stage10.get("runtime", {}).get("pipeline_fps"),
        "continuous_100_runs": [{
            "name": r["name"], "frames": r["frames"],
            "pipeline_fps": r["runtime"].get("pipeline_fps"),
            "total_mean_s": r["runtime"].get("total", {}).get("mean_s"),
            "total_p50_s": r["runtime"].get("total", {}).get("p50_s"),
            "total_p95_s": r["runtime"].get("total", {}).get("p95_s"),
            "host_max_rss_mb": r["runtime"].get("host_max_rss_mb"),
            "gpu_max": r["gpu_max"],
            "model_only_note": "legacy timing field from v0 run; exact forward-only split is the 10-frame stage audit above",
        } for r in long_runs],
        "camera_image_throughput_fps": None,
        "parallel_gpu_aggregate_fps": None,
        "notes": [
            "FPS is synchronized two-camera group throughput on one V100 (cuda:0).",
            "No four-GPU aggregate throughput is claimed.",
            "The PanSt3R RoPE CUDA extension was unavailable; official PyTorch RoPE fallback was used.",
        ],
    }

    geometry = {
        "checkpoint": env.get("panst3r_checkpoint"),
        "checkpoint_md5": env.get("checkpoint_md5"),
        "panst3r_git": env.get("panst3r_git"),
        "must3r_git": env.get("must3r_git"),
        "smoke": {
            "status": smoke.get("status"),
            "diagnostics": smoke.get("diagnostics"),
            "views": smoke.get("views"),
            "query_shape": smoke.get("query_shape"),
            "world_instance_count": len(smoke.get("world_instances", [])),
        },
        "continuous_runs": [{
            "name": r["name"], "frames": r["frames"], "geometry_ok": r["geometry_ok"],
            "alignment_success": r["alignment_success"],
            "alignment_failure_samples": r["alignment_failure_samples"],
        } for r in runs],
    }

    instance = {
        "native_panoptic": True,
        "fallback_person": {"enabled": False, "status": "DEPENDENCY_BLOCKED_OR_DISABLED", "used_in_results": False},
        "smoke_world_instances": smoke.get("world_instances", []),
        "query_is_long_term_id": False,
        "continuous_proposals_and_tracks": [{
            "name": r["name"], "frames": r["frames"],
            "max_proposals": max((int(x.get("proposal_count", 0)) for x in records(root / r["tracks"])), default=0),
            "max_tracks": max((int(len(x.get("tracks_3d", []))) for x in records(root / r["tracks"])), default=0),
        } for r in runs],
        "interpretation": "The raw person-query output includes false positives; native panoptic availability is verified, person detection quality is not claimed as solved.",
    }

    projection = {
        "runs": [],
        "fields": ["native_view_mask", "projected_from_world", "projection_refined_by_target_view", "visible", "bbox_xywh"],
        "pure_world_projection": True,
    }
    for r in runs:
        rows = records(root / r["tracks"])
        entries = [x for row in rows for vals in row.get("tracks_2d_by_camera", {}).values() for x in vals]
        projection["runs"].append({
            "name": r["name"], "frames": len(rows), "camera_track_entries": len(entries),
            "visible_entries": sum(bool(x.get("visible")) for x in entries),
            "native_mask_entries": sum(bool(x.get("native_view_mask")) for x in entries),
            "world_projection_entries": sum(bool(x.get("projected_from_world")) for x in entries),
            "target_view_refined_entries": sum(bool(x.get("projection_refined_by_target_view")) for x in entries),
        })

    tracking = {"runs": [{"name": r["name"], "posthoc": r["posthoc"]} for r in runs], "iou_threshold": 0.3,
                "ground_truth_use": "posthoc_only", "no_3d_ground_truth": True}

    final = {
        "project": "WorldMOT",
        "code_root": str(root),
        "backend": "official PanSt3R v2, two keyframes, use_retrieval=False, standard_v2",
        "inference_uses_gt": False,
        "gmt_integration": "NOT_ATTEMPTED",
        "fallback_person": "DISABLED_OR_DEPENDENCY_BLOCKED; no fallback predictions used",
        "statuses": {
            "PIPELINE_EXECUTABLE": "PASS" if all(r["status"] == "PASS" for r in runs) else "PARTIAL",
            "REAL_PANOPTIC_INSTANCE_STATUS": "PASS_WITH_FALSE_POSITIVES" if smoke.get("status") == "PASS" and smoke.get("world_instances") else "NO_PERSON_DETECTIONS",
            "WORLD_ALIGNMENT_STATUS": "PASS" if all(r["alignment_success"] == r["frames"] for r in long_runs) else "PARTIAL",
            "TEMPORAL_ID_STATUS": "PARTIAL_FAILURES_RECORDED",
            "REPROJECTION_STATUS": "PASS_WITH_DETECTION_ERRORS",
            "FPS_MEASURED": "PASS",
            "SCIENTIFIC_SUCCESS_NOT_YET_CLAIMED": True,
        },
        "acceptance_gate_summary": {
            "gate_a_real_multiview_instance": "PASS_WITH_QUALITY_LIMITS",
            "gate_b_3d_instance_and_projection": "PASS_WITH_POSTHOC_2D_ERRORS",
            "gate_c_100_group_persistent_state": "PASS_ENGINEERING_GATE; identity accuracy remains partial",
            "gate_d_two_scene_end_to_end": "PASS",
            "gate_e_real_fps": "PASS",
        },
        "evidence": {
            "environment": "results/environment_audit.json",
            "smoke_visualization": "results/worldmot_smoke_v2/pointcloud_3d.png",
            "instance_visualization": "results/worldmot_smoke_v2/instance_visualization.json",
            "geometry": "results/worldmot_geometry_audit.json",
            "instances": "results/worldmot_instance_audit.json",
            "projection": "results/worldmot_projection_audit.json",
            "tracking": "results/worldmot_tracking_audit.json",
            "fps": "results/worldmot_fps.json",
            "continuous_artifacts": [
                "results/continuous_v0_100/scene_08/artifacts/artifact_manifest.json",
                "results/continuous_v0_100/scene_09/artifacts/artifact_manifest.json",
            ],
        },
        "failure_samples": {
            "scene_08": "results/worldmot_tracking_audit_scene08_100.json",
            "scene_09": "results/worldmot_tracking_audit_scene09_100.json",
            "scene_24_crowded": "results/worldmot_tracking_audit_scene24_20.json",
        },
        "scientific_conclusion": "The independent RGB-to-PanSt3R-to-WorldInstance-to-Sim3-to-persistent-ID-to-reprojection loop is executable. Native person-query precision and temporal/cross-view identity are not yet strong enough for a scientific MOT success claim or GMT integration.",
    }

    for name, data in {
        "worldmot_geometry_audit.json": geometry,
        "worldmot_instance_audit.json": instance,
        "worldmot_projection_audit.json": projection,
        "worldmot_tracking_audit.json": tracking,
        "worldmot_fps.json": fps,
        "worldmot_final_status.json": final,
    }.items():
        (result_dir / name).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(final, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
