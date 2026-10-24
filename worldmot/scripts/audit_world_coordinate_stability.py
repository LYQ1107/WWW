#!/usr/bin/env python3
"""Audit G0 alignment stability from the immutable sparse support cache."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from worldmot.data.worldobject_cache import iter_complete_scenes


def val(x):
    return float(x) if x is not None and np.isfinite(float(x)) else np.nan


def main():
    p = argparse.ArgumentParser(); p.add_argument("--cache", required=True); p.add_argument("--output-dir", required=True); p.add_argument("--split", choices=["train", "test"], default="train")
    a = p.parse_args(); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []; summary = {}
    for cache in iter_complete_scenes(a.cache):
        scene_rows = []
        for frame in cache.frames():
            rec = frame["record"]; dual = rec.get("alignment", {}); original = rec.get("original_view_alignment", {})
            per = dual.get("per_camera", {})
            row = {"scene": cache.scene, "timestamp": frame["timestamp"],
                   "g0a_success": bool(original.get("success", False)), "g0a_reason": original.get("reason", "NOT_RECORDED"),
                   "g0a_correspondences": original.get("correspondences"), "g0a_inliers": original.get("inliers"),
                   "g0a_inlier_ratio": val(original.get("inlier_ratio")), "g0a_residual_p95": val(original.get("residual_p95")),
                   "g0a_scale": val(original.get("scale")), "g0b_success": bool(dual.get("success", False)),
                   "g0b_reason": dual.get("reason", ""), "g0b_correspondences": dual.get("correspondences"),
                   "g0b_inliers": dual.get("inliers"), "g0b_inlier_ratio": val(dual.get("inlier_ratio")),
                   "g0b_residual_p95": val(dual.get("residual_p95")), "g0b_scale": val(dual.get("scale")),
                   "g0b_roundtrip_error": val(dual.get("roundtrip_error")),
                   "camera1_success": bool(per.get("1", {}).get("success", False)),
                   "camera2_success": bool(per.get("2", {}).get("success", False)),
                   "camera1_inliers": per.get("1", {}).get("inliers"), "camera2_inliers": per.get("2", {}).get("inliers"),
                   "camera1_residual_p95": val(per.get("1", {}).get("residual_p95")),
                   "camera2_residual_p95": val(per.get("2", {}).get("residual_p95")),
                   "used_cameras": ";".join(map(str, dual.get("used_cameras", []))),
                   "object_build_blocked": not bool(dual.get("success", False))}
            rows.append(row); scene_rows.append(row)
        def median(key):
            x = [r[key] for r in scene_rows if np.isfinite(r[key])] if scene_rows else []
            return float(np.median(x)) if x else None
        summary[str(cache.scene)] = {"timestamps": len(scene_rows),
            "g0a_success_rate": float(np.mean([r["g0a_success"] for r in scene_rows])) if scene_rows else 0.0,
            "g0b_success_rate": float(np.mean([r["g0b_success"] for r in scene_rows])) if scene_rows else 0.0,
            "g0a_residual_p95_median": median("g0a_residual_p95"), "g0b_residual_p95_median": median("g0b_residual_p95"),
            "g0b_roundtrip_error_median": median("g0b_roundtrip_error"),
            "g0b_scale_median": median("g0b_scale"),
            "g0b_object_build_blocked_rate": float(np.mean([r["object_build_blocked"] for r in scene_rows])) if scene_rows else 0.0,
            "g0b_failure_reasons": dict(Counter(r["g0b_reason"] for r in scene_rows if not r["g0b_success"]))}
    csv_path = out / f"G0_world_alignment_all{len(summary)}.csv"
    fields = list(rows[0]) if rows else ["scene", "timestamp"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary_path = out / "G0_world_alignment_per_scene.json"
    summary_path.write_text(json.dumps({"schema": "worldobject_g0_alignment_v1", "cache": str(Path(a.cache).resolve()),
                                        "split": a.split, "scenes": summary, "timestamps": len(rows),
                                        "note": "G0-A is the original first-view alignment recorded alongside G0-B; errors are internal consistency, not absolute metric accuracy."}, indent=2, allow_nan=False))
    print(json.dumps({"timestamps": len(rows), "scenes": len(summary), "csv": str(csv_path), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__": main()
