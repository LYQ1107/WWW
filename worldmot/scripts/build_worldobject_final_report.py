#!/usr/bin/env python3
"""Build compact, reproducible World Object TRAIN/TEST result tables.

This formatter never fills missing measurements with estimates.  A route is
reported as ``BLOCKED`` until an official evaluation file exists, which keeps
the final comparison honest while large cache/replay jobs are still running.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


BASELINES = [
    {"route": "GMT_NATIVE_OFFICIAL", "scope": "TEST", "HOTA": 66.06458719840953, "AssA": 69.57874927900562, "DetA": 63.066450212409954, "IDF1": 80.11214466022774, "CVIDF1": 78.06130184629053, "MOTA": 72.9636167241879, "IDSW": 6044, "Frag": 1584, "fps_pairs": 3.3312858917096975, "status": "REFERENCE"},
    {"route": "GMTDET_V2_PRIMARY", "scope": "TEST", "HOTA": 61.017707694844106, "AssA": 60.227445225827594, "DetA": 62.10987666049185, "IDF1": 74.21890907445533, "CVIDF1": 72.69292549942308, "MOTA": 75.45643754775139, "IDSW": 598, "Frag": 1289, "fps_pairs": 0.7480599696996867, "status": "REFERENCE"},
    {"route": "GMTDET_V2_NO_GEOMETRY", "scope": "TEST", "HOTA": 59.20432734237009, "AssA": 59.702570089297005, "DetA": 58.96237911501368, "IDF1": 71.72844368590725, "CVIDF1": 69.95229949942308, "MOTA": 71.482467002134, "IDSW": 699, "fps_pairs": 0.7303701651210337, "status": "REFERENCE"},
    {"route": "GMTDET_V1", "scope": "TEST", "HOTA": 48.483054016911495, "AssA": 41.560070768195324, "DetA": 56.87551509925592, "IDF1": 53.07887398432617, "CVIDF1": 52.521154863214576, "MOTA": 67.83043970809074, "IDSW": 2761, "fps_pairs": 0.6740990575379859, "status": "REFERENCE"},
]


def load(path: Path, default=None):
    if not path or not path.is_file():
        return default
    return json.loads(path.read_text())


def eval_row(route: str, path: Path, fps_by_route: dict[str, dict[str, object]]) -> dict:
    report = load(path)
    row = {"route": route, "scope": "TEST", "status": "BLOCKED" if report is None else report.get("status", "UNKNOWN")}
    for key in ["HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW", "Frag"]:
        row[key] = None
    row["CVIDF1"] = None
    row["CVMA"] = None
    if report:
        te = report.get("TrackEval", {}).get("combined", {})
        row.update({k: te.get(k) for k in ["HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDSW", "Frag"]})
        mat = report.get("VisionTrack_MATLAB", {}).get("combined", {})
        row["CVIDF1"] = mat.get("CVIDF1")
        row["CVMA"] = mat.get("CVMA")
    speed = fps_by_route.get(route, {})
    row["fps_pairs"] = speed.get("fps")
    row["fps_source"] = speed.get("source")
    return row


def fps_map(path: Path) -> dict[str, dict[str, object]]:
    out = {}
    if not path.is_file():
        return out
    for row in csv.DictReader(path.open()):
        # Keep the speed definition explicit.  A cache replay is useful for
        # association profiling but is not an online RGB->ID measurement.
        stage = row.get("stage")
        if stage not in {"live_e2e_s", "cache_replay_s", "total_cache_step_s"}:
            continue
        route = row.get("route", "")
        if stage == "live_e2e_s":
            route = "W3_PERSISTENT_BANK"
            source = "live_rgb_to_id"
        elif stage == "cache_replay_s":
            source = "cache_replay_no_rgb_omega"
            if "W3_PERSISTENT_BANK" in route:
                route = "W3_PERSISTENT_BANK"
            elif "W4_LEARNED_UPDATE" in route:
                route = "W4_LEARNED_UPDATE"
            elif "W5_NO_GEOMETRY_CONTROL" in route:
                route = "W5_NO_GEOMETRY_CONTROL"
        else:
            # The cache-build route is intentionally not attached to a
            # tracker route: it includes Ω and support construction.
            route = "OMEGA_CACHE_BUILD"
            source = "cache_step_rgb_to_support"
        candidate = {"fps": float(row["p50_fps_pairs"]), "source": source}
        # If a live measurement and replay measurement coexist, live wins for
        # the final route table while the raw FPS CSV retains both rows.
        if route not in out or source == "live_rgb_to_id":
            out[route] = candidate
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--evaluation-dir", required=True)
    p.add_argument("--fps", required=True)
    p.add_argument("--train-audit-dir", required=True)
    p.add_argument("--test-cache", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args(); out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    fp = fps_map(Path(a.fps))
    rows = list(BASELINES)
    for route, filename in [("W3_PERSISTENT_BANK", "W3_persistent_bank/metrics.json"), ("W4_LEARNED_UPDATE", "W4_learned/metrics.json"), ("W5_NO_GEOMETRY_CONTROL", "W5_no_geometry/metrics.json")]:
        rows.append(eval_row(route, Path(a.evaluation_dir) / filename, fp))
    fields = ["route", "scope", "status", "HOTA", "AssA", "DetA", "IDF1", "CVIDF1", "CVMA", "MOTA", "IDSW", "Frag", "fps_pairs", "fps_source"]
    with (out / "worldobject_final_comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows({k: r.get(k) for k in fields} for r in rows)

    g0 = load(Path(a.train_audit_dir) / "G0_world_alignment_per_scene.json", {})
    g1 = load(Path(a.train_audit_dir) / "G1_object_support_summary.json", {})
    g2 = load(Path(a.train_audit_dir) / "G2_multiview_object_all24.csv", {})
    g3 = load(Path(a.train_audit_dir) / "G3_temporal_identity_failures.json", {})
    err = {
        "schema": "worldobject_error_attribution_v1",
        "scope": "TRAIN_DIAGNOSTIC",
        "detector_cache": {"source": "immutable GMT candidate cache", "GT_in_cache": False},
        "alignment": {"g0": g0.get("scenes", {}), "failure_reason_counts": Counter(x.get("g0b_reason", "") for x in [])},
        "support": g1.get("summary", {}),
        "formation": g2.get("summary", {}),
        "persistence": {k: g3.get(k) for k in ["split", "scenes"]},
        "interpretation": "Counts are diagnostics. They do not replace official complete-video MOT scores and do not use GT in inference.",
    }
    (out / "worldobject_error_attribution.json").write_text(json.dumps(err, indent=2, allow_nan=False))

    train_cov = {"scope": "TRAIN", "audit": "see results/worldobject_train", "status": "COMPLETE" if g0 else "BLOCKED"}
    test_manifests = list(Path(a.test_cache).glob("scene_*/manifest.json"))
    test_cov = {"scope": "TEST", "manifest_count": len(test_manifests), "status": "COMPLETE" if len(test_manifests) == 22 else "IN_PROGRESS"}
    (out / "worldobject_all24_train_audit.json").write_text(json.dumps(train_cov, indent=2))
    (out / "worldobject_all22_test_coverage.json").write_text(json.dumps(test_cov, indent=2))
    print(json.dumps({"comparison": str(out / "worldobject_final_comparison.csv"), "routes": len(rows), "test_cache_manifests": len(test_manifests)}, indent=2))


if __name__ == "__main__":
    main()

