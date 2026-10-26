#!/usr/bin/env python3
"""Publish honest V5 progress or final native results without running models."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import io
import json
from pathlib import Path
import time

import numpy as np

from worldmot.runtime.world_object_v5_contract import atomic_json, v5_source_fingerprint, LEGAL_TEST
from worldmot.data.worldobject_cache import sha256

VARIANTS = ("B0", "B1", "B2", "B3", "B4")
METRICS = ("HOTA", "DetA", "AssA", "IDF1", "CVIDF1", "CVMA", "MOTA", "IDSW", "Frag", "CLR_TP", "CLR_FP", "CLR_FN")
NAMES = {"B0": "Frozen V4 C1 reproduction", "B1": "Frozen membership + independent birth/policies",
         "B2": "Trained membership + original geometry features", "B3": "Trained membership + world geometry (main)",
         "B4": "Equal-capacity trained membership without explicit Omega"}


def read(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def save_text(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(value)
    temporary.replace(path)


def save_csv(path, columns, rows):
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    save_text(path, out.getvalue())


def native_values(metrics):
    values = dict(metrics["TrackEval"]["combined"])
    values.update({k: metrics["VisionTrack_MATLAB"]["combined"][k] for k in ("CVIDF1", "CVMA")})
    return {k: values[k] for k in METRICS}


def birth_summary(root):
    scenes = {}
    methods, contrasts = defaultdict(Counter), defaultdict(Counter)
    for scene in (4, 8, 12, 16, 20, 24):
        path = root / f"train_birth_attribution/scene_{scene:02d}/report.json"
        if not path.exists():
            continue
        row = read(path)
        if row["status"] != "COMPLETE_TRAIN_ONLY_CONDITIONAL_BIRTH_DIAGNOSTIC":
            continue
        scenes[str(scene)] = {"timestamps": row["timestamps"], "report_sha256": sha256(path),
                              "methods": row["methods"], "versus_original": row["versus_original"],
                              "counts": row["counts"], "examples": row["examples"]}
        for key, counts in row["methods"].items():
            methods[key].update(counts)
        for key, counts in row["versus_original"].items():
            contrasts[key].update(counts)
    complete = len(scenes) == 6 and sum(x["timestamps"] for x in scenes.values()) == 5860
    return {"status": "COMPLETE_TRAIN6_CONDITIONAL_ATTRIBUTION" if complete else "PARTIAL_TRAIN_ONLY_DIAGNOSTIC",
            "scenes_complete": len(scenes), "timestamps": sum(x["timestamps"] for x in scenes.values()),
            "methods": {k: dict(v) for k, v in methods.items()}, "versus_original": {k: dict(v) for k, v in contrasts.items()},
            "per_scene": scenes, "formal_TEST_metric": False, "original_R2_training_included_these_scenes": True,
            "counterfactual_decisions_advance_history": False, "conditional_on_original_C1_predicted_history": True}


def cached_runtime(root, variant):
    records, manifests = [], []
    if not (root / "predictions" / variant).is_dir():
        return {"status": "NOT_MEASURED", "variant": variant}
    for scene in range(1, 23):
        folder = root / f"predictions/{variant}/scene_{scene:02d}"
        manifest = read(folder / "manifest.json")
        if not manifest or manifest["status"] != "COMPLETE":
            continue
        manifests.append(manifest)
        with (folder / "timings.jsonl").open() as stream:
            records.extend(json.loads(line) for line in stream if line.strip())
    if not records:
        return {"status": "NOT_MEASURED", "variant": variant}
    keys = sorted({key for row in records for key in row if key.endswith("_s")})
    timings = {k: {"P50_s": float(np.percentile([r[k] for r in records if k in r], 50)),
                   "P95_s": float(np.percentile([r[k] for r in records if k in r], 95)),
                   "sum_s": float(sum(r.get(k, 0.) for r in records)),
                   "measured_timestamps": sum(k in r for r in records)} for k in keys}
    total = sum(r["complete_timestamp_s"] for r in records)
    wall = sum(m["worker_wall_s"] for m in manifests)
    return {"status": "COMPLETE_FULL22_CACHED_REPLAY" if len(manifests) == 22 else "PARTIAL_CACHED_REPLAY",
            "variant": variant, "scenes": len(manifests), "paired_times": len(records), "timings": timings,
            "CACHED_single_worker_step_FPS": len(records) / total,
            "CACHED_worker_wall_FPS_including_per_scene_initialization_and_output_hashing": len(records) / wall,
            "LIVE_RGB_to_ID_FPS": None, "precision": "float32", "devices": sorted({m["device"] for m in manifests}),
            "maximum_peak_RSS_bytes": max(m["peak_RSS_bytes"] for m in manifests),
            "maximum_CUDA_peak_allocated_bytes": max(m["CUDA_peak_allocated_bytes"] for m in manifests),
            "multiscene_aggregate_GPU_throughput": False,
            "stage_times_overlap_and_must_not_be_summed": True,
            "step_timer_excludes_timing_log_write_and_final_checksums": True}


def main():
    p = argparse.ArgumentParser()
    for key in ("repo", "runtime", "output-root"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--final", action="store_true")
    a = p.parse_args()
    results = a.repo / "worldmot/results"
    docs = a.repo / "worldmot/docs"
    baseline_path = results / "v4_c1_official_test.json"
    inherited = read(baseline_path)
    controller = read(a.runtime / "controller_state.json", {})
    initial = read(a.runtime / "selection_launches.json", {})
    mirror = read(a.runtime / "train_input_mirror.json", {})
    common = read(a.output_root / "provenance/common_train_budget.json")
    final_gates = {v: read(a.output_root / f"train/{v}/train_gate.json") for v in ("B2", "B3", "B4")}
    official, pending, proofs = {}, [], {}
    for v in VARIANTS:
        path = a.output_root / f"evaluation/{v}/metrics.json"
        metric = read(path)
        if metric and metric.get("status") == "COMPLETE_VALID_SCOPE":
            if (set(metric["evaluated_scenes"]) != set(inherited["TrackEval"]["per_scene"])
                    or metric["raw_GT_modified"] or sorted(x["scene"] for x in metric["coverage"]) != list(range(1, 23))
                    or sum(x["frames"] for x in metric["coverage"]) != 29019
                    or sorted(x["scene"] for x in metric["blocked"]) != sorted(set(range(1, 23)) - set(LEGAL_TEST))
                    or any(x["reason"] != "RAW_GT_DUPLICATE_OR_VERSION_INVALID" for x in metric["blocked"])):
                raise ValueError("FULL22_INFERENCE_AND_RAW_LEGAL14_NATIVE_SCOPE_REQUIRED")
            official[v] = native_values(metric)
            proofs[v] = {"metrics_sha256": sha256(path), "coverage": metric["coverage"],
                         "excluded_raw_GT_scenes": [{"scene": x["scene"], "reason": x["reason"]} for x in metric["blocked"]],
                         "TrackEval_dependency_hashes": metric["TrackEval"]["tool_sha256"],
                         "native_MATLAB_dependency_hashes": metric["VisionTrack_MATLAB"]["toolkit_sha256"],
                         "per_scene_TrackEval": metric["TrackEval"]["per_scene"],
                         "per_scene_MATLAB": metric["VisionTrack_MATLAB"]["per_scene"],
                         "raw_GT_modified": metric["raw_GT_modified"]}
        else:
            pending.append(v)
    birth = birth_summary(a.output_root)
    attribution_path = a.output_root / "v5_identity_attribution.json"
    attribution = read(attribution_path)
    live = read(a.output_root / "live_fixed_train_prefix/runtime.json")
    final_ready = (not pending and attribution is not None and attribution.get("status") == "COMPLETE_FROZEN_ALL5_FULL22_IDENTITY_ATTRIBUTION"
                   and birth["status"] == "COMPLETE_TRAIN6_CONDITIONAL_ATTRIBUTION"
                   and live is not None and live.get("status") == "ALL5_FINITE_REAL_LIVE_COMPLETE")
    if a.final and not final_ready:
        raise ValueError("ALL5_NATIVE_SCORES_BIRTH_DIAGNOSTICS_POSTHOC_AND_REAL_LIVE_REQUIRED")
    now = datetime.now(timezone.utc).isoformat()
    progress = {v: {phase: read(a.output_root / f"train/{v}/{phase}/progress.json") for phase in ("selection", "final")}
                for v in ("B2", "B3", "B4")}
    state = {"status": "FULL_RESEARCH_ARTIFACTS_COMPLETE" if final_ready else "REGISTERED_RESEARCH_IN_PROGRESS",
             "updated_UTC": now, "v5_source_sha256": v5_source_fingerprint(),
             "base_commit": "eda0761b15f4049ead1971facd17d846bbab94cf", "variants": NAMES,
             "TRAIN24_inputs_verified": mirror.get("status") == "TRAIN24_FROZEN_INPUTS_VERIFIED",
             "TRAIN_paired_times": mirror.get("timestamps"), "TRAIN_candidates": mirror.get("observations", 557002),
             "training_progress": progress, "common_training_budget": common,
             "selection_devices": {k: v["device"] for k, v in initial.get("jobs", {}).items()},
             "controller_stage": controller.get("stage"), "controller_last_exception": controller.get("last_exception"),
             "final_checkpoint_SHA256": {v: g["checkpoint_sha256"] if g else None for v, g in final_gates.items()},
             "official_TEST_complete_variants": list(official), "official_TEST_pending_variants": pending,
             "TEST_inference_scene_counts": controller.get("completed_TEST_scenes", {v: 0 for v in VARIANTS}),
             "birth_diagnostic_complete_scenes": birth["scenes_complete"], "posthoc_complete": attribution is not None,
             "real_RGB_to_ID_measured": live is not None and live.get("status") == "ALL5_FINITE_REAL_LIVE_COMPLETE",
             "no_TEST_parameter_selection": True, "negative_results_preserved": True,
             "scientific_improvement_claimed": False}
    atomic_json(results / "v5_experiment_status.json", state)
    atomic_json(results / "v5_train_birth_attribution.json", birth)
    baselines = {k: {m: row[m] for m in METRICS} for k, row in inherited["baselines"].items()}
    deltas = {}
    if "B3" in official:
        deltas = {name: {key: official["B3"][key] - values[key] for key in METRICS}
                  for name, values in {**baselines, **{v: row for v, row in official.items() if v != "B3"}}.items()}
    b0_reproduced = None
    if "B0" in official:
        b0_reproduced = all(abs(official["B0"][k] - baselines["V4 C1"][k]) <= 1e-8 for k in METRICS)
        if not b0_reproduced:
            raise ValueError("FROZEN_B0_NATIVE_METRICS_DIFFER_FROM_ARCHIVED_C1")
    science_success = (final_ready and all(official["B3"][k] > baselines["GMT Native"][k] for k in ("HOTA", "AssA", "CVIDF1")))
    state["scientific_improvement_claimed"] = bool(science_success)
    atomic_json(results / "v5_experiment_status.json", state)
    report = {"status": "COMPLETE_NATIVE_LEGAL14_ALL5" if not pending else "OFFICIAL_V5_RESULTS_PENDING",
              "updated_UTC": now, "official_legal_scene_ids": LEGAL_TEST, "full_inference_scenes_per_variant": 22,
              "full_inference_paired_times_per_variant": 29019, "frozen_candidates_per_variant": 548583,
              "values_are_historical_references_until_V5_completes": bool(pending),
              "historical_baselines": baselines, "V5_official_metrics": official, "pending_variants": pending,
              "B3_minus_reference": deltas, "B0_native_reproduction_passed": b0_reproduced,
              "baseline_source_sha256": sha256(baseline_path), "proofs": proofs,
              "baseline_tool_comparability": inherited["baseline_tool_comparability"],
              "shared_tracking_dependency_hashes_equal_in_archived_comparison": inherited["shared_native_tracking_dependency_hashes_equal_to_baselines"],
              "checkpoint_SHA256": state["final_checkpoint_SHA256"], "v5_source_sha256": state["v5_source_sha256"],
              "TEST_parameter_selection": False, "raw_GT_modified": False,
              "exploratory_TEST_previously_informed_design": True,
              "scientific_target_exceeded_GMT_on_HOTA_AssA_CVIDF1": bool(science_success),
              "full_research_delivery_complete": final_ready}
    atomic_json(results / "v5_official_test.json", report)
    metric_rows = [{"method": k, "kind": "historical_reference", **v} for k, v in baselines.items()]
    metric_rows += [{"method": v, "kind": "V5_formal_TEST" if v in official else "PENDING_NO_SCORE", **official.get(v, {})} for v in VARIANTS]
    save_csv(results / "v5_official_metrics.csv", ("method", "kind", *METRICS), metric_rows)
    runtimes = {v: cached_runtime(a.output_root, v) for v in VARIANTS}
    runtime_rows = []
    for v, cached in runtimes.items():
        row = {"method": v, "CACHED_FPS": cached.get("CACHED_single_worker_step_FPS"),
               "CACHED_worker_wall_FPS": cached.get("CACHED_worker_wall_FPS_including_per_scene_initialization_and_output_hashing"),
               "cached_paired_times": cached.get("paired_times", 0), "REAL_RGB_to_ID_FPS": None,
               "real_paired_times": 0, "real_scope": "NOT_MEASURED"}
        if live and v in live.get("variants", {}):
            measured = live["variants"][v]
            row.update(REAL_RGB_to_ID_FPS=measured["LIVE_RGB_to_final_ID_FPS"], real_paired_times=measured["paired_times"],
                       real_scope="TRAIN4_8_12_FIRST61_EACH_REAL_RGB_NOT_FULL_TEST_FPS")
        runtime_rows.append(row)
    atomic_json(results / "v5_runtime.json", {"cached": runtimes, "real_live": live, "actual_full_TEST_RGB_to_ID_FPS": None,
                "full_TEST_runs_use_frozen_perception_caches": True,
                "LIVE_input_geometry_has_actual_current_transform_unavailable_in_inherited_cache": True,
                "LIVE_accuracy_not_compared_to_cached_full_TEST": True, "six_GPU_total_throughput_not_single_video_FPS": True})
    save_csv(results / "v5_runtime.csv", ("method", "CACHED_FPS", "CACHED_worker_wall_FPS", "cached_paired_times", "REAL_RGB_to_ID_FPS", "real_paired_times", "real_scope"), runtime_rows)
    if attribution:
        atomic_json(results / "v5_identity_attribution.json", attribution)
        atomic_json(results / "v5_failure_cases.json", {"status": "COMPLETE_FROZEN_TEST_POSTHOC", "examples": attribution["examples"],
                    "raw_prediction_history_rewritten": False, "original_GT_modified": False})
    else:
        atomic_json(results / "v5_failure_cases.json", {"status": "TEST_POSTHOC_NOT_YET_AVAILABLE",
                    "TRAIN_conditional_examples": [e for s in birth["per_scene"].values() for e in s["examples"]],
                    "formal_TEST_failures": None})
    lines = ["WorldMOT V5: decoupled observation birth and persistent membership", "",
             "Updated UTC: " + now + ". Status: **" + state["status"] + "**.", "",
             "The finite B0–B4 registration, code fingerprint and original V3/V4 artifacts are frozen. "
             "The model predicts membership and a per-camera no-observation option. Birth, lifecycle, "
             "identity writes and C1 Kalman/Joseph updates use separate auditable policies. GT never repairs the predicted history.", "",
             "Source SHA-256: `" + state["v5_source_sha256"] + "`.", "",
             "All24 TRAIN inputs are verified: 28,754 paired times and 557,002 detections. "
             "Fit18/validation6 selection uses 12 epochs with the fixed common 16-epoch extension rule. "
             "Final weights start fresh and must prove all24 coverage in every epoch.", "",
             "| Variant | Selection completed epochs | Current selection scene count | Final completed epochs | Final weight SHA-256 |",
             "|---|---:|---:|---:|---|"]
    for v in ("B2", "B3", "B4"):
        selection = progress[v]["selection"] or {}
        final = progress[v]["final"] or {}
        lines.append(f"| {v} | {selection.get('completed_epochs', 0)} | {selection.get('current_epoch_complete_scenes', 0)}/18 | {final.get('completed_epochs', 0)} | {state['final_checkpoint_SHA256'][v] or 'not available'} |")
    lines += ["", "Selection currently uses CPU, one Torch/BLAS thread per worker. "
              "Final B2/B3/B4 training and complete TEST use truly idle V100 FP32 workers. "
              "Other users' processes are never stopped. Independent-scene concurrency is not single-stream FPS.", "",
              "Official MC-MOT uses original legal14 full TEST videos, after full22 causal inference "
              "(29,019 paired times / 548,583 shared candidates). Invalid8 remain separate. "
              "The following historical reference numbers are not new V5 measurements.", "",
              "| Method | HOTA | DetA | AssA | IDF1 | CVIDF1 | CVMA | MOTA | IDSW | Frag | TP | FP | FN |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in metric_rows:
        values = [str(row[k]) if k in ("IDSW", "Frag", "CLR_TP", "CLR_FP", "CLR_FN") else f"{row[k]:.4f}" for k in METRICS] if "HOTA" in row else ["pending"] * len(METRICS)
        lines.append("| " + row["method"] + " | " + " | ".join(values) + " |")
    if "B3" in official:
        lines += ["", "B3 differences are in percentage points for percentage metrics and signed counts for event totals.", "",
                  "| Reference | ΔHOTA | ΔAssA | ΔIDF1 | ΔCVIDF1 | ΔMOTA | ΔIDSW |", "|---|---:|---:|---:|---:|---:|---:|"]
        for name, diff in deltas.items():
            lines.append(f"| {name} | {diff['HOTA']:+.4f} | {diff['AssA']:+.4f} | {diff['IDF1']:+.4f} | {diff['CVIDF1']:+.4f} | {diff['MOTA']:+.4f} | {diff['IDSW']:+d} |")
    lines += ["", f"TRAIN conditional birth audit: {birth['scenes_complete']}/6 complete scenes, {birth['timestamps']} paired times. "
              "Only original C1 decisions update history; alternative births are discarded after comparison. "
              "These are TRAIN clean-label diagnostic events, not HOTA or CVIDF1.", ""]
    for name, counts in birth["versus_original"].items():
        lines.append(f"{name}: same-person pair corrections {counts.get('same_person_shared/corrections', 0)}, "
                     f"regressions {counts.get('same_person_shared/regressions', 0)}, net {counts.get('same_person_shared/net', 0):+d}; "
                     f"different-person pair net {counts.get('different_person_separate/net', 0):+d}; "
                     f"false-birth avoidance net {counts.get('false_birth_avoided/net', 0):+d}.")
        lines.append("")
    lines += ["B1−B0 also changes explicit lifecycle and memory-write policy, so its full online delta cannot "
              "be assigned solely to Birth. B2−B1 includes a new trained head and capacity/budget changes. "
              "B3−B4 is the equal-capacity, equal-budget explicit-geometry comparison.", "",
              "The inherited cache has world-space person supports but lacks the rotation/translation "
              "needed to move its local camera matrices into that same world gauge. Reprojection explicitly "
              "abstains except at verified reference transforms. Fit18 calibration found verified cameras "
              "at 18/22,894 paired times. This limits what B3 can establish about reprojection. "
              "New real LIVE timing uses actual current transforms and is not a full-video accuracy comparison.", "",
              "Cached replay speed and real RGB-to-final-ID speed are separate. The finite real timing scope "
              "is TRAIN scenes4/8/12, first61 paired times each, after5 warm-up times and a state reset. "
              "It includes actual RGB read, detector, per-frame Ω (except B4), alignment, membership, C1 updates "
              "and final ID serialization/flush. It is not full TEST RGB-to-ID FPS.", "",
              "| Method | Cached worker step FPS | Cached worker wall FPS | Real RGB-to-ID FPS | Real scope |",
              "|---|---:|---:|---:|---|"]
    for row in runtime_rows:
        values = [f"{row[k]:.4f}" if row[k] is not None else "not measured" for k in ("CACHED_FPS", "CACHED_worker_wall_FPS", "REAL_RGB_to_ID_FPS")]
        lines.append("| " + row["method"] + " | " + " | ".join(values) + " | " + row["real_scope"] + " |")
    lines += ["", "Native tool dependencies and wrapper comparability are retained from the archived V4 report. "
              "Historical GMT/V2 exports used two-decimal coordinates; current V3/V4/V5 exports preserve .17g. "
              "Original GT and shared native tracking dependencies were verified in the archived comparison. "
              "This is a documented system-reference limitation, not permission to alter GT or re-tune TEST.", ""]
    if final_ready:
        lines.append("The complete registered research artifacts are available. " + (
            "B3 exceeds GMT Native on all three preregistered main metrics (HOTA, AssA, CVIDF1)." if science_success else
            "The scientific target of exceeding GMT on HOTA, AssA and CVIDF1 was not achieved; the exact differences above are retained."))
    else:
        lines.append("V5 research remains in progress. No V5 improvement claim is made before the registered full training and formal TEST finish.")
    lines.append("")
    save_text(docs / "WORLDMOT_V5_DECOUPLED_BIRTH.md", "\n".join(lines))
    if a.final:
        atomic_json(a.output_root / "v5_research_delivery_receipt.json", {"status": "FULL_REGISTERED_RESEARCH_ARTIFACTS_COMPLETE",
                    "scientific_target_reached": bool(science_success), "v5_source_sha256": state["v5_source_sha256"],
                    "public_reports_sha256": {p.name: sha256(p) for p in sorted(results.glob("v5_*.json"))}, "created_UTC": now,
                    "GitHub_publication_still_requires_verification": True})
    print(json.dumps({"status": state["status"], "official_variants": list(official), "birth_scenes": birth["scenes_complete"], "updated_UTC": now}), flush=True)


if __name__ == "__main__":
    main()
