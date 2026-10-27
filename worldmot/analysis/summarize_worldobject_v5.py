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
from worldobject_v5_identity_summary import (
    build_identity_summary, write_identity_summary, COMPLETE as IDENTITY_ANALYSIS_COMPLETE)

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


def frozen_native_for_reporting(root, runtime, variant):
    """Read completed control scores without adopting a new-model TEST scope."""
    if variant not in ("B0", "B1"):
        return None
    folder = root / "evaluation_frozen_references" / variant
    receipt_path = folder / "frozen_scoring_receipt.json"
    if not receipt_path.is_file():
        return None
    from worldobject_v5_frozen_reference_scoring import require_publication, validate_metric, paths, FULL_SHA
    scope = require_publication(root, runtime, variant)
    receipt = read(receipt_path)
    path = folder / "metrics.json"
    inputs_path = root / "provenance/frozen_reference_score_inputs" / (variant + ".json")
    inputs = read(inputs_path)
    if (receipt.get("status") != "COMPLETE_FROZEN_REFERENCE_FULL22_NATIVE14"
            or receipt.get("variant") != variant or receipt.get("metrics_sha256") != sha256(path)
            or receipt.get("input_byte_verification_sha256") != sha256(inputs_path)
            or receipt.get("checkpoint_sha256") != FULL_SHA
            or receipt.get("scope_sha256") != sha256(paths(root)[0])
            or receipt.get("authorization_sha256") != sha256(paths(root)[1])
            or receipt.get("original_annotations_sha256") != scope["original_TEST_annotations_sha256"]
            or receipt.get("original_GT_modified") is not False
            or receipt.get("TRAIN_parameters_or_budget_rules_changed") is not False
            or inputs.get("status") != "ALL22_FROZEN_REFERENCE_OUTPUT_BYTES_VERIFIED_BEFORE_GT"
            or inputs.get("variant") != variant or inputs.get("GT_read") is not False
            or inputs.get("source_sha256") != scope["v5_source_sha256"]
            or inputs.get("scope_sha256") != receipt["scope_sha256"]
            or set(inputs["scenes"]) != {str(s) for s in range(1, 23)}
            or sum(x["candidates"] for x in inputs["scenes"].values()) != 548583):
        raise ValueError("COMPLETED_FROZEN_NATIVE_RECEIPT_AND_EXACT_INPUTS_REQUIRED_FOR_REPORT")
    metric = read(path)
    audit = Path(scope["splits"]["test"]["support_cache"]).parents[1] / "fulltrain_20261022/audit/data_audit.json"
    validate_metric(metric, inputs, read(audit))
    return path, metric, {"kind": "FROZEN_CONTROL_FORMAL_TEST_BEFORE_NEW_MODEL_FINAL_TRAIN",
        "scope_sha256": receipt["scope_sha256"], "authorization_sha256": receipt["authorization_sha256"],
        "input_byte_verification_sha256": receipt["input_byte_verification_sha256"],
        "native_scoring_receipt_sha256": sha256(receipt_path), "checkpoint_sha256": FULL_SHA,
        "canonical_all3_final_gate_adoption_complete": False,
        "full24_eligibility": "original frozen R2 head; no new membership training"}


def frozen_control_report(results, docs, official, proofs, origins, baselines, now):
    controls = {v: official[v] for v in ("B0", "B1") if v in official}
    if not controls:
        return
    reference = controls.get("B0", baselines["V4 C1"])
    reference_label = "newly scored B0" if "B0" in controls else "archived V4 C1"
    delta = {k: controls["B1"][k] - reference[k] for k in METRICS} if "B1" in controls else None
    per_scene = {}
    if "B1" in controls and "B0" in controls:
        for name, values in proofs["B1"]["per_scene_TrackEval"].items():
            row = {k: values[k] for k in METRICS if k not in ("CVIDF1", "CVMA")}
            row.update({k: proofs["B1"]["per_scene_MATLAB"][name][k] for k in ("CVIDF1", "CVMA")})
            old = proofs["B0"]["per_scene_TrackEval"][name]
            old = {**old, **{k: proofs["B0"]["per_scene_MATLAB"][name][k] for k in ("CVIDF1", "CVMA")}}
            per_scene[name] = {"B1": row, "B0": {k: old[k] for k in METRICS},
                "B1_minus_B0": {k: row[k] - old[k] for k in METRICS}}
    report = {"status": "COMPLETE_B0_B1_FULL22_NATIVE14" if len(controls) == 2 else "PARTIAL_FROZEN_CONTROL_NATIVE_RESULTS",
        "updated_UTC": now, "metrics": controls, "native_score_origins": {v: origins[v] for v in controls},
        "metric_sha256": {v: proofs[v]["metrics_sha256"] for v in controls},
        "B1_minus_reference": delta, "reference": reference_label,
        "B1_minus_GMT": {k: controls["B1"][k] - baselines["GMT Native"][k] for k in METRICS}
            if "B1" in controls else None,
        "per_scene": per_scene,
        "CVIDF1_scene_improvements": sum(x["B1_minus_B0"]["CVIDF1"] > 1e-8 for x in per_scene.values()),
        "CVIDF1_scene_regressions": sum(x["B1_minus_B0"]["CVIDF1"] < -1e-8 for x in per_scene.values()),
        "scope_per_method": {"inference_videos": 22, "paired_times": 29019, "frozen_candidates": 548583,
            "native_original_legal_scenes": LEGAL_TEST, "excluded_raw_GT_scenes": sorted(set(range(1, 23)) - set(LEGAL_TEST))},
        "B0_exact_C1_all_times": "independently verified during inference; not inferred from aggregate scores",
        "metrics_are_new_formal_control_measurements": True,
        "B1_is_dedicated_trained_B3_model": False, "B1_minus_B0_is_birth_only_causal_effect": False,
        "B3_minus_B4_geometry_independent_effect": "PENDING_FINAL_FULL24_TRAIN_AND_TEST",
        "new_TEST_event_attribution_run": False, "native_per_scene_aggregates_only": True,
        "TRAIN_model_parameters_or_budget_rule_changed_from_TEST": False, "original_GT_modified": False,
        "B1_output_boxes_scores_ROI_preserved": True,
        "small_CLEAR_detection_count_changes_do_not_imply_detector_changed": True,
        "identity_conditioned_matching_mechanism_check": read(results / "v5_identity_conditioned_metrics_check.json"),
        "native_interleaved_CLEAR_switches_are_not_TrackEval_IDSW": True}
    atomic_json(results / "v5_frozen_reference_native_scores.json", report)
    save_csv(results / "v5_frozen_reference_native_scores.csv", ("method", *METRICS),
             [{"method": v, **values} for v, values in controls.items()])
    lines = ["WorldMOT V5 frozen controls: actual full-video native TEST results", "", "Updated UTC: " + now + ".", "",
        "Each reported control completed all22 videos,29,019 paired times and548,583 frozen candidates. "
        "TrackEval and the unchanged VisionTrack MATLAB toolkit scored the original legal14 videos. "
        "The other8 were fully inferred and excluded for original GT defects. No GT was cleaned. "
        "The early scoring order was published before reading these scores; all B2/B3/B4 training rules remain frozen.", "",
        "| Method | HOTA | DetA | AssA | IDF1 | CVIDF1 | CVMA | MOTA | IDSW | Frag | TP | FP | FN |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in {"GMT Native (historical)": baselines["GMT Native"], "V4 C1 (historical)": baselines["V4 C1"], **controls}.items():
        cells = [str(row[k]) if k in ("IDSW", "Frag", "CLR_TP", "CLR_FP", "CLR_FN") else f"{row[k]:.4f}" for k in METRICS]
        lines.append("| " + name + " | " + " | ".join(cells) + " |")
    if delta:
        lines += ["", f"B1 minus {reference_label}: HOTA {delta['HOTA']:+.4f}, AssA {delta['AssA']:+.4f}, "
            f"IDF1 {delta['IDF1']:+.4f}, CVIDF1 {delta['CVIDF1']:+.4f}, CVMA {delta['CVMA']:+.4f}, "
            f"MOTA {delta['MOTA']:+.4f} percentage points; IDSW {delta['IDSW']:+d}, Frag {delta['Frag']:+d}.", "",
            "This is a negative aggregate result for replacing the old policies while retaining the old "
            "identity head. B1 also changes history assignment, lifecycle, spatial and appearance-write "
            "policies, so the full delta cannot isolate Birth alone. B2/B3/B4 use a newly trained dedicated "
            "membership head; their complete results remain pending. These control scores do not establish "
            "the independent value of Omega geometry.", "",
            f"B1 IDF1 minus CVIDF1 is {controls['B1']['IDF1'] - controls['B1']['CVIDF1']:.4f} points; "
            f"the reference gap is {reference['IDF1'] - reference['CVIDF1']:.4f}. "
            "Together with the CVIDF1 regression, this is consistent with poorer cross-camera identity "
            "agreement. It does not identify exact false births, wrong reuses or recovery failures. "
            "The registered all110-prediction posthoc analysis will count those events after final model TEST.", "",
            "The frozen detections were preserved. Small CLEAR TP/FP/FN differences can arise while "
            "the candidates remain identical; these results are not evidence of a detector change. "
            "CLEAR prioritizes prior identities and HOTA uses global identity alignment before matching. "
            "A generated identical-boxes check reproduces changes in CLEAR counts and DetA from IDs "
            "alone; it does not locate the real seven events. See V5_IDENTITY_CONDITIONED_DETECTION_COUNTS.md. "
            "TrackEval IDSW is the switch metric in the table. VisionTrack's interleaved-camera CLEAR "
            "switch count has a different definition and must not replace it.", ""]
    if per_scene:
        lines += [f"CVIDF1 improves in {report['CVIDF1_scene_improvements']}/14 scenes and regresses in "
            f"{report['CVIDF1_scene_regressions']}/14. Per-scene aggregates below do not alter model selection.", "",
            "| Scene | Delta HOTA | Delta AssA | Delta IDF1 | Delta CVIDF1 | Delta IDSW |",
            "|---|---:|---:|---:|---:|---:|"]
        for name, row in per_scene.items():
            d = row["B1_minus_B0"]
            lines.append(f"| {name} | {d['HOTA']:+.4f} | {d['AssA']:+.4f} | {d['IDF1']:+.4f} | {d['CVIDF1']:+.4f} | {d['IDSW']:+d} |")
        lines += ["", "The combined metric is the native combined result, not an unweighted mean of this table.", ""]
    for variant in controls:
        lines += [f"{variant} metrics SHA-256: `{proofs[variant]['metrics_sha256']}`.", ""]
    lines += ["Both controls use frozen R2 head SHA-256 "
        "`cfda40050e89b31dd4480ca4d5010e23067a94ba18ea7d0e2425d5935d3f7fb8`. "
        "No optimizer, checkpoint selection, new-model TEST or TEST parameter search is authorized by "
        "the control-only scoring scope. Cached replay FPS is reported separately; real RGB-to-ID timing is pending.", ""]
    save_text(docs / "V5_FROZEN_REFERENCE_NATIVE_RESULTS.md", "\n".join(lines))


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


def policy_summary(root):
    scenes, totals, paired, global_pairs = {}, defaultdict(Counter), Counter(), Counter()
    base = root / "train_frozen_policy_audit"
    if base.is_dir():
        for scene in (4, 8, 12, 16, 20, 24):
            path = base / f"scene_{scene:02d}/report.json"
            row = read(path)
            if not row or row.get("status") != "COMPLETE_TRAIN_ONLY_FROZEN_POLICY_CLOSED_LOOPS":
                continue
            scenes[str(scene)] = {"timestamps": row["timestamps"], "report_sha256": sha256(path),
                "counts": row["counts"], "paired_B1_minus_C1": row["paired_B1_minus_C1"],
                "global_one_to_one_paired": row["global_one_to_one_paired"], "identity_graphs": row["identity_graphs"],
                "device": row["device"], "examples": row["examples"]}
            for key, counts in row["counts"].items():
                totals[key].update(counts)
            paired.update(row["paired_B1_minus_C1"])
            global_pairs.update(row["global_one_to_one_paired"])
    complete = len(scenes) == 6 and sum(x["timestamps"] for x in scenes.values()) == 5860
    return {"status": "COMPLETE_TRAIN6_FROZEN_POLICY_AUDIT" if complete else "PARTIAL_TRAIN_FROZEN_POLICY_AUDIT",
            "scenes_complete": len(scenes), "timestamps": sum(x["timestamps"] for x in scenes.values()),
            "counts": {k: dict(v) for k, v in totals.items()}, "paired_B1_minus_C1": dict(paired),
            "global_one_to_one_paired": dict(global_pairs), "per_scene": scenes,
            "independent_predicted_histories": True, "frozen_R2_head_previously_saw_these_TRAIN_scenes": True,
            "formal_TEST_metric": False, "birth_only_causal_effect": False, "optimizer_created": False,
            "budget_or_model_selection_uses_this_audit": False}


def training_curves(root):
    def scene_summary(scene):
        n = scene["timestamps"]
        d = scene["diagnostics"]
        same = d.get("same_person_cross_view_pairs", 0)
        different = d.get("different_person_cross_view_pairs", 0)
        return {"scene": scene["scene"], "paired_times": n,
                "mean_detections_per_time": d.get("current_detections", 0) / max(1, n),
                "per_camera_membership_loss_per_time": scene["loss_sums"]["per_camera_membership"] / max(1, n),
                "same_person_pairs": same, "same_person_splits": d.get("same_person_split", 0),
                "same_person_split_rate": d.get("same_person_split", 0) / max(1, same),
                "different_person_pairs": different, "false_merges": d.get("false_cross_view_merge", 0),
                "different_person_false_merge_rate": d.get("false_cross_view_merge", 0) / max(1, different),
                "mean_mixed_historical_objects": d.get("mixed_historical_objects_before", 0) / max(1, n)}
    variants = {}
    for v in ("B2", "B3", "B4"):
        variants[v] = {}
        for phase in ("selection", "final"):
            path = root / f"train/{v}/{phase}/history.json"
            history = read(path, [])
            rows = []
            for epoch in history:
                coverage = epoch.get("fit_coverage", epoch.get("coverage", []))
                rows.append({"epoch": epoch["epoch"], "train": epoch["train"], "validation": epoch.get("validation"),
                             "complete_training_scenes": len(coverage), "complete_training_timestamps": sum(x["timestamps"] for x in coverage),
                             "gradient_finite": all(x["gradient_finite"] for x in coverage),
                             "maximum_unclipped_gradient_norm": max((x["gradient_norm_max"] or 0 for x in coverage), default=None),
                             "scene_internal_ID_resets": sum(x["scene_reset_count"] - 1 for x in coverage),
                             "per_train_scene": [scene_summary(x) for x in coverage],
                             "per_validation_scene": [scene_summary(x) for x in epoch.get("validation_coverage", [])]})
            variants[v][phase] = rows
    return {"status": "TRAIN_ONLY_CURVES", "formal_TEST_metrics": False, "variants": variants,
            "selection_rule": "registered common B3 epoch12 convergence rule only; no TEST feedback",
            "training_history": "genuinely predicted from scene origin; no GT ID repair"}


def cached_runtime(root, variant):
    records, manifests = [], []
    storage_modes = read(root / "provenance/prediction_output_storage_modes.json", {}).get("per_variant_scene", {})
    storage_counts = Counter()
    if not (root / "predictions" / variant).is_dir():
        return {"status": "NOT_MEASURED", "variant": variant}
    for scene in range(1, 23):
        folder = root / f"predictions/{variant}/scene_{scene:02d}"
        manifest = read(folder / "manifest.json")
        if not manifest or manifest["status"] != "COMPLETE":
            continue
        manifests.append(manifest)
        storage_counts[storage_modes.get(variant, {}).get(str(scene), "UNRECORDED")] += 1
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
            "step_timer_excludes_timing_log_write_and_final_checksums": True,
            "original_output_storage_scene_counts": dict(storage_counts),
            "uniform_recorded_local_output_storage": set(storage_counts) == {"LOCAL_DURABLE"},
            "B0_storage_differs_from_other_methods_after_NAS_recovery": True,
            "cached_FPS_not_a_controlled_algorithm_speed_ranking": True}


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
    native_smoke = read(results / "v5_native_protocol_checks.json")
    cuda_head_check = read(results / "v5_cuda_head_preflight.json")
    controller = read(a.runtime / "controller_state.json", {})
    reference_status = read(a.runtime / "reference_inference_status.json", {})
    reference_registration = read(a.output_root / "provenance/reference_test_registration.json")
    reference_adoption = read(a.output_root / "provenance/reference_test_adoption_complete.json")
    initial = read(a.runtime / "selection_launches.json", {})
    mirror = read(a.runtime / "train_input_mirror.json", {})
    common = read(a.output_root / "provenance/common_train_budget.json")
    final_gates = {v: read(a.output_root / f"train/{v}/train_gate.json") for v in ("B2", "B3", "B4")}
    official, pending, proofs, score_origins = {}, [], {}, {}
    canonical_complete = []
    for v in VARIANTS:
        path = a.output_root / f"evaluation/{v}/metrics.json"
        metric = read(path)
        if metric and metric.get("status") == "COMPLETE_VALID_SCOPE":
            canonical_complete.append(v)
            score_origins[v] = {"kind": "CANONICAL_REGISTERED_V5_FORMAL_TEST"}
        else:
            frozen = frozen_native_for_reporting(a.output_root, a.runtime, v)
            if frozen:
                path, metric, score_origins[v] = frozen
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
    policy = policy_summary(a.output_root)
    curves = training_curves(a.output_root)
    attribution_path = a.output_root / "v5_identity_attribution.json"
    attribution = read(attribution_path)
    live = read(a.output_root / "live_fixed_train_prefix/runtime.json")
    if attribution and attribution.get("analysis_sha256") != sha256(
            a.repo / "worldmot/analysis/attribute_worldobject_v5_frozen_test.py"):
        raise ValueError("COMPLETED_IDENTITY_ATTRIBUTION_SOURCE_CHANGED")
    identity_official = {
        "status": "COMPLETE_NATIVE_LEGAL14_ALL5" if not pending else "PENDING",
        "canonical_TEST_scope_complete_variants": canonical_complete,
        "raw_GT_modified": False, "TEST_parameter_selection": False,
        "v5_source_sha256": v5_source_fingerprint(),
        "V5_official_metrics": official, "proofs": proofs,
        "checkpoint_SHA256": {v: g["checkpoint_sha256"] if g else None for v, g in final_gates.items()},
        "historical_baselines": inherited["baselines"],
    }
    identity_summary = build_identity_summary(attribution, identity_official, final_gates, common, birth)
    if attribution:
        identity_summary["source_attribution_sha256"] = sha256(attribution_path)
        identity_summary["source_native_metrics_sha256"] = {v: proofs[v]["metrics_sha256"] for v in VARIANTS}
    final_ready = (not pending and canonical_complete == list(VARIANTS)
                   and attribution is not None and attribution.get("status") == "COMPLETE_FROZEN_ALL5_FULL22_IDENTITY_ATTRIBUTION"
                   and identity_summary["status"] == IDENTITY_ANALYSIS_COMPLETE
                   and birth["status"] == "COMPLETE_TRAIN6_CONDITIONAL_ATTRIBUTION"
                   and policy["status"] == "COMPLETE_TRAIN6_FROZEN_POLICY_AUDIT"
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
             "native_score_origins": score_origins, "canonical_TEST_scope_complete_variants": canonical_complete,
             "TEST_inference_scene_counts": controller.get("completed_TEST_scenes", {v: 0 for v in VARIANTS}),
             "frozen_reference_GT_free_inference": {"status": reference_status.get("status", "NOT_STARTED"),
                 "complete_scenes": reference_status.get("complete_scenes", {"B0": 0, "B1": 0}),
                 "scoring_allowed_by_subset": False,
                 "separate_frozen_control_evaluation_authorization_registered":
                    (a.output_root / "provenance/frozen_reference_scoring_registration.json").is_file()},
             "birth_diagnostic_complete_scenes": birth["scenes_complete"], "posthoc_complete": attribution is not None,
             "identity_analysis_report_status": identity_summary["status"],
             "frozen_policy_TRAIN_audit_complete_scenes": policy["scenes_complete"],
             "real_RGB_to_ID_measured": live is not None and live.get("status") == "ALL5_FINITE_REAL_LIVE_COMPLETE",
             "no_TEST_parameter_selection": True, "negative_results_preserved": True,
             "scientific_improvement_claimed": False}
    state["native_synthetic_toolchain_passed"] = bool(native_smoke and native_smoke.get("status") == "PASSED_SYNTHETIC_TRACKEVAL_AND_NATIVE_MATLAB_PROTOCOL")
    atomic_json(results / "v5_experiment_status.json", state)
    atomic_json(results / "v5_train_birth_attribution.json", birth)
    atomic_json(results / "v5_train_frozen_policy_audit.json", policy)
    atomic_json(results / "v5_training_curves.json", curves)
    if reference_registration:
        atomic_json(results / "v5_reference_inference_registration.json", reference_registration)
    if reference_adoption:
        atomic_json(results / "v5_reference_adoption.json", reference_adoption)
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
    report = {"status": "COMPLETE_NATIVE_LEGAL14_ALL5" if not pending else
              "PARTIAL_COMPLETE_NATIVE_LEGAL14_VARIANTS" if official else "OFFICIAL_V5_RESULTS_PENDING",
              "updated_UTC": now, "official_legal_scene_ids": LEGAL_TEST, "full_inference_scenes_per_variant": 22,
              "full_inference_paired_times_per_variant": 29019, "frozen_candidates_per_variant": 548583,
              "values_are_historical_references_until_V5_completes": not bool(official),
              "historical_baselines": baselines, "V5_official_metrics": official, "pending_variants": pending,
              "native_score_origins": score_origins, "canonical_TEST_scope_complete_variants": canonical_complete,
              "B1_minus_B0": {k: official["B1"][k] - official["B0"][k] for k in METRICS}
                  if "B0" in official and "B1" in official else None,
              "B3_minus_reference": deltas, "B0_native_reproduction_passed": b0_reproduced,
              "baseline_source_sha256": sha256(baseline_path), "proofs": proofs,
              "baseline_tool_comparability": inherited["baseline_tool_comparability"],
              "shared_tracking_dependency_hashes_equal_in_archived_comparison": inherited["shared_native_tracking_dependency_hashes_equal_to_baselines"],
              "checkpoint_SHA256": state["final_checkpoint_SHA256"], "v5_source_sha256": state["v5_source_sha256"],
              "explicit_reference_scope_adoption": reference_adoption,
              "TEST_parameter_selection": False, "raw_GT_modified": False,
              "exploratory_TEST_previously_informed_design": True,
              "scientific_target_exceeded_GMT_on_HOTA_AssA_CVIDF1": bool(science_success),
              "full_research_delivery_complete": final_ready}
    atomic_json(results / "v5_official_test.json", report)
    frozen_control_report(results, docs, official, proofs, score_origins, baselines, now)
    metric_rows = [{"method": k, "kind": "historical_reference", **v} for k, v in baselines.items()]
    metric_rows += [{"method": v, "kind": score_origins[v]["kind"] if v in official else "PENDING_NO_SCORE", **official.get(v, {})} for v in VARIANTS]
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
    write_identity_summary(identity_summary, results, docs, live)
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
    waiting = [v for v in ("B2", "B3", "B4") if (progress[v]["selection"] or {}).get("complete") and common is None]
    if waiting:
        lines += ["", "Completed base12 selection, waiting for B3's common12/16 decision: " + ", ".join(waiting) +
                  ". These fit18/VAL6 checkpoints are not final fresh-all24 weights. "
                  "See v5_B4_base12_selection_completion.json for the independently read-back B4 archive.", ""]
    lines += ["", "Latest completed selection epochs below are TRAIN diagnostics. Loss is averaged over "
              "paired timestamps; scene density and evolving predicted histories differ between fit18 and VAL6. "
              "The per-scene breakdown is in v5_training_curves.json. These values do not establish convergence "
              "or formal TEST improvement.", "",
              "| Variant | Epoch | Fit membership loss | VAL membership loss | VAL same-person split rate | VAL different-person false-merge rate | Finite gradients |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for v in ("B2", "B3", "B4"):
        history = curves["variants"][v]["selection"]
        if not history:
            continue
        last = history[-1]
        val = last["validation"]
        d = val["diagnostics"]
        split = 100 * d.get("same_person_split", 0) / max(1, d.get("same_person_cross_view_pairs", 0))
        merge = 100 * d.get("false_cross_view_merge", 0) / max(1, d.get("different_person_cross_view_pairs", 0))
        lines.append(f"| {v} | {last['epoch']} | {last['train']['losses']['per_camera_membership']:.4f} | {val['losses']['per_camera_membership']:.4f} | {split:.2f}% | {merge:.2f}% | {last['gradient_finite']} |")
    lines += ["", "Selection currently uses CPU, one Torch/BLAS thread per worker. "
              "Final B3/B4 training and complete TEST use truly idle V100 FP32 workers. "
              "B2 final training uses a third idle V100 if available, otherwise one CPU thread; its resumes keep that device family. "
              "Other users' processes are never stopped. Independent-scene concurrency is not single-stream FPS.", "",
              "A preregistered scheduling amendment allows already frozen B0/B1 GT-free reference inference "
              "during TRAIN selection. A second published scoring-order amendment permits their formal "
              "TrackEval/MATLAB scoring only after fresh full22 output-byte checks and verification that "
              "the original frozen R2 head completed full24 TRAIN. These are actual frozen-control TEST "
              "measurements; B2/B3/B4 require their own completed fresh full24 training. Their common "
              "TRAIN-only budget rule is unchanged. Canonical adoption later verifies all3 final loaders "
              "and identical prediction bytes; all110 posthoc and all5-before-real-LIVE barriers remain.", "",
              "Official MC-MOT uses original legal14 full TEST videos, after full22 causal inference "
              "(29,019 paired times / 548,583 shared candidates). Invalid8 remain separate. "
              "The following historical reference numbers are not new V5 measurements.", "",
              "| Method | HOTA | DetA | AssA | IDF1 | CVIDF1 | CVMA | MOTA | IDSW | Frag | TP | FP | FN |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    if native_smoke:
        lines[-2:-2] = ["Native scorer engineering smoke: three generated synthetic cases and six native MATLAB jobs passed. "
                        "No real TRAIN/TEST data or model was used. The native toolkit hashes exactly match archived V4 C1. "
                        "This is evaluator validation, not a V5 tracking score.", ""]
    for row in metric_rows:
        values = [str(row[k]) if k in ("IDSW", "Frag", "CLR_TP", "CLR_FP", "CLR_FN") else f"{row[k]:.4f}" for k in METRICS] if "HOTA" in row else ["pending"] * len(METRICS)
        lines.append("| " + row["method"] + " | " + " | ".join(values) + " |")
    if "B1" in official:
        control = official.get("B0", baselines["V4 C1"])
        diff = {k: official["B1"][k] - control[k] for k in METRICS}
        lines += ["", "The frozen B1 control keeps the old R2 identity head and changes the independent "
                  "assignment/birth/lifecycle/write policies; it is not the trained B3 main model. "
                  "B1 minus " + ("newly scored B0" if "B0" in official else "archived V4 C1") + ": "
                  f"HOTA {diff['HOTA']:+.4f}, AssA {diff['AssA']:+.4f}, IDF1 {diff['IDF1']:+.4f}, "
                  f"CVIDF1 {diff['CVIDF1']:+.4f} percentage points; IDSW {diff['IDSW']:+d}. "
                  "The aggregate does not support a standalone improvement from the frozen-head policy "
                  "replacement. Registered B2/B3/B4 training continues unchanged. "
                  "See V5_FROZEN_REFERENCE_NATIVE_RESULTS.md for per-scene native results and limits.", ""]
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
    lines += [f"Independent frozen C1/B1 TRAIN closed-loop audit: {policy['scenes_complete']}/6 complete scenes, "
              f"{policy['timestamps']} paired times. Each method advances its own predicted history with no optimizer. "
              "The frozen head was trained on these scenes, so these are mechanistic TRAIN diagnostics, not held-out TEST scores. "
              "This audit cannot alter the registered membership model or its training budget.", ""]
    if policy["paired_B1_minus_C1"]:
        c = policy["paired_B1_minus_C1"]
        lines += [f"Frozen online B1 minus C1: same-person pair corrections {c.get('same_person_shared_ID/corrections', 0)}, "
                  f"regressions {c.get('same_person_shared_ID/regressions', 0)}, net {c.get('same_person_shared_ID/net', 0):+d}; "
                  f"different-person separation net {c.get('different_person_separate_ID/net', 0):+d}. "
                  f"Clean global one-to-one overlap net {policy['global_one_to_one_paired'].get('net', 0):+d} observations, not CVIDF1.", ""]
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
              "Following the NAS disconnect, B0 scenes1–20 retain their original NAS-write timing; "
              "B0 scenes21–22 and all other new outputs write local durable storage before verified NAS archival. "
              "Cached wall FPS therefore cannot rank algorithm speed under a common storage condition. "
              "The registered real RGB timing uses the same local output storage for all five methods.", "",
              "| Method | Cached worker step FPS | Cached worker wall FPS | Real RGB-to-ID FPS | Real scope |",
              "|---|---:|---:|---:|---|"]
    for row in runtime_rows:
        values = [f"{row[k]:.4f}" if row[k] is not None else "not measured" for k in ("CACHED_FPS", "CACHED_worker_wall_FPS", "REAL_RGB_to_ID_FPS")]
        lines.append("| " + row["method"] + " | " + " | ".join(values) + " | " + row["real_scope"] + " |")
    lines += ["", "Native tool dependencies and wrapper comparability are retained from the archived V4 report. "
              "Historical GMT/V2 exports used two-decimal coordinates; current V3/V4/V5 exports preserve .17g. "
              "Original GT and shared native tracking dependencies were verified in the archived comparison. "
              "This is a documented system-reference limitation, not permission to alter GT or re-tune TEST.", ""]
    if (cuda_head_check and cuda_head_check.get("status") == "PASSED_GENERATED_FP32_CPU_AND_V100_FORWARD_BACKWARD"
            and cuda_head_check.get("v5_source_sha256") == state["v5_source_sha256"]):
        lines += ["Generated-input V100 readiness: eight FP32 CPU/CUDA forward/backward checks passed "
                  "for the unchanged dedicated Membership Head, including empty history/observations and "
                  "192-history/144-observation inputs. CPU and CUDA initialization hashes match. "
                  "No real dataset, trained checkpoint or optimizer was used. This does not establish "
                  "convergence, full training VRAM or actual video FPS. "
                  "See V5_GENERATED_CUDA_HEAD_PREFLIGHT.md and v5_cuda_head_preflight.json.", ""]
    lines += ["Identity event interpretation: " + identity_summary["status"] +
              ". See V5_FULL_TEST_IDENTITY_ANALYSIS.md and v5_identity_event_rates.csv, "
              "v5_identity_event_contrasts.csv, v5_identity_video_quarters.csv. "
              "Clean-join diagnostics remain separate from native HOTA, IDSW and CVIDF1.", ""]
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
