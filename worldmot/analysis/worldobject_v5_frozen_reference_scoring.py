"""Explicit early native scoring for the two already eligible frozen controls.

This changes scoring order only. It cannot train, infer a new model, select a
checkpoint or admit B2/B3/B4. Their final-all24 gates remain unchanged.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time

from worldmot.data.worldobject_cache import sha256
from worldmot.runtime.experiment_authorization import APPROVAL_SCHEMA, source_commit, source_fingerprint
from worldmot.runtime.world_object_v5_contract import (
    atomic_json, require_v5_scope, v5_source_fingerprint, FULL_SHA, LEGAL_TEST)
from worldmot.runtime.world_object_v4_test_contract import install_no_gt_guard
from worldobject_v5_reference_scope import test_runner_completed

ANNOTATIONS = "7a1e735bf25026a56148deaeb432e0520c6ad5dad97d5362aa58739255389593"
NAMES = ("B0", "B1")


def read(path):
    return json.loads(Path(path).read_text())


def paths(root):
    private = Path(root) / "provenance"
    return private / "scope_v5_frozen_reference_scoring.json", private / "authorization_v5_frozen_reference_scoring.json"


def require_frozen_scoring(root, variant):
    if variant not in NAMES:
        raise PermissionError("ONLY_ALREADY_TRAINED_FROZEN_B0_B1_CAN_BE_SCORED_EARLY")
    scope_path, auth_path = paths(root)
    scope, _ = require_v5_scope(scope_path, auth_path, "evaluate", "test", variant)
    auth = read(auth_path)
    if (scope.get("frozen_reference_scoring_only") is not True or scope["variants"] != list(NAMES)
            or scope["frozen_C1_checkpoint_sha256"] != FULL_SHA
            or scope["R2_full24_eligibility"]["checkpoint_sha256"] != FULL_SHA
            or set(auth["operations"]) != {"evaluate"} or auth["splits"] != ["test"]
            or set(auth["method_ids"]) != {"V5_B0", "V5_B1"}
            or scope.get("trained_B2_B3_B4_eligible") is not False):
        raise PermissionError("FROZEN_REFERENCE_SCORING_SCOPE_MISMATCH")
    for name, digest in scope["unchanged_original_locks"].items():
        if sha256(name) != digest:
            raise ValueError("ORIGINAL_TRAIN_OR_REFERENCE_LOCK_CHANGED")
    reference = read(Path(root) / "provenance/scope_v5_reference_test.json")
    same_reference_inputs(scope, reference)
    return scope


def same_reference_inputs(scope, reference):
    fields = ("base_commit", "source_sha256", "v5_source_sha256", "registered_config_sha256",
              "calibration_sha256", "frozen_C1_checkpoint_sha256", "human_instruction_sha256")
    if (any(scope[k] != reference[k] for k in fields)
            or scope["splits"]["test"] != reference["splits"]["test"]
            or scope["variant_configs"]["B1"] != reference["variant_configs"]["B1"]):
        raise ValueError("SCORING_MUST_USE_EXACT_ALREADY_FROZEN_REFERENCE_INPUTS_AND_CONFIG")


def require_publication(root, runtime, variant):
    scope = require_frozen_scoring(root, variant)
    registration = Path(root) / "provenance/frozen_reference_scoring_registration.json"
    publication = Path(runtime) / "frozen_reference_scoring_publication.json"
    if not publication.is_file():
        raise PermissionError("PUBLISH_SCORING_ORDER_AMENDMENT_BEFORE_READING_NEW_TEST_GT")
    proof = read(publication)
    if (proof.get("status") != "GITHUB_REF_TREE_AND_LOCAL_FETCH_VERIFIED"
            or proof.get("registration_sha256") != sha256(registration)
            or proof.get("scoring_driver_sha256") != sha256(__file__)
            or proof.get("v5_source_sha256") != v5_source_fingerprint()):
        raise PermissionError("PUBLISH_SCORING_ORDER_AMENDMENT_BEFORE_READING_NEW_TEST_GT")
    frozen = read(registration)
    if frozen["scope_sha256"] != sha256(paths(root)[0]) or frozen["authorization_sha256"] != sha256(paths(root)[1]):
        raise PermissionError("PUBLISHED_FROZEN_SCORING_SCOPE_OR_AUTHORIZATION_CHANGED")
    return scope


def register(root, request, v4_scope):
    root, request, v4_scope = map(Path, (root, request, v4_scope))
    private = root / "provenance"
    score_path, auth_path = paths(root)
    if score_path.exists() or auth_path.exists():
        raise FileExistsError("Do not overwrite the separate scoring registration")
    original_scope = private / "scope_v5_reference_test.json"
    original_auth = private / "authorization_v5_reference_test.json"
    original, _ = require_v5_scope(original_scope, original_auth, "preflight", "test", "B0")
    if original["human_instruction_sha256"] != sha256(request):
        raise ValueError("CURRENT_HUMAN_V5_TASK_MUST_AUTHORIZE_THE_REGISTERED_METHODS")
    coverage = read(private / "reference_test_coverage.json")
    if coverage["status"] != "COMPLETE_ALL22_B0_B1_GT_FREE_INFERENCE":
        raise ValueError("BOTH_FROZEN_METHODS_MUST_HAVE_COMPLETE_INFERENCE")
    guard = install_no_gt_guard()
    from worldmot.models.world_object_v4_checkpoint import load_frozen_r2_head
    raw = read(Path(original["splits"]["train"]["raw_cache"]) / "DETECTOR_LOCK.json")
    gate = Path(original["splits"]["train"]["support_cache"]).parent / "train/FULL_WORLD_OBJECT_V3/train_gate.json"
    model, eligibility = load_frozen_r2_head(gate, raw, "cpu")
    del model
    if any(len(coverage["scenes"][v]) != 22
           or sum(x["timestamps"] for x in coverage["scenes"][v].values()) != 29019
           or sum(x["candidates"] for x in coverage["scenes"][v].values()) != 548583 for v in NAMES):
        raise ValueError("FULL22_REFERENCE_INPUTS_REQUIRED")
    original_paths = [private / "scope_v5_train.json", private / "authorization_v5_train.json",
                      original_scope, original_auth, v4_scope]
    preserved = {str(p): sha256(p) for p in original_paths}
    scope = deepcopy(original)
    scope.update(source_commit=source_commit(), frozen_reference_scoring_only=True,
        scoring_before_all3_full24_gates=True, trained_B2_B3_B4_eligible=False,
        R2_full24_eligibility=eligibility, R2_full24_gate_path=str(gate),
        R2_full24_gate_sha256=sha256(gate), original_inference_scope_sha256=sha256(original_scope),
        original_inference_authorization_sha256=sha256(original_auth), unchanged_original_locks=preserved,
        original_TEST_annotations_sha256=ANNOTATIONS, reference_coverage_sha256=sha256(private / "reference_test_coverage.json"),
        scheduling_amendment="Evaluate already trained frozen B1/B0 after their own full22 output-byte checks; new-model gates and TRAIN-only budget rule unchanged",
        score_directory="evaluation_frozen_references", posthoc_all110_barrier_unchanged=True)
    atomic_json(score_path, scope)
    auth = {"schema": APPROVAL_SCHEMA, "approved": True, "source_commit": source_commit(),
        "source_sha256": source_fingerprint(), "v5_source_sha256": v5_source_fingerprint(),
        "user_authorization_reference": str(request) + " SHA256 " + sha256(request),
        "scope_lock_sha256": sha256(score_path), "operations": ["evaluate"], "splits": ["test"],
        "method_ids": ["V5_B0", "V5_B1"], "TEST_parameter_selection": False,
        "new_model_training_or_inference_or_live_or_diagnostics_authorized": False,
        "original_GT_modified": False}
    atomic_json(auth_path, auth)
    if guard["blocked_GT_reads"] or any(sha256(p) != digest for p, digest in preserved.items()):
        raise ValueError("GT_READ_ATTEMPT_OR_OLD_LOCK_MODIFIED")
    proof = {"status": "FROZEN_REFERENCE_NATIVE_SCORING_ORDER_REGISTERED_BEFORE_NEW_TEST_SCORES",
        "UTC": datetime.now(timezone.utc).isoformat(), "scope_sha256": sha256(score_path),
        "authorization_sha256": sha256(auth_path), "original_inference_scope_sha256": sha256(original_scope),
        "original_inference_authorization_sha256": sha256(original_auth), "unchanged_locks": preserved,
        "actual_R2_full24_loader_proof": eligibility, "source_sha256": v5_source_fingerprint(),
        "scoring_driver_sha256": sha256(__file__), "TEST_GT_read": False,
        "method_parameters_seeds_loss_initialization_or_budget_rules_changed": False,
        "B2_B3_B4_gates_relaxed": False, "formal_protocol": "unchanged full22 inference/original legal14 TrackEval and MATLAB",
        "previous_all3_before_any_score_order_superseded_only_for_B0_B1": True}
    atomic_json(private / "frozen_reference_scoring_registration.json", proof)
    return proof


def validate_inputs(root, runtime, variant):
    scope = require_publication(root, runtime, variant)
    guard = install_no_gt_guard()
    original = read(Path(root) / "provenance/scope_v5_reference_test.json")
    verify = test_runner_completed()
    scenes = {}
    for scene in range(1, 23):
        folder = Path(root) / f"predictions/{variant}/scene_{scene:02d}"
        while True:
            try:
                manifest = verify(folder, original, variant, scene)
                break
            except OSError as error:
                if error.errno not in {5, 103, 104, 107, 110, 116}:
                    raise
                atomic_json(Path(runtime) / f"frozen_scoring_{variant}_validation_progress.json", {
                    "status": "WAITING_SAME_INPUT_NAS_READ", "scene": scene, "errno": error.errno,
                    "model_jobs_restarted": False})
                time.sleep(30.)
        if (manifest["scope_sha256"] != scope["original_inference_scope_sha256"]
                or manifest["approval"]["approval_sha256"] != scope["original_inference_authorization_sha256"]
                or manifest["GT_in_inference"] or not manifest["causal"]
                or (variant == "B0" and manifest["exact_original_C1_timestamps"] != manifest["timestamps"])):
            raise ValueError("FROZEN_REFERENCE_MANIFEST_OR_AUTHORIZATION_MISMATCH")
        scenes[str(scene)] = {"manifest_sha256": sha256(folder / "manifest.json"),
            "timestamps": manifest["timestamps"], "candidates": manifest["candidate_count"],
            "output_sha256": manifest["output_sha256"]}
        atomic_json(Path(runtime) / f"frozen_scoring_{variant}_validation_progress.json", {
            "status": "NO_GT_FULL_OUTPUT_BYTE_VERIFICATION", "complete_scenes": scene, "total_scenes": 22})
    if guard["blocked_GT_reads"] or sum(x["timestamps"] for x in scenes.values()) != 29019:
        raise ValueError("NO_GT_COMPLETE22_BARRIER_FAILED")
    proof = {"status": "ALL22_FROZEN_REFERENCE_OUTPUT_BYTES_VERIFIED_BEFORE_GT", "variant": variant,
        "scope_sha256": sha256(paths(root)[0]), "scenes": scenes, "GT_read": False,
        "source_sha256": v5_source_fingerprint(), "UTC": datetime.now(timezone.utc).isoformat()}
    path = Path(root) / f"provenance/frozen_reference_score_inputs/{variant}.json"
    atomic_json(path, proof)
    return path


def score(root, runtime, variant, audit):
    scope = require_publication(root, runtime, variant)
    verified = Path(root) / f"provenance/frozen_reference_score_inputs/{variant}.json"
    proof = read(verified)
    if (proof["status"] != "ALL22_FROZEN_REFERENCE_OUTPUT_BYTES_VERIFIED_BEFORE_GT"
            or proof["variant"] != variant or proof["GT_read"] is not False
            or proof["source_sha256"] != scope["v5_source_sha256"]
            or proof["scope_sha256"] != sha256(paths(root)[0]) or set(proof["scenes"]) != {str(s) for s in range(1, 23)}
            or sum(x["timestamps"] for x in proof["scenes"].values()) != 29019
            or sum(x["candidates"] for x in proof["scenes"].values()) != 548583):
        raise PermissionError("FULL22_FRESH_OUTPUT_CHECKS_REQUIRED_BEFORE_GT")
    for scene, row in proof["scenes"].items():
        if sha256(Path(root) / f"predictions/{variant}/scene_{int(scene):02d}/manifest.json") != row["manifest_sha256"]:
            raise ValueError("INPUT_MANIFEST_CHANGED_AFTER_VERIFICATION")
    annotations = Path(scope["dataset_root"]) / "annotations/test.json"
    if sha256(annotations) != ANNOTATIONS:
        raise ValueError("ORIGINAL_TEST_GT_CHANGED")
    output = Path(root) / "evaluation_frozen_references" / variant
    if output.exists():
        raise FileExistsError("Preserve any earlier native scoring attempt")
    entry = Path(__file__).resolve().parents[1] / "scripts/evaluate_fullvideo.py"
    sys.argv = [str(entry), "--dataset-root", scope["dataset_root"], "--split", "test", "--audit", str(audit),
        "--predictions", str(Path(root) / "predictions" / variant), "--output", str(output),
        "--authorization", str(paths(root)[1]), "--method-id", "V5_" + variant]
    runpy.run_path(str(entry), run_name="__main__")
    metric = read(output / "metrics.json")
    validate_metric(metric, proof, read(audit))
    if sha256(annotations) != ANNOTATIONS:
        raise ValueError("ORIGINAL_TEST_GT_CHANGED_DURING_NATIVE_SCORING")
    if v5_source_fingerprint() != scope["v5_source_sha256"]:
        raise ValueError("FROZEN_MODEL_SOURCE_CHANGED_DURING_NATIVE_SCORING")
    receipt = {"status": "COMPLETE_FROZEN_REFERENCE_FULL22_NATIVE14", "variant": variant,
        "scope_sha256": sha256(paths(root)[0]), "authorization_sha256": sha256(paths(root)[1]),
        "input_byte_verification_sha256": sha256(verified), "metrics_sha256": sha256(output / "metrics.json"),
        "original_annotations_sha256": ANNOTATIONS, "checkpoint_sha256": FULL_SHA,
        "TRAIN_parameters_or_budget_rules_changed": False, "original_GT_modified": False,
        "UTC": datetime.now(timezone.utc).isoformat()}
    atomic_json(output / "frozen_scoring_receipt.json", receipt)
    return receipt


def validate_metric(metric, inputs, audit):
    names = {x["name"] for x in audit["test"]["scenes"] if x["scene"] in LEGAL_TEST}
    if (metric["status"] != "COMPLETE_VALID_SCOPE" or set(metric["evaluated_scenes"]) != names
            or metric["raw_GT_modified"] or sorted(x["scene"] for x in metric["coverage"]) != list(range(1, 23))
            or sum(x["frames"] for x in metric["coverage"]) != 29019
            or sorted(x["scene"] for x in metric["blocked"]) != sorted(set(range(1, 23)) - set(LEGAL_TEST))
            or any(x["reason"] != "RAW_GT_DUPLICATE_OR_VERSION_INVALID" for x in metric["blocked"])):
        raise ValueError("EXACT_FULL22_INFERENCE_AND_ORIGINAL_LEGAL14_NATIVE_SCOPE_REQUIRED")
    if any(row["predictions_sha256"] != inputs["scenes"][str(row["scene"])]["output_sha256"]["tracks.jsonl"]
           for row in metric["coverage"]):
        raise ValueError("NATIVE_METRICS_USED_DIFFERENT_PREDICTION_BYTES")


def adopt_scores(root, repo):
    """Reuse native results only after the original canonical all3 gates pass."""
    from worldobject_v5_reference_scope import (equivalent_reference_inputs, verify_canonical_final_models,
                                                require_canonical_or_adopted)
    from worldmot.evaluation.official_fullvideo import native_tracking_sources
    root, repo = Path(root), Path(repo)
    canonical_path = root / "provenance/scope_v5_test.json"
    canonical, _ = require_v5_scope(canonical_path, root / "provenance/authorization_v5_test.json", "evaluate", "test", "B0")
    old_inference = read(root / "provenance/scope_v5_reference_test.json")
    equivalent_reference_inputs(old_inference, canonical)
    actual_models = verify_canonical_final_models(root, canonical)
    scoring = require_frozen_scoring(root, "B0")
    same_reference_inputs(canonical, scoring)
    audit_path = Path(canonical["splits"]["test"]["support_cache"]).parents[1] / "fulltrain_20261022/audit/data_audit.json"
    audit = read(audit_path)
    kit = repo / "MOTChallengeEvalKit_cv_test/matlab_devkit"
    native_hashes = {str(p.relative_to(kit)): sha256(p) for p in native_tracking_sources(kit)}
    trackeval = repo / "TrackEval"
    trackeval_hashes = {str(p.relative_to(trackeval)): sha256(p) for p in sorted((trackeval / "trackeval").rglob("*.py"))}
    receipts = {}
    for variant in NAMES:
        folder = root / "evaluation_frozen_references" / variant
        record_path = folder / "frozen_scoring_receipt.json"
        if not record_path.is_file():
            continue
        record, metric = read(record_path), read(folder / "metrics.json")
        input_path = root / f"provenance/frozen_reference_score_inputs/{variant}.json"
        inputs = read(input_path)
        if (record["status"] != "COMPLETE_FROZEN_REFERENCE_FULL22_NATIVE14"
                or record["scope_sha256"] != sha256(paths(root)[0])
                or record["input_byte_verification_sha256"] != sha256(input_path)
                or record["metrics_sha256"] != sha256(folder / "metrics.json")
                or metric["TrackEval"]["tool_sha256"] != trackeval_hashes
                or metric["VisionTrack_MATLAB"]["toolkit_sha256"] != native_hashes
                or metric["VisionTrack_MATLAB"]["adapter_sha256"] != sha256(repo / "worldmot/scripts/official_worldmot_metrics.m")):
            raise ValueError("EARLY_NATIVE_SCORE_OR_PROTOCOL_CHANGED")
        validate_metric(metric, inputs, audit)
        for scene, source in inputs["scenes"].items():
            manifest_path = root / f"predictions/{variant}/scene_{int(scene):02d}/manifest.json"
            manifest = read(manifest_path)
            require_canonical_or_adopted(root, canonical_path, variant, int(scene), manifest)
            if sha256(manifest_path) != source["manifest_sha256"] or manifest["output_sha256"] != source["output_sha256"]:
                raise ValueError("CANONICAL_PREDICTIONS_DIFFER_FROM_EARLY_NATIVE_INPUTS")
        target = root / "evaluation" / variant
        target.parent.mkdir(exist_ok=True)
        if target.exists() or target.is_symlink():
            if target.resolve() != folder.resolve():
                raise FileExistsError("Never replace an independent canonical score")
        else:
            target.symlink_to(folder, target_is_directory=True)
        receipts[variant] = {"metrics_sha256": record["metrics_sha256"], "early_score_receipt_sha256": sha256(record_path),
            "canonical_scope_sha256": sha256(canonical_path), "prediction_or_metric_bytes_modified": False,
            "strict_final_models_loaded_before_adoption": True}
    proof = {"status": "COMPLETE_FROZEN_REFERENCE_NATIVE_SCORE_ADOPTION", "variants": sorted(receipts),
        "canonical_scope_sha256": sha256(canonical_path), "all3_actual_final_model_loader_proofs": actual_models,
        "per_method": receipts, "original_GT_modified": False, "formal_metrics_recomputed": False}
    atomic_json(root / "provenance/frozen_reference_native_score_adoption.json", proof)
    return proof


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=("register", "validate", "score", "adopt"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--variant", choices=NAMES)
    parser.add_argument("--request-file", type=Path)
    parser.add_argument("--v4-scope", type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--repo", type=Path)
    args = parser.parse_args()
    if args.operation == "register":
        result = register(args.output_root, args.request_file, args.v4_scope)
    elif args.operation == "validate":
        result = str(validate_inputs(args.output_root, args.runtime, args.variant))
    elif args.operation == "score":
        result = score(args.output_root, args.runtime, args.variant, args.audit)
    else:
        result = adopt_scores(args.output_root, args.repo)
    print(json.dumps(result), flush=True)
