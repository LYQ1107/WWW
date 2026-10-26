"""Explicit inference-only scopes for already frozen B0/B1 references.

The trained B2/B3/B4 all24 gates remain mandatory in the canonical registrar.
Reference outputs can only be reused through byte-verified scope-equivalence
receipts; an earlier authorization is never silently treated as the final one.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from worldmot.runtime.world_object_v5_contract import (atomic_json, require_v5_scope,
    v5_source_fingerprint, FULL_SHA)
from worldmot.runtime.experiment_authorization import APPROVAL_SCHEMA, source_commit
from worldmot.runtime.world_object_v4_test_contract import install_no_gt_guard
from worldmot.data.worldobject_cache import sha256


def read(path):
    return json.loads(Path(path).read_text())


def test_runner_completed():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from run_worldobject_v5_test import completed
    return completed


def equivalent_reference_inputs(reference, canonical):
    if reference.get("reference_only") is not True or reference.get("variants") != ["B0", "B1"]:
        raise ValueError("ONLY_FROZEN_B0_B1_REFERENCE_SCOPE_CAN_BE_ADOPTED")
    if set(canonical.get("final_weights", {})) != {"B2", "B3", "B4"}:
        raise ValueError("CANONICAL_ALL3_FULL24_GATES_REQUIRED_BEFORE_ADOPTION_OR_SCORING")
    if any(s.get("TEST_parameter_selection") is not False or s.get("GT_in_model_inputs") is not False
           for s in (reference, canonical)):
        raise ValueError("NO_TEST_SELECTION_OR_GT_MODEL_INPUTS_ALLOWED")
    fields = ["base_commit", "source_sha256", "v5_source_sha256", "registered_config_sha256",
              "calibration_sha256", "frozen_C1_checkpoint_sha256", "original_C1_reference_predictions",
              "dataset_root", "human_instruction_sha256", "input_mirror_receipt_sha256"]
    if any(reference.get(k) != canonical.get(k) for k in fields):
        raise ValueError("REFERENCE_CODE_CONFIG_OR_WEIGHTS_DIFFER_FROM_CANONICAL_SCOPE")
    if reference["splits"]["test"] != canonical["splits"]["test"]:
        raise ValueError("REFERENCE_TEST_INPUTS_DIFFER_FROM_CANONICAL_SCOPE")
    if reference["variant_configs"]["B1"] != canonical["variant_configs"]["B1"]:
        raise ValueError("REFERENCE_B1_CONFIG_CHANGED")
    return fields + ["splits/test", "variant_configs/B1"]


def adoption_path(root, variant, scene):
    return Path(root) / f"provenance/reference_test_adoption/{variant}_scene{scene:02d}.json"


def adopt_reference_scene(root, canonical_path, variant, scene):
    root, canonical_path = Path(root), Path(canonical_path)
    private = root / "provenance"
    reference_path = private / "scope_v5_reference_test.json"
    authorization = private / "authorization_v5_reference_test.json"
    reference, _ = require_v5_scope(reference_path, authorization, "replay", "test", variant)
    canonical = read(canonical_path)
    fields = equivalent_reference_inputs(reference, canonical)
    folder = root / f"predictions/{variant}/scene_{scene:02d}"
    manifest = test_runner_completed()(folder, reference, variant, scene)
    if (variant not in {"B0", "B1"} or manifest["scope_sha256"] != sha256(reference_path)
            or manifest["approval"]["approval_sha256"] != sha256(authorization)
            or not manifest["causal"] or manifest["GT_in_inference"]
            or manifest["checkpoint_sha256"] != FULL_SHA):
        raise ValueError("REFERENCE_ORIGINAL_AUTHORIZATION_OR_MANIFEST_MISMATCH")
    if variant == "B0" and manifest["exact_original_C1_timestamps"] != manifest["timestamps"]:
        raise ValueError("ALL_REFERENCE_B0_TIMESTAMPS_MUST_EXACTLY_REPRODUCE_C1")
    receipt = {"schema": "v5_reference_scope_adoption_v1", "variant": variant, "scene": scene,
               "reference_scope_sha256": sha256(reference_path), "reference_authorization_sha256": sha256(authorization),
               "canonical_scope_sha256": sha256(canonical_path), "original_manifest_sha256": sha256(folder / "manifest.json"),
               "output_sha256": manifest["output_sha256"], "equivalent_fields": fields,
               "weights_or_predictions_modified": False, "training_gates_skipped": False,
               "reference_scope_allowed_TEST_scoring": False}
    path = adoption_path(root, variant, scene)
    if path.exists():
        if read(path) != receipt:
            raise ValueError("IMMUTABLE_REFERENCE_ADOPTION_RECEIPT_CHANGED")
    else:
        atomic_json(path, receipt)
    return receipt


def require_canonical_or_adopted(root, canonical_path, variant, scene, manifest):
    canonical_sha = sha256(canonical_path)
    if manifest["scope_sha256"] == canonical_sha:
        return
    path = adoption_path(root, variant, scene)
    if not path.exists():
        raise ValueError("EXPLICIT_REFERENCE_SCOPE_ADOPTION_REQUIRED")
    receipt = read(path)
    folder = Path(root) / f"predictions/{variant}/scene_{scene:02d}"
    if (variant not in {"B0", "B1"} or receipt["canonical_scope_sha256"] != canonical_sha
            or receipt["reference_scope_sha256"] != manifest["scope_sha256"]
            or receipt["original_manifest_sha256"] != sha256(folder / "manifest.json")
            or receipt["output_sha256"] != manifest["output_sha256"]
            or receipt["weights_or_predictions_modified"] or receipt["training_gates_skipped"]):
        raise ValueError("REFERENCE_ADOPTION_EVIDENCE_MISMATCH")


def register(root, mirror, request, v4_scope_path):
    root, mirror, request, v4_scope_path = map(Path, (root, mirror, request, v4_scope_path))
    private = root / "provenance"
    if (private / "scope_v5_test.json").exists() or (root / "predictions").exists():
        raise ValueError("REGISTER_REFERENCE_SCHEDULING_BEFORE_ANY_NEW_TEST_PREDICTIONS")
    refpath, authpath = private / "scope_v5_reference_test.json", private / "authorization_v5_reference_test.json"
    if refpath.exists() or authpath.exists():
        raise FileExistsError("Do not overwrite reference scopes")
    original_train = private / "scope_v5_train.json"
    original_train_sha, original_v4_sha = sha256(original_train), sha256(v4_scope_path)
    base = read(original_train)
    if base["human_instruction_sha256"] != sha256(request) or base["v5_source_sha256"] != v5_source_fingerprint():
        raise ValueError("FROZEN_HUMAN_INSTRUCTION_AND_MODEL_SOURCE_REQUIRED")
    old = read(v4_scope_path)
    guard = install_no_gt_guard()
    support = mirror / "worldobject_v3_birthfix_r2_20261025/cache_test_v2"
    raw = mirror / "gmt_detector_comparison_20261023/raw_test_v1"
    scenes = {}
    for sid in range(1, 23):
        sp, rp = support / f"scene_{sid:02d}/manifest.json", raw / f"scene_{sid:02d}/manifest.json"
        sm, rm = read(sp), read(rp)
        if (sm["status"] != "COMPLETE" or rm["status"] != "COMPLETE" or sm["timestamps"] != rm["frames"]
                or sm["counts"]["observations"] != rm["observations"]
                or sha256(sp) != old["scenes"][str(sid)]["cache_manifest_sha256"]
                or sha256(rp) != old["scenes"][str(sid)]["raw_manifest_sha256"]):
            raise ValueError("ORIGINAL_FROZEN_TEST_CACHE_PROVENANCE_REQUIRED")
        scenes[str(sid)] = {"support_manifest_sha256": sha256(sp), "raw_manifest_sha256": sha256(rp),
                            "timestamps": sm["timestamps"], "candidates": sm["counts"]["observations"]}
    if sum(x["timestamps"] for x in scenes.values()) != 29019 or sum(x["candidates"] for x in scenes.values()) != 548583:
        raise ValueError("FULL22_REFERENCE_SCOPE_REQUIRED")
    scope = deepcopy(base)
    scope.update(source_commit=source_commit(), variants=["B0", "B1"], reference_only=True,
                 original_C1_reference_predictions="/data1/liuyeqiang/worldmot_v4_c1_fulltest_outputs_20261026/predictions/V4_C1",
                 scoring_before_all3_full24_gates=False, current_models_or_budget_changed=False,
                 scheduling_amendment="GT-free frozen reference inference only; all scores deferred until canonical all3 gates")
    scope["splits"]["test"] = {"support_cache": str(support), "raw_cache": str(raw),
        "raw_lock_sha256": sha256(raw / "DETECTOR_LOCK.json"), "scenes": scenes, "timestamps": 29019, "candidates": 548583}
    atomic_json(refpath, scope)
    auth = {"schema": APPROVAL_SCHEMA, "approved": True, "source_commit": source_commit(),
            "source_sha256": scope["source_sha256"], "v5_source_sha256": v5_source_fingerprint(),
            "user_authorization_reference": str(request) + " SHA256 " + sha256(request),
            "scope_lock_sha256": sha256(refpath), "operations": ["replay", "preflight"],
            "splits": ["test"], "method_ids": ["V5_B0", "V5_B1"],
            "TEST_parameter_selection": False, "original_GT_modified": False,
            "evaluate_diagnostics_train_live_NOT_authorized_by_this_subset": True}
    atomic_json(authpath, auth)
    if (sha256(original_train) != original_train_sha or sha256(v4_scope_path) != original_v4_sha
            or guard["blocked_GT_reads"]):
        raise ValueError("ORIGINAL_LOCK_CHANGED_OR_GT_READ")
    proof = {"status": "FROZEN_B0_B1_INFERENCE_ONLY_SCOPE_REGISTERED", "reference_scope_sha256": sha256(refpath),
             "authorization_sha256": sha256(authpath), "unchanged_v5_train_scope_sha256": original_train_sha,
             "unchanged_v4_scope_sha256": original_v4_sha, "frozen_head_sha256": FULL_SHA,
             "inference_source_sha256": v5_source_fingerprint(), "registrar_sha256": sha256(__file__),
             "registered_UTC": datetime.now(timezone.utc).isoformat(), "prior_V5_TEST_predictions": 0,
             "no_original_GT_read": True, "scoring_allowed": False, "new_trained_models_eligible": False}
    atomic_json(private / "reference_test_registration.json", proof)
    return proof


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--operation", choices=("register", "adopt"), default="register")
    p.add_argument("--output-root", type=Path, required=True)
    for key in ("mirror-root", "request-file", "v4-scope"):
        p.add_argument("--" + key, type=Path)
    a = p.parse_args()
    if a.operation == "register":
        if not all((a.mirror_root, a.request_file, a.v4_scope)):
            p.error("register requires mirror-root, request-file and v4-scope")
        print(json.dumps(register(a.output_root, a.mirror_root, a.request_file, a.v4_scope)), flush=True)
    else:
        canonical = a.output_root / "provenance/scope_v5_test.json"
        authorization = a.output_root / "provenance/authorization_v5_test.json"
        require_v5_scope(canonical, authorization, "replay", "test", "B0")
        coverage = read(a.output_root / "provenance/reference_test_coverage.json")
        if coverage["status"] != "COMPLETE_ALL22_B0_B1_GT_FREE_INFERENCE":
            raise ValueError("COMPLETE_REFERENCE_INFERENCE_REQUIRED_BEFORE_BATCH_ADOPTION")
        receipts = {}
        for variant in ("B0", "B1"):
            for scene in range(1, 23):
                receipt = adopt_reference_scene(a.output_root, canonical, variant, scene)
                receipts[f"{variant}/scene{scene:02d}"] = sha256(adoption_path(a.output_root, variant, scene))
                print(json.dumps({"adopted": variant, "scene": scene}), flush=True)
        atomic_json(a.output_root / "provenance/reference_test_adoption_complete.json", {
            "status": "COMPLETE_BYTE_VERIFIED_REFERENCE_ADOPTION", "canonical_scope_sha256": sha256(canonical),
            "per_scene_receipts": receipts, "original_weights_or_predictions_modified": False})
