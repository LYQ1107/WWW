"""Verify one completed V5 TRAIN checkpoint, without inference or optimization."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worldmot.runtime.world_object_v5_contract import (
    load_v5_membership_gate, registration_path, state_tensor_sha256, v5_source_fingerprint,
)
from worldmot.runtime.experiment_authorization import source_fingerprint
from worldmot.tracking.world_object_v5_config import registered_config
from worldmot.training.world_object_membership_loss import LOSS_WEIGHTS


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def tensors_sha(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        h.update(name.encode() + b"\0")
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    for name in ("repo", "work", "runtime", "worker-log", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--variant", choices=["B2", "B3", "B4"], required=True)
    p.add_argument("--expected-checkpoint-sha", required=True)
    p.add_argument("--expected-gate-sha", required=True)
    p.add_argument("--worker-pid", type=int, required=True)
    p.add_argument("--worker-start-tick", required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("PRESERVE_EXISTING_FINAL_TRAIN_VERIFICATION")
    helper_sha = sha(Path(__file__))
    gp = args.work / "train" / args.variant / "train_gate.json"
    if sha(gp) != args.expected_gate_sha:
        raise ValueError("EXPECTED_TRAIN_GATE_SHA_MISMATCH")
    gate = json.loads(gp.read_text())
    if gate.get("checkpoint_sha256") != args.expected_checkpoint_sha:
        raise ValueError("EXPECTED_CHECKPOINT_SHA_MISMATCH")
    if gate.get("variant") != args.variant:
        raise ValueError("EXPECTED_VARIANT_MISMATCH")
    checkpoint = gp.parent / gate["checkpoint"]
    if checkpoint.parent != gp.parent or sha(checkpoint) != args.expected_checkpoint_sha:
        raise ValueError("ACTUAL_CHECKPOINT_SHA_MISMATCH")

    import numpy as np
    import torch
    torch.set_num_threads(1)
    source = "c4f9d1c0b2a290062aba52ae7c0d322176ae5376f9c85688f74d89159e16f733"
    expected_initial = "3fb3f3d6336e8aa32d635ecff3c05929b0d4b8f265f4a0ccf56771814d13ed23"
    calibration = args.work / "provenance/calibration_train_fit18.json"
    scope_path = args.work / "provenance/scope_v5_train.json"
    authorization = args.work / "provenance/authorization_v5_train.json"
    budget_path = args.work / "provenance/common_train_budget.json"
    if (v5_source_fingerprint() != source or gate["v5_source_sha256"] != source
            or gate["source_sha256"] != source_fingerprint()
            or sha(calibration) != gate["calibration_sha256"]
            or sha(scope_path) != gate["scope_sha256"]
            or sha(budget_path) != gate["common_budget_sha256"]
            or sha(registration_path()) != gate["registered_config_sha256"]):
        raise ValueError("FROZEN_SOURCE_OR_TRAIN_PROVENANCE_MISMATCH")
    expected_hashes = {
        "calibration": "97103af38a151ebff1baaecfe2f29ea56d021a78ea7cf85037b1eb2ded878e7a",
        "scope": "c841a987453adb9f49bfdb1722c161e085e349ced682b9e57b150b6e7b6b2707",
        "authorization": "26876952093fc04771c1c250e5611c78c61db822c87b39bd306e0a5bfb35ca90",
        "common_budget": "8c721f9d2e1dc98d02069649e1df26302718c67a3283c9f1998559c9fcddc50d",
        "registered_config": "07ce5975d379803213bfe6a22e67fc5018e9a2b44c483e103681796c38c2ba1f",
    }
    actual_hashes = {"calibration": sha(calibration), "scope": sha(scope_path),
                     "authorization": sha(authorization), "common_budget": sha(budget_path),
                     "registered_config": sha(registration_path())}
    if actual_hashes != expected_hashes:
        raise ValueError("TRAIN_PROVENANCE_SHA_MISMATCH")
    reg = json.loads(registration_path().read_text())
    cal = json.loads(calibration.read_text())
    scope = json.loads(scope_path.read_text())
    budget = json.loads(budget_path.read_text())
    annotation = Path(scope["dataset_root"]) / "annotations/train.json"
    annotation_sha = sha(annotation)
    if (annotation_sha != cal["TRAIN_annotations_sha256"]
            or annotation_sha != "9093c36204bae482c2f74997f531bfe2d8a46464f4541093703ef22db0068c85"
            or budget["epochs"] != 12 or budget["TEST_read"] is not False
            or gate["seed"] != 20261027 or gate["initial_state_tensor_sha256"] != expected_initial
            or gate["final_epochs"] != 12 or gate["final_total_training_timestamps"] != 345048
            or gate["trainable_parameters"] != 606468 or gate["optimizer"] != "AdamW"
            or gate["optimizer_steps"] <= 0 or gate["loss_weights"] != LOSS_WEIGHTS
            or gate["GT_repair_of_predicted_history"] is not False or gate["clip_identity_reset"] is not False):
        raise ValueError("REGISTERED_TRAIN_CONFIGURATION_MISMATCH")

    config = registered_config(args.variant, cal)
    model, metadata = load_v5_membership_gate(gp, config, "cpu", args.expected_checkpoint_sha)
    if (model.training or sum(x.numel() for x in model.parameters()) != 606468
            or any(x.dtype != torch.float32 or not torch.isfinite(x).all().item() or x.requires_grad
                   for x in model.parameters())):
        raise ValueError("CHECKPOINT_PARAMETER_VALIDATION_FAILED")
    trained_tensor_sha = state_tensor_sha256(model)
    if trained_tensor_sha == expected_initial or sha(checkpoint) != args.expected_checkpoint_sha:
        raise ValueError("TRAINED_TENSOR_OR_CHECKPOINT_CHANGED")
    hp = args.work / "train" / args.variant / "final/history.json"
    history = json.loads(hp.read_text())
    if len(history) != 12 or history != gate["training_history"]:
        raise ValueError("FINAL_HISTORY_GATE_MISMATCH")
    train_scope = scope["splits"]["train"]
    manifests = []
    for sid in range(1, 25):
        entry = train_scope["scenes"][str(sid)]
        for cache, key in (("support_cache", "support_manifest_sha256"), ("raw_cache", "raw_manifest_sha256")):
            path = Path(train_scope[cache]) / f"scene_{sid:02d}/manifest.json"
            if sha(path) != entry[key]:
                raise ValueError("TRAIN_INPUT_MANIFEST_SHA_MISMATCH")
            manifests.append({"scene": sid, "kind": cache, "sha256": entry[key]})
    logs = []
    for epoch, record in enumerate(history, 1):
        coverage, training = record["coverage"], record["train"]
        expected_order = np.random.default_rng(reg["seed"] + epoch).permutation(reg["final_train_scenes"]).tolist()
        if (record["epoch"] != epoch or coverage != gate["final_epoch_coverage"][epoch - 1]
                or [x["scene"] for x in coverage] != expected_order or len(coverage) != 24
                or sum(x["timestamps"] for x in coverage) != training["frames"]
                or training["frames"] != 28754 or training["diagnostics"]["current_detections"] != 557002):
            raise ValueError("FINAL_TRAIN_COVERAGE_OR_ORDER_MISMATCH")
        for item in coverage:
            entry = train_scope["scenes"][str(item["scene"])]
            if (item["timestamps"] != entry["timestamps"]
                    or item["diagnostics"]["current_detections"] != entry["candidates"]
                    or item["support_manifest_sha256"] != entry["support_manifest_sha256"]
                    or item["raw_manifest_sha256"] != entry["raw_manifest_sha256"]
                    or item["first_timestamp"] != 1 or item["last_timestamp"] != item["timestamps"]
                    or not item["gradient_finite"] or not math.isfinite(item["gradient_norm_max"])
                    or item["scene_reset_count"] != 1 or item["GT_repaired_model_IDs"] != 0
                    or not all(math.isfinite(x) for x in item["loss_sums"].values())):
                raise ValueError("FINAL_TRAIN_SCENE_INVARIANT_FAILED")
            path = args.work / "train" / args.variant / "final/logs" / f"train_epoch{epoch:02d}_scene{item['scene']:02d}.jsonl"
            if sha(path) != item["log_sha256"]:
                raise ValueError("FINAL_TRAIN_SCENE_LOG_SHA_MISMATCH")
            logs.append({"path": str(path.relative_to(args.work)), "sha256": item["log_sha256"]})
    progress = json.loads((args.work / "train" / args.variant / "final/progress.json").read_text())
    if not progress["complete"] or progress["completed_epochs"] != 12 or progress["epoch"] != 13 or progress["budget"] != 12:
        raise ValueError("FINAL_TRAIN_PROGRESS_INCOMPLETE")
    try:
        fields = (Path("/proc") / str(args.worker_pid) / "stat").read_text().rsplit(") ", 1)[1].split()
        worker_live = fields[19] == args.worker_start_tick and fields[0] != "Z"
    except (OSError, IndexError):
        worker_live = False
    if worker_live:
        raise ValueError("FINAL_TRAIN_WORKER_STILL_LIVE")
    with args.worker_log.open("rb") as stream:
        stream.seek(max(0, args.worker_log.stat().st_size - 24000))
        lines = stream.read().splitlines()
    footer = None
    for line in reversed(lines):
        try:
            candidate = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if candidate.get("event") == "FULL_TRAIN_GATE_PASSED":
            footer = candidate
            break
    if not footer or footer.get("variant") != args.variant or footer.get("checkpoint_sha256") != args.expected_checkpoint_sha or footer.get("final_epochs") != 12:
        raise ValueError("FINAL_TRAIN_COMPLETION_FOOTER_MISSING")

    pointer_path = args.work / "train" / args.variant / "final/RESUME.json"
    pointer = json.loads(pointer_path.read_text())
    resume_path = pointer_path.parent / pointer["checkpoint"]
    if (resume_path.parent != pointer_path.parent or sha(resume_path) != pointer["sha256"]
            or pointer["v5_source_sha256"] != source or pointer["scope_sha256"] != gate["scope_sha256"]):
        raise ValueError("FINAL_RESUME_PAYLOAD_SHA_MISMATCH")
    resume = torch.load(resume_path, map_location="cpu", weights_only=False)
    if (not resume["progress"]["complete"] or resume["progress"]["epoch"] != 13
            or resume["progress"]["history"] != history or resume["progress"]["optimizer_steps"] != gate["optimizer_steps"]
            or resume["tracking_config"] != gate["tracking_config"]
            or resume["initial_state_tensor_sha256"] != expected_initial
            or not resume["optimizer"]["state"] or not resume["optimizer"]["param_groups"]
            or not {"python", "numpy", "torch", "cuda"} <= set(resume["RNG"])
            or not isinstance(resume["RNG"]["cuda"], list) or len(resume["RNG"]["cuda"]) != 1
            or not isinstance(resume["RNG"]["torch"], torch.Tensor)
            or not all(isinstance(x, torch.Tensor) and x.dtype == torch.uint8 for x in resume["RNG"]["cuda"])
            or tensors_sha(resume["model"]) != trained_tensor_sha):
        raise ValueError("FINAL_RESUME_PAYLOAD_CONTENT_MISMATCH")
    if any(not math.isfinite(group["lr"]) or group["lr"] <= 0 for group in resume["optimizer"]["param_groups"]):
        raise ValueError("NONFINITE_OPTIMIZER_LEARNING_RATE")
    for values in resume["optimizer"]["state"].values():
        for value in values.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all().item():
                raise ValueError("NONFINITE_OPTIMIZER_STATE")

    initial_proof = args.repo / "worldmot/results/v5_fresh_final_initialization_check.json"
    initial = json.loads(initial_proof.read_text())
    if (initial["common_initial_tensor_sha256"] != expected_initial or initial["seed"] != 20261027
            or initial["model_trainable_parameters"] != 606468 or initial["v5_source_sha256"] != source
            or initial["variants"][args.variant]["initial_state_tensor_sha256"] != expected_initial):
        raise ValueError("FRESH_INITIALIZATION_PROOF_MISMATCH")
    if v5_source_fingerprint() != source or sha(Path(__file__)) != helper_sha:
        raise ValueError("VERIFICATION_SOURCE_CHANGED")
    launch_receipt = args.runtime / "phase21_postpublication_state.json"
    launch = json.loads(launch_receipt.read_text())["models"][args.variant]
    result = {"status": args.variant + "_FRESH_FULL24_FINAL12_STRICTLY_VERIFIED",
              "verified_UTC": datetime.now(timezone.utc).isoformat(), "variant": args.variant,
              "checkpoint_sha256": args.expected_checkpoint_sha, "train_gate_sha256": metadata["train_gate_sha256"],
              "trained_tensor_sha256": trained_tensor_sha, "initial_state_tensor_sha256": expected_initial,
              "v5_source_sha256": source, "source_commit_recorded_in_gate": gate["source_commit"],
              "provenance_sha256": actual_hashes, "TRAIN_annotation_current_sha256": annotation_sha,
              "TRAIN_annotation_bytes_hashed_for_integrity": True, "TEST_GT_read": False,
              "final_epochs": 12, "scenes_per_epoch": 24, "paired_timestamps_per_epoch": 28754,
              "total_training_paired_timestamps": 345048, "candidates_per_epoch": 557002,
              "total_candidate_exposures": 557002 * 12, "parameters": 606468, "precision": "float32",
              "optimizer_steps": gate["optimizer_steps"], "all_parameters_finite": True,
              "strict_frozen_loader_cpu": metadata, "forward_calls": 0, "optimizer_created_during_verification": False,
              "actual_scene_log_files_verified": len(logs), "actual_input_manifests_verified": len(manifests),
              "all_gradients_finite": True, "scene_internal_ID_resets": 0, "GT_repaired_model_IDs": 0,
              "registered_per_epoch_scene_order_verified": True, "history_sha256": sha(hp),
              "final_TRAIN_membership_loss": history[-1]["train"]["losses"]["per_camera_membership"],
              "TRAIN_epoch_compute_wall_s_sum_not_inference_FPS": sum(x["train"]["wall_s"] for x in history),
              "worker_terminal": True, "normal_completion_footer_verified": True, "detached_worker_exit_code": None,
              "worker_pid": args.worker_pid, "worker_start_tick": args.worker_start_tick,
              "original_launch_receipt_sha256": sha(launch_receipt), "original_launch": launch,
              "progress_epoch13_is_post_loop_index_not_extra_training": True,
              "verification_includes_TEST": False, "verification_started_any_model_TEST": False,
              "resume_payload_verification": {"relative_path": str(resume_path.relative_to(args.work)),
                  "sha256": pointer["sha256"], "model_tensor_sha256": trained_tensor_sha, "complete_epochs": 12,
                  "history_identical_to_final_gate": True, "optimizer_state_present": True,
                  "optimizer_state_parameter_entries": len(resume["optimizer"]["state"]),
                  "optimizer_state_tensors_finite": True, "RNG_states_present": sorted(resume["RNG"]),
                  "CUDA_RNG_devices": len(resume["RNG"]["cuda"]), "optimizer_created_or_stepped_for_verification": False},
              "fresh_initialization_proof_sha256": sha(initial_proof), "verification_helper_sha256": helper_sha,
              "all_scene_log_files": logs, "input_manifests": manifests}
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("all_scene_log_files", "input_manifests")}))


if __name__ == "__main__":
    main()
