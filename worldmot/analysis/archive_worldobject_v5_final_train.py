"""Archive a strictly verified completed V5 TRAIN checkpoint and provenance."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from worldmot.runtime.world_object_v5_contract import v5_source_fingerprint
from worldobject_v5_durable_storage import pinned_nas_root, verified_copy


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    for key in ("repo", "work", "runtime", "nas-root", "strict-proof", "worker-log", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--variant", choices=["B2", "B3", "B4"], required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("PRESERVE_EXISTING_FINAL_TRAIN_COMPLETION")
    own_source_sha = digest(Path(__file__))
    source = "c4f9d1c0b2a290062aba52ae7c0d322176ae5376f9c85688f74d89159e16f733"
    if v5_source_fingerprint() != source:
        raise ValueError("FROZEN_MODEL_SOURCE_CHANGED")
    proof = json.loads(args.strict_proof.read_text())
    gate_path = args.work / "train" / args.variant / "train_gate.json"
    gate = json.loads(gate_path.read_text())
    if (proof["status"] != args.variant + "_FRESH_FULL24_FINAL12_STRICTLY_VERIFIED"
            or proof["variant"] != args.variant or proof["final_epochs"] != 12
            or proof["v5_source_sha256"] != source or proof["train_gate_sha256"] != digest(gate_path)
            or gate["checkpoint_sha256"] != proof["checkpoint_sha256"]
            or gate["source_commit"] != proof["source_commit_recorded_in_gate"]
            or not proof["worker_terminal"] or not proof["normal_completion_footer_verified"]
            or proof["forward_calls"] != 0 or proof["optimizer_created_during_verification"]):
        raise ValueError("STRICT_TRAIN_VERIFICATION_MISMATCH")
    if Path(gate["checkpoint"]).name != gate["checkpoint"]:
        raise ValueError("INVALID_FINAL_CHECKPOINT_NAME")
    final_dir = args.work / "train" / args.variant / "final"
    pointer = json.loads((final_dir / "RESUME.json").read_text())
    if Path(pointer["checkpoint"]).name != pointer["checkpoint"]:
        raise ValueError("INVALID_FINAL_RESUME_NAME")
    resume = final_dir / pointer["checkpoint"]
    if (digest(gate_path.parent / gate["checkpoint"]) != proof["checkpoint_sha256"]
            or digest(resume) != proof["resume_payload_verification"]["sha256"]
            or pointer["sha256"] != proof["resume_payload_verification"]["sha256"]):
        raise ValueError("FINAL_TRAIN_PAYLOAD_SHA_MISMATCH")
    verifier = args.repo / "worldmot/analysis/verify_worldobject_v5_final_train.py"
    verifier_checks = args.repo / "worldmot/results/v5_final_train_verifier_checks.json"
    archive_checks = args.repo / "worldmot/results/v5_final_train_archive_checks.json"
    check_data = json.loads(verifier_checks.read_text())
    archive_check_data = json.loads(archive_checks.read_text())
    if (digest(verifier) != proof["verification_helper_sha256"]
            or check_data["status"] != "PASSED_READ_ONLY_FINAL_TRAIN_VERIFIER_NEGATIVE_CHECKS"
            or check_data["verification_source_sha256"] != proof["verification_helper_sha256"]
            or archive_check_data["status"] != "PASSED_READ_ONLY_FINAL_TRAIN_ARCHIVER_NEGATIVE_CHECK"
            or archive_check_data["archiver_source_sha256"] != own_source_sha):
        raise ValueError("FINAL_TRAIN_VERIFIER_SOURCE_MISMATCH")
    init = args.repo / "worldmot/results/v5_fresh_final_initialization_check.json"
    launch = args.runtime / "phase21_postpublication_state.json"
    if digest(init) != proof["fresh_initialization_proof_sha256"] or digest(launch) != proof["original_launch_receipt_sha256"]:
        raise ValueError("FINAL_TRAIN_INITIALIZATION_OR_LAUNCH_MISMATCH")

    matching = []
    for path in args.runtime.glob("phase*_source_reports_NAS_archive.json"):
        value = json.loads(path.read_text())
        if value.get("remote_commit") == gate["source_commit"]:
            matching.append((path, value))
    if len(matching) != 1:
        raise ValueError("EXACT_FROZEN_SOURCE_SNAPSHOT_RECEIPT_REQUIRED")
    source_receipt, source_info = matching[0]
    source_archive = args.runtime / Path(source_info["path"]).name
    if digest(source_archive) != source_info["sha256"]:
        raise ValueError("FROZEN_SOURCE_SNAPSHOT_SHA_MISMATCH")

    sources = []
    def add(path, name):
        sources.append((Path(path), name))
    for relative in (f"train/{args.variant}/train_gate.json", f"train/{args.variant}/{gate['checkpoint']}",
                     f"train/{args.variant}/final/history.json", f"train/{args.variant}/final/progress.json",
                     f"train/{args.variant}/final/RESUME.json", str(resume.relative_to(args.work)),
                     "provenance/scope_v5_train.json", "provenance/authorization_v5_train.json",
                     "provenance/calibration_train_fit18.json", "provenance/common_train_budget.json"):
        add(args.work / relative, relative)
    for item in proof["all_scene_log_files"]:
        path = args.work / item["path"]
        if digest(path) != item["sha256"]:
            raise ValueError("SCENE_LOG_CHANGED_AFTER_VERIFICATION")
        add(path, item["path"])
    add(args.strict_proof, "verification/strict_verification.json")
    add(verifier, "verification/verify_worldobject_v5_final_train.py")
    add(verifier_checks, "verification/v5_final_train_verifier_checks.json")
    add(archive_checks, "verification/v5_final_train_archive_checks.json")
    add(init, "verification/v5_fresh_final_initialization_check.json")
    add(launch, "verification/original_launch_receipt.json")
    add(args.worker_log, "verification/worker_stdout.log")
    add(args.repo / "worldmot/configs/worldobject_v5/registered.json", "source/registered.json")
    add(source_receipt, "source/source_reports_NAS_archive_receipt.json")
    add(source_archive, "source/" + source_archive.name)
    add(Path(__file__), "verification/" + Path(__file__).name)
    if len(proof["all_scene_log_files"]) != 288 or len({name for _, name in sources}) != len(sources):
        raise ValueError("FINAL_TRAIN_ARCHIVE_SOURCE_SET_MISMATCH")
    files = [{"path": name, "bytes": path.stat().st_size, "sha256": digest(path)} for path, name in sources]
    manifest = {"schema": "worldmot_v5_complete_final_training_archive_v1", "variant": args.variant,
                "created_UTC": datetime.now(timezone.utc).isoformat(), "checkpoint_sha256": proof["checkpoint_sha256"],
                "train_gate_sha256": proof["train_gate_sha256"], "source_commit": gate["source_commit"],
                "v5_source_sha256": source, "final_epochs": 12, "paired_timestamps_per_epoch": 28754,
                "all_files": files}
    mp = args.runtime / f"{args.variant}_final_full24_archive_manifest.json"
    temporary = args.runtime / f"{args.variant}_final_full24_archive_build.tar.gz"
    if mp.exists() or temporary.exists():
        raise FileExistsError("PRESERVE_EXISTING_FINAL_TRAIN_ARCHIVE_WORK")
    mp.write_text(json.dumps(manifest, indent=2) + "\n")
    with tarfile.open(temporary, "w:gz", compresslevel=6) as archive:
        for path, name in sources:
            archive.add(path, arcname=name, recursive=False)
        archive.add(mp, arcname="archive_manifest.json", recursive=False)
    with tarfile.open(temporary, "r:gz") as archive:
        if len(archive.getnames()) != len(files) + 1:
            raise ValueError("FINAL_TRAIN_ARCHIVE_MEMBER_COUNT_MISMATCH")
        for item in files:
            if hashlib.sha256(archive.extractfile(item["path"]).read()).hexdigest() != item["sha256"]:
                raise ValueError("FINAL_TRAIN_ARCHIVE_MEMBER_SHA_MISMATCH")
        if hashlib.sha256(archive.extractfile("archive_manifest.json").read()).hexdigest() != digest(mp):
            raise ValueError("FINAL_TRAIN_ARCHIVE_MANIFEST_SHA_MISMATCH")
    for (path, _), item in zip(sources, files):
        if digest(path) != item["sha256"]:
            raise ValueError("FINAL_TRAIN_ARCHIVE_SOURCE_CHANGED")
    if v5_source_fingerprint() != source or digest(Path(__file__)) != own_source_sha:
        raise ValueError("FINAL_TRAIN_ARCHIVER_SOURCE_CHANGED")
    archive_sha = digest(temporary)
    name = f"{args.variant}_final_full24_{proof['checkpoint_sha256'][:16]}_{archive_sha[:16]}.tar.gz"
    local = args.runtime / name
    if local.exists():
        raise FileExistsError("PRESERVE_EXISTING_FINAL_TRAIN_ARCHIVE")
    temporary.rename(local)
    with pinned_nas_root(args.nas_root) as (anchor, mount):
        readback = verified_copy(local, anchor / "final_training" / name, archive_sha)
    receipt = {"status": "COMPLETE_FINAL_TRAINING_ARCHIVE_FULL_READBACK_VERIFIED",
               "verified_UTC": datetime.now(timezone.utc).isoformat(),
               "path": str(args.nas_root / "final_training" / name), "bytes": local.stat().st_size,
               "sha256": archive_sha, "manifest_sha256": digest(mp), "actual_internal_files_verified": len(files) + 1,
               "model_checkpoint_sha256": proof["checkpoint_sha256"], "train_gate_sha256": proof["train_gate_sha256"],
               "resume_payload_sha256": pointer["sha256"], "strict_verification_receipt_sha256": digest(args.strict_proof),
               "source_snapshot_commit": gate["source_commit"], "archiver_source_sha256": own_source_sha,
               "mount": mount, "readback": readback, "model_processes_modified": False}
    all_gates_present = all((args.work / "train" / v / "train_gate.json").is_file() for v in ("B2", "B3", "B4"))
    completion = {"status": args.variant + "_FINAL_FULL24_TRAINING_COMPLETE_STRICTLY_VERIFIED_AND_NAS_ARCHIVED",
                  "strict_verification": proof, "archive": receipt,
                  "all_three_gate_files_present_at_archive": all_gates_present,
                  "new_model_TEST_status": ("GATE_FILES_PRESENT_CHECK_CONTROLLER_STATE" if all_gates_present
                                            else "NOT_STARTED_REGISTERED_ALL3_GATE_BARRIER"),
                  "this_TRAIN_completion_claims_scientific_improvement": False}
    with args.output.open("x") as stream:
        json.dump(completion, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
