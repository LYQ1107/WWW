#!/usr/bin/env python3
"""Finite frozen-reference TEST inference only; never imports a GT evaluator."""
import argparse
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

from run_worldobject_v5_controller import (process_identity, gpu_snapshot, read_json, write_json,
                                         gpu_launch_lock, live_gpu_reservations)
from worldobject_v5_reference_scope import test_runner_completed
from worldmot.runtime.world_object_v5_contract import require_v5_scope
from worldmot.data.worldobject_cache import sha256


def terminal_artifact_status(job, artifact_verified):
    if artifact_verified:
        return "COMPLETE"
    job["missing_output_checks"] = job.get("missing_output_checks", 0) + 1
    return ("ENGINEERING_REVIEW_REQUIRED" if job.get("exit_code") not in {None, 0}
            or job["missing_output_checks"] >= 3 else "EXITED_PENDING_ARTIFACT")


def prior_verified_completion(proof, manifest, manifest_sha, scope_sha, auth_sha, checkpoint_sha):
    """Reuse an earlier output-byte verification for scheduling only.

    The manifest binds all output hashes. Canonical adoption must still reread
    every output byte after all three final TRAIN gates have been frozen.
    """
    return (proof.get("manifest_sha256") == manifest_sha
            and manifest.get("status") == "COMPLETE"
            and manifest.get("scope_sha256") == scope_sha
            and manifest.get("approval", {}).get("approval_sha256") == auth_sha
            and manifest.get("checkpoint_sha256") == checkpoint_sha == proof.get("checkpoint_sha256")
            and manifest.get("GT_in_inference") is False and manifest.get("causal") is True
            and proof.get("timestamps") == manifest.get("timestamps")
            and proof.get("candidates") == manifest.get("candidate_count")
            and proof.get("tracks_sha256") == manifest.get("output_sha256", {}).get("tracks.jsonl"))


def main():
    p = argparse.ArgumentParser()
    for key in ("repo", "runtime", "output-root"):
        p.add_argument("--" + key, type=Path, required=True)
    a = p.parse_args()
    lock = (a.runtime / "reference_test_queue.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    private = a.output_root / "provenance"
    scope_path, auth_path = private / "scope_v5_reference_test.json", private / "authorization_v5_reference_test.json"
    scope, _ = require_v5_scope(scope_path, auth_path, "replay", "test", "B0")
    auth = read_json(auth_path)
    if scope["variants"] != ["B0", "B1"] or set(auth["operations"]) != {"replay", "preflight"}:
        raise ValueError("INFERENCE_ONLY_FROZEN_REFERENCE_SCOPE_REQUIRED")
    path = a.runtime / "auxiliary_jobs.json"
    jobs, children = read_json(path, {}), {}
    verify = test_runner_completed()
    checked, complete = {}, {"B0": {}, "B1": {}}
    prior = read_json(a.runtime / "reference_inference_status.json", {})
    scope_sha, auth_sha = sha256(scope_path), sha256(auth_path)
    if (prior.get("reference_scope_sha256") == scope_sha
            and prior.get("reference_authorization_sha256") == auth_sha
            and prior.get("checkpoint_sha256") == scope["frozen_C1_checkpoint_sha256"]):
        for variant in ("B0", "B1"):
            for scene_text, proof in prior.get("scenes", {}).get(variant, {}).items():
                scene = int(scene_text)
                if not 1 <= scene <= 22:
                    raise ValueError("INVALID_PRIOR_COMPLETION_SCENE")
                manifest = a.output_root / f"predictions/{variant}/scene_{scene:02d}/manifest.json"
                if not manifest.is_file():
                    continue
                m, manifest_sha = read_json(manifest), sha256(manifest)
                if not prior_verified_completion(proof, m, manifest_sha, scope_sha, auth_sha,
                                                  scope["frozen_C1_checkpoint_sha256"]):
                    raise ValueError("PRIOR_BYTE_VERIFIED_REFERENCE_MANIFEST_CHANGED")
                if variant == "B0" and m["exact_original_C1_timestamps"] != m["timestamps"]:
                    raise ValueError("B0_MUST_REPRODUCE_EVERY_ORIGINAL_C1_TIMESTAMP")
                checked[variant, scene] = manifest_sha
                complete[variant][scene_text] = {**proof,
                    "scheduling_reused_prior_output_byte_verification": True,
                    "canonical_adoption_still_requires_full_output_byte_verification": True}
    def record_completed(variant, scene):
        folder = a.output_root / f"predictions/{variant}/scene_{scene:02d}"
        manifest = folder / "manifest.json"
        if not manifest.exists():
            return False
        m = verify(folder, scope, variant, scene)
        if (m["scope_sha256"] != sha256(scope_path) or m["approval"]["approval_sha256"] != sha256(auth_path)
                or m["GT_in_inference"] or not m["causal"]):
            raise ValueError("REFERENCE_PROCESS_OR_AUTHORIZATION_CHANGED")
        if variant == "B0" and m["exact_original_C1_timestamps"] != m["timestamps"]:
            raise ValueError("B0_MUST_REPRODUCE_EVERY_ORIGINAL_C1_TIMESTAMP")
        checked[variant, scene] = sha256(manifest)
        complete[variant][str(scene)] = {"manifest_sha256": checked[variant, scene],
            "timestamps": m["timestamps"], "candidates": m["candidate_count"],
            "tracks_sha256": m["output_sha256"]["tracks.jsonl"], "checkpoint_sha256": m["checkpoint_sha256"]}
        return True

    while True:
        for variant in ("B0", "B1"):
            if not (a.output_root / "predictions" / variant).is_dir():
                continue
            for scene in range(1, 23):
                folder = a.output_root / f"predictions/{variant}/scene_{scene:02d}"
                manifest = folder / "manifest.json"
                if manifest.exists() and (variant, scene) not in checked:
                    record_completed(variant, scene)
        attention = []
        for key, job in jobs.items():
            if not key.startswith("reference_test_"):
                continue
            pair = job["variant"], job["scene"]
            if pair in checked and job["status"] != "RUNNING":
                job["status"] = "COMPLETE"
                continue
            if job["status"] not in {"RUNNING", "EXITED_PENDING_ARTIFACT"}:
                continue
            identity = process_identity(job["pid"])
            if identity == job["identity"]:
                continue
            if key in children:
                job["exit_code"] = children[key].poll()
            else:
                try:
                    child, status = os.waitpid(job["pid"], os.WNOHANG)
                    if child:
                        job["exit_code"] = os.waitstatus_to_exitcode(status)
                except ChildProcessError:
                    pass
            # The worker may rename its complete directory AFTER the scan
            # above but BEFORE exit is observed. Recheck at the terminal
            # boundary; a stale scan must never create a failed experiment.
            job["status"] = terminal_artifact_status(job, pair in checked or record_completed(*pair))
            if job["status"] == "ENGINEERING_REVIEW_REQUIRED":
                attention.append(key)
        attention += [key for key, job in jobs.items() if key.startswith("reference_test_") and job["status"] == "ENGINEERING_REVIEW_REQUIRED"]
        status = {"status": "COMPLETE_ALL22_B0_B1_GT_FREE_INFERENCE" if all(len(x) == 22 for x in complete.values())
                  else "ENGINEERING_REVIEW_REQUIRED" if attention else "RUNNING_REFERENCE_ONLY_INFERENCE",
                  "complete_scenes": {v: len(s) for v, s in complete.items()}, "scenes": complete,
                  "reference_scope_sha256": sha256(scope_path), "reference_authorization_sha256": sha256(auth_path),
                  "checkpoint_sha256": scope["frozen_C1_checkpoint_sha256"], "scoring_executed": False,
                  "TEST_GT_used": False, "new_models_or_budget_changed": False,
                  "canonical_adoption_still_requires_full_output_byte_verification": True,
                  "engineering_attention": sorted(set(attention)), "updated_UTC": datetime.now(timezone.utc).isoformat()}
        write_json(a.runtime / "reference_inference_status.json", status)
        write_json(path, jobs)
        if status["status"] == "COMPLETE_ALL22_B0_B1_GT_FREE_INFERENCE":
            write_json(private / "reference_test_coverage.json", status)
            return
        # No current model is stopped, including other references in flight.
        # Scene1 is a full-video B0 engineering precheck before scaling out.
        if attention:
            time.sleep(30.)
            continue
        main_state = read_json(a.runtime / "controller_state.json", {})
        reservations = {j["GPU_UUID"] for j in main_state.get("jobs", {}).values()
                        if j.get("status") == "RUNNING" and j.get("GPU_UUID")}
        reservations.update(j["GPU_UUID"] for j in jobs.values() if j["status"] == "RUNNING" and j.get("GPU_UUID"))
        pending = [(v, s) for v in ("B0", "B1") for s in range(1, 23) if (v, s) not in checked]
        if ("B0", 1) not in checked:
            pending = [("B0", 1)]
        for variant, scene in pending:
            key = f"reference_test_{variant}_scene{scene:02d}"
            if key in jobs or len(reservations) >= 6:
                continue
            idle = [g for g in gpu_snapshot() if not g["occupied"] and g["uuid"] not in reservations
                    and "V100" in g["name"] and g["free_MiB"] >= 25000]
            if not idle or not os.path.ismount("/data1/liuyeqiang/nas/A100"):
                break
            gpu = idle[0]
            command = [sys.executable, "-u", str(a.repo / "worldmot/scripts/run_worldobject_v5_test.py"),
                       "--scope-lock", str(scope_path), "--authorization", str(auth_path),
                       "--calibration", str(private / "calibration_train_fit18.json"),
                       "--output", str(a.output_root), "--variant", variant, "--scene", str(scene), "--device", "cuda:0"]
            log = a.runtime / f"reference_test_{variant}_scene{scene:02d}.log"
            env = {**os.environ, "PYTHONPATH": str(a.repo / "worldmot"), "CUDA_VISIBLE_DEVICES": gpu["uuid"],
                   "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
            with gpu_launch_lock(a.runtime):
                latest_main = read_json(a.runtime / "controller_state.json", {})
                reserved_now = live_gpu_reservations(latest_main.get("jobs", {}), jobs)
                occupied_now = {g["uuid"] for g in gpu_snapshot() if g["occupied"]}
                if len(reserved_now) >= 6 or gpu["uuid"] in reserved_now or gpu["uuid"] in occupied_now:
                    continue
                with log.open("x") as stream:
                    child = subprocess.Popen(command, cwd=a.repo, env=env, stdout=stream,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                jobs[key] = {"pid": child.pid, "identity": process_identity(child.pid), "status": "RUNNING",
                             "variant": variant, "scene": scene, "command": command, "GPU_UUID": gpu["uuid"],
                             "device": "cuda:0", "precision": "float32", "log": str(log), "exit_code": None,
                             "started_UTC": datetime.now(timezone.utc).isoformat()}
                children[key] = child
                reservations.add(gpu["uuid"])
                write_json(path, jobs)
            print({"started_reference": variant, "scene": scene, "pid": child.pid, "GPU_UUID": gpu["uuid"]}, flush=True)
        time.sleep(30.)


if __name__ == "__main__":
    main()
