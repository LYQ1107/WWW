#!/usr/bin/env python3
"""Adopt the frozen V5 runs and continue their finite registered pipeline.

This scheduler never kills a process, changes a model option, reads TEST scores
to choose a model, or edits inference/training source. Completed scenes and
weight gates remain authoritative. An unexpected failure is preserved for
engineering review instead of silently launching a different experiment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


FROZEN_SOURCE = "c4f9d1c0b2a290062aba52ae7c0d322176ae5376f9c85688f74d89159e16f733"
VARIANTS = ("B0", "B1", "B2", "B3", "B4")
TRAINED = ("B3", "B4", "B2")
LEGAL = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 15, 16, 18, 19]


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial." + str(os.getpid()))
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def process_identity(pid):
    """Start tick prevents adopting an unrelated process after PID recycling."""
    try:
        text = Path(f"/proc/{int(pid)}/stat").read_text()
        suffix = text[text.rfind(")") + 2:].split()
        if suffix[0] == "Z":
            return None
        argv = Path(f"/proc/{int(pid)}/cmdline").read_bytes().split(b"\0")
        return {"start_tick": suffix[19], "argv": [v.decode() for v in argv if v]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def gpu_snapshot():
    devices = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,name,memory.total,memory.free", "--format=csv,noheader,nounits"
    ], text=True)
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"
    ], text=True)
    occupied = {line.split(",")[0].strip() for line in apps.splitlines() if line.strip()}
    rows = []
    for line in devices.splitlines():
        uuid, name, total, free = [x.strip() for x in line.split(",")]
        rows.append({"uuid": uuid, "name": name, "total_MiB": int(total),
                     "free_MiB": int(free), "occupied": uuid in occupied})
    return rows


class Controller:
    def __init__(self, args):
        self.args = args
        self.repo, self.runtime, self.root = args.repo, args.runtime, args.output_root
        self.private = self.root / "provenance"
        self.state_path = self.runtime / "controller_state.json"
        self.state = read_json(self.state_path, {
            "schema": "v5_finite_controller_v1", "created_UTC": utc(),
            "frozen_source_sha256": FROZEN_SOURCE, "jobs": {}, "events": [],
            "TEST_parameter_selection": False, "normal_jobs_terminated": 0,
        })
        if self.state["frozen_source_sha256"] != FROZEN_SOURCE:
            raise ValueError("CONTROLLER_SOURCE_CONTRACT_CHANGED")
        self.children = {}
        self.source_path = Path(__file__).resolve()
        self.state["controller_sha256"] = digest(self.source_path)
        self.state["controller_pid"] = os.getpid()
        self.state["inference_source_excludes_analysis"] = True
        self.environment = {**os.environ, "PYTHONPATH": str(self.repo / "worldmot"),
                            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                            "MKL_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
        self.adopt_selections()

    def event(self, kind, **fields):
        row = {"UTC": utc(), "event": kind, **fields}
        self.state["events"].append(row)
        if not self.args.dry_run:
            with (self.runtime / "controller_events.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)

    def save(self):
        self.state["updated_UTC"] = utc()
        if not self.args.dry_run:
            write_json(self.state_path, self.state)

    def adopt_selections(self):
        ledger = read_json(self.runtime / "selection_launches.json")
        if ledger is None or ledger["source_sha256"] != FROZEN_SOURCE:
            raise ValueError("EXISTING_SELECTION_LAUNCH_LEDGER_REQUIRED")
        for variant, original in ledger["jobs"].items():
            key = "train_" + variant + "_selection"
            if key in self.state["jobs"]:
                continue
            identity = process_identity(original["pid"])
            if identity and identity["argv"] != original["command"]:
                raise ValueError("REFUSE_TO_ADOPT_DIFFERENT_PROCESS")
            self.state["jobs"][key] = {
                **original, "status": "RUNNING" if identity else "EXITED_ADOPTED",
                "adopted": True, "identity": identity, "attempt": 1,
                "exit_code": None, "result": None,
            }

    def refresh_processes(self):
        for key, job in self.state["jobs"].items():
            if job["status"] != "RUNNING":
                continue
            process = self.children.get(key)
            if process is not None and process.poll() is not None:
                job["exit_code"] = process.returncode
            identity = process_identity(job["pid"])
            if identity and identity == job["identity"]:
                continue
            job["status"] = "EXITED"
            job["ended_UTC"] = utc()
            # An adopted process is not our child: no fictional exit code.
            self.event("JOB_EXITED", key=key, exit_code=job["exit_code"],
                       adopted=job.get("adopted", False))

    def running(self, key):
        return self.state["jobs"].get(key, {}).get("status") == "RUNNING"

    def launch(self, key, command, gpu=None, resume=False):
        if self.running(key):
            return False
        previous = self.state["jobs"].get(key)
        if previous and (previous.get("exit_code") not in (None, 0)
                         or (not resume and previous.get("result") != "COMPLETE")):
            # A failed attempt requires a diagnosed, explicit retry receipt.
            retry = read_json(self.runtime / "engineering_retries" / (key + ".json"))
            if not retry or retry.get("previous_pid") != previous["pid"] or not retry.get("same_frozen_inputs"):
                self.state.setdefault("engineering_attention", {})[key] = {"reason": "REVIEW_EXIT_WITHOUT_VALID_OUTPUT", "previous": previous}
                return False
        if gpu:
            occupied = {x["uuid"] for x in gpu_snapshot() if x["occupied"]}
            if gpu["uuid"] in occupied:
                return False
        if not self.storage_ready():
            return False
        attempt = (previous or {}).get("attempt", 0) + 1
        log = self.runtime / "worker_logs" / f"{key}.attempt{attempt:02d}.log"
        if self.args.dry_run:
            self.state.setdefault("dry_run_launches", []).append({"key": key, "command": command, "GPU": gpu})
            return True
        log.parent.mkdir(exist_ok=True)
        env = {**self.environment, "CUDA_VISIBLE_DEVICES": gpu["uuid"] if gpu else ""}
        with log.open("x") as stream:
            process = subprocess.Popen(command, cwd=self.repo, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        identity = process_identity(process.pid)
        self.children[key] = process
        if previous:
            self.state.setdefault("past_attempts", []).append({"key": key, **previous})
        self.state["jobs"][key] = {
            "pid": process.pid, "command": command, "identity": identity,
            "status": "RUNNING", "attempt": attempt, "log": str(log),
            "device": "cuda:0" if gpu else "cpu", "GPU_UUID": gpu["uuid"] if gpu else None,
            "GPU_name": gpu["name"] if gpu else None, "precision": "float32",
            "started_UTC": utc(), "exit_code": None, "result": None, "adopted": False,
        }
        self.event("JOB_STARTED", key=key, pid=process.pid, GPU_UUID=gpu["uuid"] if gpu else None, resume=resume)
        self.save()
        return True

    def mark_complete(self, key, result):
        job = self.state["jobs"].get(key)
        if job and job.get("result") != "COMPLETE":
            job["result"] = "COMPLETE"
            job["completion_proof"] = result
            self.event("ARTIFACT_COMPLETE", key=key, proof=result)

    def storage_ready(self):
        mount = Path("/data1/liuyeqiang/nas/A100")
        if not os.path.ismount(mount):
            self.state["waiting_reason"] = "NAS_NOT_MOUNTED_EXISTING_JOBS_NOT_TERMINATED"
            return False
        if shutil.disk_usage(self.runtime).free < 15 * 1024**3:
            self.state["waiting_reason"] = "LOCAL_FREE_SPACE_RESERVE_15_GiB"
            return False
        return True

    def free_gpus(self):
        rows = gpu_snapshot()
        reservations = {j["GPU_UUID"] for j in self.state["jobs"].values()
                        if j["status"] == "RUNNING" and j.get("GPU_UUID")}
        # Read-only TRAIN audits may have been launched outside this scheduler.
        # Include their live leases in the human's six-device maximum.
        auxiliary = read_json(self.runtime / "auxiliary_jobs.json", {})
        for job in auxiliary.values():
            if job.get("GPU_UUID") and process_identity(job["pid"]) == job.get("identity"):
                reservations.add(job["GPU_UUID"])
        self.state["hardware"] = rows
        capacity = max(0, 6 - len(reservations))
        return [x for x in rows if not x["occupied"] and x["uuid"] not in reservations
                and "V100" in x["name"] and x["free_MiB"] >= 25000][:capacity]

    def command(self, script, **arguments):
        out = [str(self.args.python), "-u", str(self.repo / "worldmot/scripts" / script)]
        for key, value in arguments.items():
            if value is True:
                out.append("--" + key.replace("_", "-"))
            elif value is not None and value is not False:
                out += ["--" + key.replace("_", "-"), str(value)]
        return out

    def training(self):
        common = read_json(self.private / "common_train_budget.json")
        target = common["epochs"] if common else 12
        selected = {}
        for v in TRAINED:
            base = self.root / "train" / v
            gate = read_json(base / "selection/selection_gate.json")
            key = f"train_{v}_selection"
            selected[v] = bool(gate and gate["epochs"] == target and gate["v5_source_sha256"] == FROZEN_SOURCE)
            if selected[v]:
                self.mark_complete(key, {"epochs": gate["epochs"], "full24_checkpoint": False})
            elif gate and gate["epochs"] == 12 and target == 16 and not self.running(key):
                original = self.state["jobs"][key]
                command = list(original["command"])
                if "--resume" not in command:
                    command.append("--resume")
                # Selection extensions stay on their existing CPU execution
                # family. They restore optimizer/RNG and do not restart epochs.
                self.launch(key, command, resume=True)
        if not common:
            self.state["stage"] = "TRAIN_SELECTION_COMMON_B3_CONVERGENCE_RULE_PENDING"
            return False
        if common["v5_source_sha256"] != FROZEN_SOURCE or target not in (12, 16):
            raise ValueError("COMMON_BUDGET_CHANGED")
        self.state["common_epochs"] = target
        final_policy = self.private / "final_device_contract.json"
        policy = {"status": "FROZEN_BEFORE_FINAL_OR_TEST", "variants": ["B2", "B3", "B4"],
                  "device_family": "Tesla V100-DGXS-32GB", "precision": "float32",
                  "threads": 1, "fresh_initialization": True,
                  "selection_execution": "CPU_1_THREAD_FOR_ALL3_DISCARDED_SELECTION_WEIGHTS",
                  "v5_source_sha256": FROZEN_SOURCE, "TEST_scores_used": False}
        if final_policy.exists():
            if read_json(final_policy) != policy:
                raise ValueError("FINAL_DEVICE_POLICY_CHANGED")
        elif not self.args.dry_run:
            write_json(final_policy, policy)
        complete = True
        for v in TRAINED:
            base = self.root / "train" / v
            gate = read_json(base / "train_gate.json")
            key = f"train_{v}_final"
            if gate:
                if (gate["v5_source_sha256"] != FROZEN_SOURCE or gate["final_epochs"] != target
                        or gate["final_scenes"] != list(range(1, 25)) or gate["final_timestamps"] != 28754):
                    raise ValueError("FINAL_TRAIN_GATE_SCOPE_ERROR")
                self.mark_complete(key, {"checkpoint_sha256": gate["checkpoint_sha256"], "epochs": target})
                continue
            complete = False
            if not selected[v] or self.running(key):
                continue
            free = self.free_gpus()
            if not free:
                continue
            resume = (base / "final/RESUME.json").exists()
            command = self.command("train_worldobject_v5.py", scope_lock=self.private / "scope_v5_train.json",
                authorization=self.private / "authorization_v5_train.json", calibration=self.private / "calibration_train_fit18.json",
                output=base, common_budget=self.private / "common_train_budget.json", variant=v, phase="final", device="cuda:0", resume=resume)
            self.launch(key, command, free[0], resume=resume)
        self.state["stage"] = "FRESH_FULL24_FINAL_TRAINING" if any(self.running(f"train_{v}_final") for v in TRAINED) else "TRAIN_SELECTION_OR_WAITING_FOR_IDLE_V100"
        return complete

    def register_test(self):
        lock, auth = self.private / "scope_v5_test.json", self.private / "authorization_v5_test.json"
        if lock.exists() and auth.exists():
            self.mark_complete("register_test", {"scope_sha256": digest(lock), "authorization_sha256": digest(auth)})
            return True
        if lock.exists() != auth.exists():
            raise ValueError("PARTIAL_TEST_REGISTRATION_REQUIRES_REVIEW_DO_NOT_OVERWRITE")
        self.state["stage"] = "REGISTER_ALL3_FULL24_FROZEN_TEST_SCOPE"
        command = self.command("register_worldobject_v5.py", stage="test", output_root=self.root,
            mirror_root=self.args.mirror_root, calibration=self.private / "calibration_train_fit18.json",
            request_file=self.args.request_file, mirror_receipt=self.private / "train_input_mirror.json")
        self.launch("register_test", command)
        return False

    def inference(self):
        done = {v: [] for v in VARIANTS}
        for v in VARIANTS:
            for scene in range(1, 23):
                manifest_path = self.root / f"predictions/{v}/scene_{scene:02d}/manifest.json"
                key = f"test_{v}_scene{scene:02d}"
                if manifest_path.exists():
                    manifest = read_json(manifest_path)
                    if manifest["status"] != "COMPLETE" or manifest["v5_source_sha256"] != FROZEN_SOURCE:
                        raise ValueError("COMPLETED_TEST_SCENE_MANIFEST_MISMATCH")
                    done[v].append(scene)
                    self.mark_complete(key, {"manifest_sha256": digest(manifest_path), "timestamps": manifest["timestamps"]})
        self.state["completed_TEST_scenes"] = {v: len(s) for v, s in done.items()}
        # All variants use V100 FP32. This is independent-scene concurrency,
        # never advertised as one video's latency or RGB-to-ID FPS.
        for v in TRAINED + ("B1", "B0"):
            for scene in range(1, 23):
                key = f"test_{v}_scene{scene:02d}"
                if scene in done[v] or self.running(key):
                    continue
                free = self.free_gpus()
                if not free:
                    return done
                command = self.command("run_worldobject_v5_test.py", scope_lock=self.private / "scope_v5_test.json",
                    authorization=self.private / "authorization_v5_test.json", calibration=self.private / "calibration_train_fit18.json",
                    output=self.root, variant=v, scene=scene, device="cuda:0")
                self.launch(key, command, free[0])
        return done

    def evaluation(self, scenes):
        complete = []
        audit = read_json(self.args.mirror_root / "fulltrain_20261022/audit/data_audit.json")
        expected_names = {s["name"] for s in audit["test"]["scenes"] if s["scene"] in LEGAL}
        for v in VARIANTS:
            metrics = read_json(self.root / f"evaluation/{v}/metrics.json")
            if metrics and metrics.get("status") == "COMPLETE_VALID_SCOPE":
                blocked = metrics["blocked"]
                covered = metrics["coverage"]
                if (set(metrics["evaluated_scenes"]) != expected_names or metrics["raw_GT_modified"]
                        or sorted(x["scene"] for x in covered) != list(range(1, 23))
                        or sum(x["frames"] for x in covered) != 29019
                        or sorted(x["scene"] for x in blocked) != sorted(set(range(1, 23)) - set(LEGAL))
                        or any(x["reason"] != "RAW_GT_DUPLICATE_OR_VERSION_INVALID" for x in blocked)):
                    raise ValueError("LEGAL_ORIGINAL_GT14_SCOPE_REQUIRED")
                complete.append(v)
                self.mark_complete("evaluate_" + v, {"metrics_sha256": digest(self.root / f"evaluation/{v}/metrics.json"), "scenes": LEGAL})
        self.state["completed_native_evaluations"] = complete
        if any(self.running("evaluate_" + v) for v in VARIANTS):
            return complete
        for v in TRAINED + ("B1", "B0"):
            if len(scenes[v]) != 22 or v in complete:
                continue
            command = self.command("evaluate_fullvideo.py", split="test", dataset_root="/data/DATASETS/TRACKING/JDE/VisionTrack",
                audit=self.args.mirror_root / "fulltrain_20261022/audit/data_audit.json", predictions=self.root / "predictions" / v,
                output=self.root / "evaluation" / v, authorization=self.private / "authorization_v5_test.json", method_id="V5_" + v)
            command[0] = str(self.args.evaluation_python)
            self.launch("evaluate_" + v, command)
            break
        return complete

    def live(self):
        proof = self.root / "live_fixed_train_prefix/runtime.json"
        if proof.exists() and read_json(proof).get("status") == "ALL5_FINITE_REAL_LIVE_COMPLETE":
            self.mark_complete("live_fixed_train_prefix", {"runtime_sha256": digest(proof)})
            return True
        if self.running("live_fixed_train_prefix"):
            return False
        free = self.free_gpus()
        if not free:
            return False
        command = self.command("run_worldobject_v5_live.py", scope_lock=self.private / "scope_v5_test.json",
            authorization=self.private / "authorization_v5_test.json", calibration=self.private / "calibration_train_fit18.json",
            output_root=self.root, metadata_root=self.args.metadata_root, device="cuda:0")
        self.launch("live_fixed_train_prefix", command, free[0])
        return False

    def posthoc(self):
        proof = self.root / "v5_identity_attribution.json"
        if proof.exists() and read_json(proof).get("status") == "COMPLETE_FROZEN_ALL5_FULL22_IDENTITY_ATTRIBUTION":
            self.mark_complete("posthoc_all5_full22", {"report_sha256": digest(proof)})
            return True
        if not self.running("posthoc_all5_full22"):
            command = [str(self.args.python), "-u", str(self.repo / "worldmot/analysis/attribute_worldobject_v5_frozen_test.py"),
                       "--scope-lock", str(self.private / "scope_v5_test.json"),
                       "--authorization", str(self.private / "authorization_v5_test.json"),
                       "--output-root", str(self.root), "--output", str(proof)]
            self.launch("posthoc_all5_full22", command)
        return False

    def final_report(self):
        proof = self.root / "v5_research_delivery_receipt.json"
        if proof.exists() and read_json(proof).get("status") == "FULL_REGISTERED_RESEARCH_ARTIFACTS_COMPLETE":
            self.mark_complete("summarize_final", {"receipt_sha256": digest(proof)})
            self.state["automated_work_complete"] = True
            self.state["stage"] = "FULL_ARTIFACTS_READY_GITHUB_PUBLICATION_REQUIRES_VERIFICATION"
            return
        for scene in (4, 8, 12, 16, 20, 24):
            report = read_json(self.root / f"train_birth_attribution/scene_{scene:02d}/report.json")
            if not report or report.get("status") != "COMPLETE_TRAIN_ONLY_CONDITIONAL_BIRTH_DIAGNOSTIC":
                self.state["waiting_reason"] = "TRAIN_BIRTH_ATTRIBUTION_NOT_COMPLETE"
                return
        command = [str(self.args.python), "-u", str(self.repo / "worldmot/analysis/summarize_worldobject_v5.py"),
                   "--repo", str(self.repo), "--runtime", str(self.runtime), "--output-root", str(self.root), "--final"]
        self.launch("summarize_final", command)

    def snapshot_training(self):
        phases = {}
        for v in TRAINED:
            phases[v] = {phase: read_json(self.root / f"train/{v}/{phase}/progress.json")
                         for phase in ("selection", "final")}
        self.state["training_progress"] = phases

    def tick(self):
        from worldmot.runtime.world_object_v5_contract import v5_source_fingerprint
        if v5_source_fingerprint() != FROZEN_SOURCE:
            raise ValueError("FROZEN_TRAIN_INFERENCE_SOURCE_CHANGED_EXISTING_JOBS_NOT_TERMINATED")
        self.refresh_processes()
        if not self.storage_ready():
            self.save()
            return
        self.state.pop("waiting_reason", None)
        self.snapshot_training()
        final_done = self.training()
        if final_done and self.register_test():
            self.state["stage"] = "FULL22_CAUSAL_TEST_AND_NATIVE14_SCORING"
            scenes = self.inference()
            evaluated = self.evaluation(scenes)
            if len(evaluated) == 5:
                self.state["stage"] = "FIXED_REAL_LIVE_AND_POSTHOC_REPORTS"
                self.state["real_live_complete"] = self.live()
                self.state["posthoc_complete"] = self.posthoc()
                if self.state["real_live_complete"] and self.state["posthoc_complete"]:
                    self.final_report()
        self.state["hardware"] = gpu_snapshot()
        self.save()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--mirror-root", type=Path, required=True)
    p.add_argument("--request-file", type=Path, required=True)
    p.add_argument("--metadata-root", type=Path, required=True)
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--evaluation-python", type=Path, default=Path("/home/liuyeqiang/anaconda3/bin/python"))
    p.add_argument("--interval", type=float, default=30.)
    p.add_argument("--once", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    a.runtime.mkdir(parents=True, exist_ok=True)
    lock = (a.runtime / "controller.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sys.path.insert(0, str(a.repo / "worldmot"))
    controller = Controller(a)
    while True:
        try:
            controller.tick()
            controller.state.pop("last_exception", None)
        except Exception as error:
            controller.state["last_exception"] = {"type": type(error).__name__, "message": str(error), "UTC": utc()}
            controller.event("CONTROLLER_ENGINEERING_ATTENTION", **controller.state["last_exception"])
            controller.save()
        if a.once or a.dry_run or controller.state.get("automated_work_complete"):
            print(json.dumps(controller.state, indent=2, allow_nan=False))
            break
        if digest(controller.source_path) != controller.state["controller_sha256"]:
            # Re-exec this scheduler only. Model workers keep their process,
            # optimizer and scene history; the new scheduler adopts their PIDs.
            controller.event("CONTROLLER_REEXEC_ANALYSIS_ONLY_MODEL_WORKERS_PRESERVED")
            controller.save()
            os.execv(sys.executable, [sys.executable, *sys.argv])
        time.sleep(min(60., max(10., a.interval)))


if __name__ == "__main__":
    main()
