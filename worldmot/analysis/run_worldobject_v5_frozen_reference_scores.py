#!/usr/bin/env python3
"""Finite B1/B0 full22 validation and native scoring, never touching training."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from run_worldobject_v5_controller import write_json, read_json, process_identity
from worldobject_v5_frozen_reference_scoring import require_publication


def main():
    parser = argparse.ArgumentParser()
    for name in ("repo", "runtime", "output-root", "audit"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--evaluation-python", default="/home/liuyeqiang/anaconda3/bin/python")
    args = parser.parse_args()
    lock = (args.runtime / "frozen_reference_scoring.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = args.runtime / "frozen_reference_scoring_state.json"
    state = read_json(state_path, {"jobs": {}})
    state.update({"parent_pid": os.getpid(), "parent_identity": process_identity(os.getpid()), "new_model_training_or_inference": False,
                  "TEST_parameter_selection": False, "variants": ["B1", "B0"]})
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(args.repo / "worldmot") + ":" + str(args.repo / "worldmot/analysis"),
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    driver = args.repo / "worldmot/analysis/worldobject_v5_frozen_reference_scoring.py"
    for variant in ("B1", "B0"):
        require_publication(args.output_root, args.runtime, variant)
        final = args.output_root / f"evaluation_frozen_references/{variant}/frozen_scoring_receipt.json"
        if final.exists():
            if read_json(final)["status"] != "COMPLETE_FROZEN_REFERENCE_FULL22_NATIVE14":
                raise ValueError("REVIEW_EXISTING_REFERENCE_SCORING_RECEIPT")
            continue
        for operation in ("validate", "score"):
            key = variant + "_" + operation
            previous = state["jobs"].get(key)
            if previous:
                # Adopted live handles are never duplicated. Unexpected exits
                # require explicit engineering review rather than a new score.
                if process_identity(previous["pid"]) == previous["identity"]:
                    raise RuntimeError("PREVIOUS_NATIVE_JOB_STILL_LIVE_NO_DUPLICATE")
                if previous.get("exit_code") != 0:
                    raise RuntimeError("REVIEW_FAILED_NATIVE_JOB_BEFORE_RETRY")
                if operation == "validate":
                    continue
            command = [sys.executable if operation == "validate" else args.evaluation_python,
                "-u", str(driver), "--operation", operation, "--output-root", str(args.output_root),
                "--runtime", str(args.runtime), "--variant", variant, "--audit", str(args.audit)]
            log = args.runtime / f"frozen_reference_{key}.log"
            with log.open("x") as stream:
                child = subprocess.Popen(command, cwd=args.repo, env=env, stdout=stream,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            state["jobs"][key] = {"pid": child.pid, "identity": process_identity(child.pid), "command": command,
                "status": "RUNNING", "exit_code": None, "log": str(log), "started_UTC": datetime.now(timezone.utc).isoformat()}
            state.update({"status": "RUNNING_FROZEN_REFERENCE_" + operation.upper(), "current_variant": variant})
            write_json(state_path, state)
            code = child.wait()
            state["jobs"][key].update(status="COMPLETE" if code == 0 else "ENGINEERING_REVIEW_REQUIRED", exit_code=code)
            write_json(state_path, state)
            if code:
                raise RuntimeError(f"{key} exited {code}; preserve logs and normal training")
    state.update(status="COMPLETE_B0_B1_FULL22_NATIVE14", current_variant=None)
    write_json(state_path, state)


if __name__ == "__main__":
    main()
