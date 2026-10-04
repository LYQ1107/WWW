"""Persistent, conservative monitor for the complete GMT/JEV research chain.

The monitor never changes hyperparameters and never starts a duplicate live
training or post-Stage2 process.  It resumes Stage1/Stage2 from the exact
checkpoint command and then owns the resumable inference, counterfactual,
policy, Oracle-audit, and evaluation supervisor until the final report
artifacts exist.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/liuyeqiang/anaconda3/envs/GMT/bin/python"
STAGE1_OUTPUT = ROOT / "outputs/stage1_single_gpu"
STAGE2_OUTPUT = ROOT / "outputs/stage2_single_gpu"
STAGE1_FINAL = STAGE1_OUTPUT / "model_16000.pth"
STAGE1_VALIDATION = STAGE1_OUTPUT / "checkpoint_validation.json"
STAGE2_FINAL = STAGE2_OUTPUT / "model_20000.pth"
STATUS_DIR = ROOT / "outputs/research_monitor"
STATUS_PATH = STATUS_DIR / "status.json"
# The historical post-Stage2 supervisor targets the old proxy layout and must
# never be started by this long-lived monitor.  The v2 supervisor is armed
# explicitly only after its final-checkpoint paths and resumable gates have
# been audited.
POST_PIPELINE_SCRIPT = ROOT / "reproduction_tools/chain_research_after_stage2_v2.py"
POST_PIPELINE_ARM = ROOT / "outputs/research_v2/ARM_FINAL_PIPELINE"
POST_PIPELINE_DIR = ROOT / "outputs/research_pipeline"
POST_PIPELINE_STATUS = POST_PIPELINE_DIR / "PIPELINE_COMPLETE.json"
MAX_RESTARTS = 3
POLL_SECONDS = 60


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def process_commands() -> List[Dict[str, str]]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        match = re.match(r"\s*(\d+)\s+(\d+)\s+(.*)", line)
        if match:
            rows.append({"pid": match.group(1), "ppid": match.group(2), "cmd": match.group(3)})
    return rows


def matching_processes(fragment: str) -> List[Dict[str, str]]:
    """Return processes whose real argv contains the requested script token.

    Matching arbitrary command-line substrings makes an observation command
    such as ``rg 'chain_stage2_single_gpu.sh'`` look like the Stage2
    supervisor.  Token matching keeps the monitor from suppressing a needed
    restart because the monitor's own diagnostics happen to mention a script
    name.
    """

    def is_target_token(token: str) -> bool:
        return token == fragment or token.endswith("/" + fragment)

    matches: List[Dict[str, str]] = []
    for row in process_commands():
        if int(row["pid"]) == os.getpid():
            continue
        try:
            raw_argv = Path(f"/proc/{row['pid']}/cmdline").read_bytes()
            argv = [part.decode("utf-8", errors="replace") for part in raw_argv.split(b"\0") if part]
        except ValueError:
            argv = []
        except OSError:
            # The process may exit between ``ps`` and /proc.  Keep a small
            # fallback for the rare case where /proc is unavailable, while
            # excluding shell ``-c`` wrappers whose script text is not argv.
            try:
                argv = shlex.split(row["cmd"])
            except ValueError:
                argv = []
            if len(argv) > 1 and argv[1] in {"-c", "-lc", "-ic", "-ec"}:
                argv = []
        if any(is_target_token(token) for token in argv):
            matches.append(row)
    return matches


def is_stage1_training_process(row: Dict[str, str]) -> bool:
    """Match Stage1's output target without matching its checkpoint path.

    Stage2 loads ``outputs/stage1_single_gpu/model_16000.pth``.  Testing only
    for the directory name therefore makes a live Stage2 process appear in
    the Stage1 status section even though the restart decisions remain
    correct.
    """
    command = row["cmd"]
    return (
        "OUTPUT_DIR outputs/stage1_single_gpu" in command
        or f"OUTPUT_DIR {STAGE1_OUTPUT}" in command
    )


def last_checkpoint(directory: Path) -> str:
    path = directory / "last_checkpoint"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def checkpoint_info(directory: Path) -> Dict[str, object]:
    name = last_checkpoint(directory)
    path = directory / name if name else None
    info: Dict[str, object] = {"name": name, "exists": bool(path and path.exists())}
    if path and path.exists():
        stat = path.stat()
        info.update({"path": str(path), "bytes": stat.st_size, "mtime": stat.st_mtime})
    return info


def stage1_validation_complete() -> bool:
    if not STAGE1_VALIDATION.is_file():
        return False
    try:
        payload = json.loads(STAGE1_VALIDATION.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        payload.get("status") == "PASS"
        and payload.get("iteration") == 16000
        and payload.get("scheduler_last_epoch") == 20000
    )


def iter_from_log(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines[-200:]):
        matches = re.findall(r"\biter:\s*(\d+)", line)
        if matches:
            return int(matches[-1])
    return None


def stage1_command() -> List[str]:
    return [
        PYTHON,
        "-u",
        "train_net.py",
        "--num-gpus",
        "1",
        "--config-file",
        "configs/VISION_stage1.yaml",
        "--resume",
        "SOLVER.IMS_PER_BATCH",
        "1",
        "SOLVER.TRAIN_ITER",
        "16000",
        "MODEL.WEIGHTS",
        "checkpoints/backbone/CH_FPN_1x_key_adapted.pth",
        "OUTPUT_DIR",
        "outputs/stage1_single_gpu",
    ]


def _training_environment() -> Dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "GMT_DISTRIBUTED_BACKEND": "gloo",
            "GMT_CPU_COLLECTIVES": "1",
            "GMT_CHECKPOINT_BACKBONE": "1",
            "GMT_TRAIN_PROGRESS": "1",
            "OMP_NUM_THREADS": "1",
        }
    )
    return env


def start_stage1() -> int:
    STAGE1_OUTPUT.mkdir(parents=True, exist_ok=True)
    env = _training_environment()
    log = (STAGE1_OUTPUT / "monitor_restart.log").open("a", encoding="utf-8")
    process = subprocess.Popen(
        stage1_command(),
        cwd=ROOT,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process.pid


def stage2_command() -> List[str]:
    return [
        PYTHON,
        "-u",
        "train_net.py",
        "--num-gpus",
        "1",
        "--config-file",
        "configs/VISION_stage2.yaml",
        "--resume",
        "SOLVER.IMS_PER_BATCH",
        "1",
        "SOLVER.TRAIN_ITER",
        "20000",
        "MODEL.WEIGHTS",
        str(STAGE1_FINAL),
        "OUTPUT_DIR",
        str(STAGE2_OUTPUT),
    ]


def start_stage2() -> int:
    STAGE2_OUTPUT.mkdir(parents=True, exist_ok=True)
    log = (STAGE2_OUTPUT / "monitor_restart.log").open("a", encoding="utf-8")
    process = subprocess.Popen(
        stage2_command(),
        cwd=ROOT,
        env=_training_environment(),
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process.pid


def post_pipeline_complete() -> bool:
    if not POST_PIPELINE_STATUS.is_file():
        return False
    try:
        payload = json.loads(POST_PIPELINE_STATUS.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return payload.get("status") == "COMPLETE"


def start_post_pipeline() -> int:
    if not POST_PIPELINE_SCRIPT.is_file():
        raise FileNotFoundError(
            f"v2 post-Stage2 supervisor is not armed: {POST_PIPELINE_SCRIPT}"
        )
    POST_PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    log = (POST_PIPELINE_DIR / "monitor_restart.log").open("a", encoding="utf-8")
    process = subprocess.Popen(
        [PYTHON, "-u", str(POST_PIPELINE_SCRIPT)],
        cwd=ROOT,
        env=_training_environment(),
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process.pid


def write_status(payload: Dict[str, object]) -> None:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    temporary = STATUS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(STATUS_PATH)


def snapshot(
    stage1_restarts: int,
    last_error: str = "",
    stage2_restarts: int = 0,
    post_restarts: int = 0,
) -> Dict[str, object]:
    stage1 = [
        row for row in matching_processes("train_net.py")
        if is_stage1_training_process(row)
    ]
    stage2 = [
        row
        for row in matching_processes("train_net.py")
        if "outputs/stage2_single_gpu" in row["cmd"]
    ]
    stage2_supervisor = matching_processes("chain_stage2_single_gpu.sh")
    post_processes = matching_processes("chain_research_after_stage2.py") + matching_processes(
        POST_PIPELINE_SCRIPT.name
    )
    pipeline_status: Dict[str, object] = {}
    pipeline_status_path = POST_PIPELINE_DIR / "pipeline_status.json"
    if pipeline_status_path.is_file():
        try:
            pipeline_status = json.loads(pipeline_status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pipeline_status = {"status": "UNREADABLE"}
    return {
        "updated_utc": now(),
        "stage1": {
            "processes": stage1,
            "iteration": iter_from_log(STAGE1_OUTPUT / "log.txt"),
            "checkpoint": checkpoint_info(STAGE1_OUTPUT),
            "checkpoint_validation": {
                "path": str(STAGE1_VALIDATION),
                "exists": STAGE1_VALIDATION.is_file(),
                "pass": stage1_validation_complete(),
            },
        },
        "stage2": {
            "processes": stage2,
            "checkpoint": checkpoint_info(STAGE2_OUTPUT),
            "chain_log": str(STAGE2_OUTPUT / "chain.log"),
            "supervisor_processes": stage2_supervisor,
        },
        "post_stage2": {
            "processes": post_processes,
            "status": pipeline_status,
            "complete": post_pipeline_complete(),
            "complete_marker": str(POST_PIPELINE_STATUS),
            "log": str(POST_PIPELINE_DIR / "pipeline.log"),
        },
        "restart_count": stage1_restarts + stage2_restarts + post_restarts,
        "stage1_restart_count": stage1_restarts,
        "stage2_restart_count": stage2_restarts,
        "post_stage2_restart_count": post_restarts,
        "last_error": last_error,
        "policy": {
            "max_restarts": MAX_RESTARTS,
            "poll_seconds": POLL_SECONDS,
            "hyperparameters_unchanged": True,
        },
    }


def main() -> None:
    stage1_restarts = 0
    stage2_restarts = 0
    post_restarts = 0
    last_error = ""
    while True:
        stage1_checkpoint = checkpoint_info(STAGE1_OUTPUT)
        final_exists = STAGE1_FINAL.exists()
        stage1 = [
            row for row in matching_processes("train_net.py")
            if is_stage1_training_process(row)
        ]
        if not final_exists and not stage1:
            if stage1_restarts >= MAX_RESTARTS:
                last_error = "Stage1 exited without final checkpoint; restart limit reached"
                write_status(snapshot(stage1_restarts, last_error, stage2_restarts, post_restarts))
                return
            try:
                pid = start_stage1()
                stage1_restarts += 1
                last_error = ""
                write_status(snapshot(stage1_restarts, f"Stage1 resumed with pid {pid}", stage2_restarts, post_restarts))
            except Exception as exc:  # keep the monitor status inspectable
                last_error = f"failed to resume Stage1: {type(exc).__name__}: {exc}"
                write_status(snapshot(stage1_restarts, last_error, stage2_restarts, post_restarts))
        else:
            # Once Stage1 is complete, the original supervisor owns the first
            # Stage2 launch.  If that supervisor is gone and Stage2 has not
            # started, take over with the identical command.  A live Stage2
            # process is never duplicated.
            stage2 = [
                row
                for row in matching_processes("train_net.py")
                if "outputs/stage2_single_gpu" in row["cmd"]
            ]
            supervisor = matching_processes("chain_stage2_single_gpu.sh")
            if final_exists and not STAGE2_FINAL.exists() and not stage2 and not supervisor:
                if not stage1_validation_complete():
                    last_error = "Stage1 final exists but checkpoint validation gate is missing or failed"
                    write_status(snapshot(stage1_restarts, last_error, stage2_restarts, post_restarts))
                    time.sleep(POLL_SECONDS)
                    continue
                if stage2_restarts >= MAX_RESTARTS:
                    last_error = "Stage2 exited without final checkpoint; restart limit reached"
                    write_status(snapshot(stage1_restarts, last_error, stage2_restarts, post_restarts))
                    return
                try:
                    pid = start_stage2()
                    stage2_restarts += 1
                    last_error = f"Stage2 resumed with pid {pid}"
                except Exception as exc:
                    last_error = f"failed to resume Stage2: {type(exc).__name__}: {exc}"
            # Stage2 completion transitions into the persistent research
            # pipeline.  Do not launch it while the training process or its
            # original supervisor is still alive.
            if STAGE2_FINAL.exists() and not stage2 and not supervisor and not post_pipeline_complete():
                if not POST_PIPELINE_SCRIPT.is_file() or not POST_PIPELINE_ARM.is_file():
                    last_error = (
                        "post-Stage2 held: audited v2 supervisor is not armed; "
                        "legacy supervisor will not be started"
                    )
                    write_status(snapshot(stage1_restarts, last_error, stage2_restarts, post_restarts))
                    time.sleep(POLL_SECONDS)
                    continue
                post_processes = matching_processes("chain_research_after_stage2.py") + matching_processes(
                    POST_PIPELINE_SCRIPT.name
                )
                if not post_processes:
                    if post_restarts >= MAX_RESTARTS:
                        last_error = "post-Stage2 pipeline exited; restart limit reached"
                        write_status(snapshot(stage1_restarts, last_error, stage2_restarts, post_restarts))
                        return
                    try:
                        pid = start_post_pipeline()
                        post_restarts += 1
                        last_error = f"post-Stage2 pipeline started with pid {pid}"
                    except Exception as exc:
                        last_error = f"failed to start post-Stage2 pipeline: {type(exc).__name__}: {exc}"
            write_status(snapshot(stage1_restarts, last_error, stage2_restarts, post_restarts))
        if final_exists and not stage1:
            # Keep monitoring after Stage2: the actual research deliverable is
            # the post-Stage2 inference/policy/evaluation chain.
            if STAGE2_FINAL.exists() and post_pipeline_complete():
                write_status(snapshot(stage1_restarts, "complete research chain", stage2_restarts, post_restarts))
                return
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
