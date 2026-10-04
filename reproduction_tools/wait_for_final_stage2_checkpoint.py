#!/usr/bin/env python3
"""Wait for Stage2's final checkpoint and validate it exactly once."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/liuyeqiang/anaconda3/envs/GMT/bin/python"
CHECKPOINT = ROOT / "outputs/stage2_single_gpu/model_20000.pth"
REPORT = ROOT / "outputs/stage2_single_gpu/validations/model_20000.json"
STATUS = ROOT / "outputs/research_v2/final_stage2_checkpoint_waiter.json"


def write_status(status: str, **extra: object) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "updated_utc": dt.datetime.now(dt.timezone.utc).isoformat(), **extra}
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def main() -> None:
    while not CHECKPOINT.is_file():
        write_status("WAITING", checkpoint=str(CHECKPOINT))
        time.sleep(30)
    if REPORT.is_file():
        try:
            existing = json.loads(REPORT.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            existing = {}
        if existing.get("status") == "PASS":
            write_status("PASS", checkpoint=str(CHECKPOINT), validation=str(REPORT), reused=True)
            return
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    command = [
        PYTHON,
        "-u",
        str(ROOT / "reproduction_tools/validate_training_checkpoint.py"),
        "--checkpoint",
        str(CHECKPOINT),
        "--expected-iteration",
        "20000",
        "--expected-scheduler-iteration",
        "20000",
        "--output",
        str(REPORT),
    ]
    write_status("VALIDATING", checkpoint=str(CHECKPOINT), validation=str(REPORT))
    result = subprocess.run(command, cwd=ROOT, check=False)
    try:
        report = json.loads(REPORT.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        report = {"status": "UNREADABLE"}
    if result.returncode or report.get("status") != "PASS":
        write_status("FAIL", checkpoint=str(CHECKPOINT), validation=str(REPORT), report=report)
        raise SystemExit(result.returncode or 1)
    write_status("PASS", checkpoint=str(CHECKPOINT), validation=str(REPORT), report=report)


if __name__ == "__main__":
    main()
