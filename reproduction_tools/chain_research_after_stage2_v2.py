#!/usr/bin/env python3
"""Wait for proxy screening, then run the canonical final JEV v2 chain.

The model-4500 jobs are screening only.  Once they release their GPUs, this
gate invokes the isolated v2 implementation, which refuses to proceed until
the canonical Stage2 ``model_20000.pth`` has passed validation.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2ROOT = Path("/data1/liuyeqiang/WWW_jev_v2")
FINAL_V2 = V2ROOT / "reproduction_tools/run_final_v2_pipeline.py"
SCREENING_ROOT = ROOT / "outputs/research_v2/model_4500"
SCREENING_INFERENCE_OUTPUTS = (
    SCREENING_ROOT / "inference_trace_train_full_v2",
    SCREENING_ROOT / "inference_trace_test_full_v2",
    # The proxy official-JEV screening run also reserves the GPU used by the
    # final online evaluation.  Keep the final chain behind it so two
    # controller evaluations cannot run concurrently on the same device.
    SCREENING_ROOT / "official_test_jev_h64_v4",
)
SCREENING_POLICY_ROOT = Path("/data1/liuyeqiang/WWW_jev_v2/results/formal_policy_search")
POLL_SECONDS = 30


def active_screening_processes() -> list[str]:
    """Return live jobs that can reserve GPUs used by the final chain."""

    try:
        rows = subprocess.check_output(["ps", "-eo", "args="], text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return []
    needles = tuple(str(path) for path in SCREENING_INFERENCE_OUTPUTS)
    active = []
    for row in rows:
        if "run_isolated_test_net.py" in row and any(needle in row for needle in needles):
            active.append(row)
        elif "finish_full_proxy_formal.py" in row:
            active.append(row)
        elif "build_jev_counterfactual_v2.py" in row and "formal_full_" in row:
            active.append(row)
    return active


def wait_for_screening_jobs() -> None:
    while True:
        active = active_screening_processes()
        if not active:
            return
        print(
            "[v2 gate] waiting for model-4500 screening jobs before final chain "
            f"({len(active)} active)",
            flush=True,
        )
        time.sleep(POLL_SECONDS)


def main() -> None:
    wait_for_screening_jobs()
    if not FINAL_V2.is_file():
        raise FileNotFoundError(FINAL_V2)
    environment = os.environ.copy()
    subprocess.run(
        ["/home/liuyeqiang/anaconda3/envs/GMT/bin/python", "-u", str(FINAL_V2)],
        cwd=V2ROOT,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
