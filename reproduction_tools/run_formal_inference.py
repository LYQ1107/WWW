"""Run current-repository GMT inference with an explicit verified checkpoint."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/liuyeqiang/anaconda3/envs/GMT/bin/python"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/VISION_test.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--dataset", choices=("VISION_train", "VISION_test"), default="VISION_test")
    parser.add_argument("--jev-mode", choices=("off", "shadow", "jev"), default=None)
    parser.add_argument("--jev-controller", type=Path, default=None)
    parser.add_argument("--jev-trace", type=Path, default=None)
    parser.add_argument("--source-overlay", type=Path, default=None)
    args = parser.parse_args()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    config = args.config if args.config.is_absolute() else ROOT / args.config
    output = args.output if args.output.is_absolute() else ROOT / args.output
    source_overlay = None
    if args.source_overlay is not None:
        source_overlay = args.source_overlay if args.source_overlay.is_absolute() else ROOT / args.source_overlay
        source_overlay = source_overlay.resolve()
        if not (source_overlay / "gtr").is_dir():
            raise FileNotFoundError(f"source overlay is missing gtr/: {source_overlay}")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not config.is_file():
        raise FileNotFoundError(config)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite existing inference output: {output}")
    output.mkdir(parents=True)
    log = output / "inference.log"
    # A subprocess launched with ``cwd=ROOT`` puts the canonical repository
    # directory ahead of PYTHONPATH, so merely prepending an overlay there
    # does not guarantee that its gtr package wins.  Use the isolated runner
    # when an overlay is requested; it inserts the overlay before executing
    # test_net.py and records the actual source selected for the run.
    if source_overlay is not None:
        runner = source_overlay / "reproduction_tools/run_isolated_test_net.py"
        if not runner.is_file():
            raise FileNotFoundError(f"source overlay runner is missing: {runner}")
        command = [PYTHON, "-u", str(runner)]
    else:
        command = [PYTHON, "-u", "test_net.py"]
    command.extend([
        "--num-gpus",
        str(args.num_gpus),
        "--config-file",
        str(config),
        "MODEL.WEIGHTS",
        str(checkpoint),
        "OUTPUT_DIR",
        str(output),
        "DATASETS.TEST",
        "(" + repr(args.dataset) + ",)",
    ])
    if args.jev_mode is not None or args.jev_controller is not None or args.jev_trace is not None:
        mode = args.jev_mode or ("jev" if args.jev_controller is not None else "off")
        command.extend(["MODEL.JEV.ENABLED", "True", "MODEL.JEV.MODE", mode])
        if args.jev_controller is not None:
            controller = args.jev_controller if args.jev_controller.is_absolute() else ROOT / args.jev_controller
            command.extend(["MODEL.JEV.CONTROLLER_WEIGHTS", str(controller)])
        if args.jev_trace is not None:
            trace = args.jev_trace if args.jev_trace.is_absolute() else ROOT / args.jev_trace
            command.extend(["MODEL.JEV.TRACE_PATH", str(trace)])
    manifest = {
        "status": "RUNNING",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "config": str(config),
        "config_sha256": sha256(config),
        "output": str(output),
        "command": command,
        "gpu": args.gpu,
        "dataset": args.dataset,
        "jev_mode": args.jev_mode,
        "jev_controller": str(args.jev_controller) if args.jev_controller else None,
        "jev_trace": str(args.jev_trace) if args.jev_trace else None,
        "source_overlay": str(source_overlay) if source_overlay else None,
    }
    (output / "inference_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    environment = os.environ.copy()
    if source_overlay is not None:
        environment["PYTHONPATH"] = ":".join(
            [str(source_overlay), str(ROOT), environment.get("PYTHONPATH", "")]
        )
        environment["GMT_GTR_OVERLAY"] = str(source_overlay)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": args.gpu,
            "OMP_NUM_THREADS": "1",
            "GMT_DISTRIBUTED_BACKEND": "gloo",
            "GMT_CPU_COLLECTIVES": "1",
        }
    )
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        manifest["pid"] = process.pid
        (output / "inference_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return_code = process.wait()
    manifest.update(
        {
            "status": "COMPLETE" if return_code == 0 else "FAILED",
            "return_code": return_code,
            "duration_seconds": time.monotonic() - started,
            "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
    )
    (output / "inference_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if return_code:
        raise SystemExit(return_code)
    result = output / ("inference_" + args.dataset) / "coco_instances_results.json"
    if not result.is_file():
        raise RuntimeError(f"inference exited successfully but result is missing: {result}")
    manifest["result"] = str(result)
    manifest["result_sha256"] = sha256(result)
    (output / "inference_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
