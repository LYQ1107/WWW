#!/usr/bin/env python3
"""Recoverable six-slot scheduler for the WorldObject experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from worldmot.data.visiontrack import VisionTrack


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(value, indent=2, sort_keys=True)); os.replace(tmp, path)


def make_queue(root: Path, split: str, scenes: list[int]) -> Path:
    q = root / f"queue_{split}"; q.mkdir(parents=True, exist_ok=True)
    for p in q.glob("scene_*.pending"): p.unlink()
    for p in q.glob("scene_*.running"): p.unlink()
    for p in q.glob("scene_*.failed"): p.unlink()
    for sid in scenes: (q / f"scene_{int(sid):02d}.pending").touch()
    return q


def run_cache(a, split: str) -> None:
    root = Path(a.root); ds = VisionTrack(a.dataset_root, split=split, load_annotations=False)
    scenes = ds.scene_ids(); counts = {int(s): len(ds.frames(s)) for s in scenes}
    # Longest-processing scenes are offered first; atomic rename in each worker
    # gives dynamic work stealing while keeping one model per GPU process.
    queue = make_queue(root, split, sorted(scenes, key=lambda x: (-counts[int(x)], int(x))))
    output = root / f"cache_{split}"
    raw = Path(a.raw_cache_train if split == "train" else a.raw_cache_test)
    plan = {"schema": "worldobject_six_gpu_plan_v1", "split": split,
            "gpu_ids": list(a.gpu_ids), "scenes": scenes, "frame_counts": counts,
            "queue": str(queue), "output": str(output), "raw_cache": str(raw),
            "created_at": time.time()}
    write_json(root / "diagnostics" / f"{split}_cache_plan.json", plan)
    logs = root / "diagnostics" / "cache_logs"; logs.mkdir(parents=True, exist_ok=True)
    procs = []
    for slot, gpu in enumerate(a.gpu_ids):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
               "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "1",
               "PYTHONDONTWRITEBYTECODE": "1"}
        log = (logs / f"{split}_slot{slot}_gpu{gpu}.log").open("wb")
        cmd = [sys.executable, str(Path(__file__).with_name("cache_worldobject_support.py")),
               "--dataset-root", a.dataset_root, "--split", split, "--output", str(output),
               "--raw-cache", str(raw), "--scene-queue", str(queue), "--device", "cuda:0"]
        procs.append((subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    statuses = []
    for proc, log in procs:
        rc = proc.wait(); log.close(); statuses.append(int(rc))
    write_json(root / "diagnostics" / f"{split}_cache_status.json", {"return_codes": statuses, "complete": all(x == 0 for x in statuses)})
    if not all(x == 0 for x in statuses): raise SystemExit(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu-ids", nargs="+", required=True, help="physical IDs; CUDA_VISIBLE_DEVICES is set per worker")
    p.add_argument("--phase", choices=["cache_train", "experiments_train", "cache_test", "test", "evaluate", "all"], required=True)
    p.add_argument("--resume", action="store_true"); p.add_argument("--root", required=True)
    p.add_argument("--dataset-root", default="/data/DATASETS/TRACKING/JDE/VisionTrack")
    p.add_argument("--raw-cache-train", default="/data1/liuyeqiang/nas/A100/WorldMOT/gmt_detector_comparison_20261023/raw_train_v1")
    p.add_argument("--raw-cache-test", default="/data1/liuyeqiang/nas/A100/WorldMOT/gmt_detector_comparison_20261023/raw_test_v1")
    a = p.parse_args(); a.gpu_ids = [str(x) for x in a.gpu_ids]; Path(a.root).mkdir(parents=True, exist_ok=True)
    phases = [a.phase] if a.phase != "all" else ["cache_train", "experiments_train", "cache_test", "test", "evaluate"]
    for phase in phases:
        if phase == "cache_train": run_cache(a, "train")
        elif phase == "cache_test": run_cache(a, "test")
        else:
            # Later phases are intentionally explicit entry points.  They do
            # not silently claim that training/evaluation happened when no
            # registered runner has produced a completion manifest.
            write_json(Path(a.root) / "diagnostics" / f"{phase}_status.json",
                       {"status": "PENDING_IMPLEMENTED_ROUTE", "phase": phase,
                        "message": "Use the phase-specific W0-W5 runners after shared cache verification."})


if __name__ == "__main__": main()
