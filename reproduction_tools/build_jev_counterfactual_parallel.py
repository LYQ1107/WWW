#!/usr/bin/env python3
"""Build counterfactual JEV data with independent per-video workers.

The underlying labeler is unchanged.  This wrapper only partitions a complete
causal trace by video, invokes the existing builder in isolated subprocesses,
and concatenates the resulting JSONL in deterministic video-id order.  GT is
still read only by the offline labeler; no worker receives future labels as
model features.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/liuyeqiang/anaconda3/envs/GMT/bin/python"
BUILDER = ROOT / "reproduction_tools/build_jev_counterfactual_dataset.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_trace(source: Path, directory: Path) -> List[Tuple[int, Path]]:
    handles: Dict[int, object] = {}
    paths: Dict[int, Path] = {}
    try:
        with source.open(encoding="utf-8") as reader:
            for line_number, line in enumerate(reader, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    video_id = int(event["context"]["video_id"])
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(f"invalid trace event at line {line_number}") from exc
                if video_id not in handles:
                    path = directory / f"trace_video_{video_id:08d}.jsonl"
                    paths[video_id] = path
                    handles[video_id] = path.open("w", encoding="utf-8")
                handles[video_id].write(line if line.endswith("\n") else line + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    return sorted(paths.items())


def run_workers(
    parts: List[Tuple[int, Path]],
    temporary: Path,
    annotations: Path,
    checkpoint: Path,
    horizon: int,
    workers: int,
) -> Dict[int, Path]:
    outputs: Dict[int, Path] = {}
    pending = list(parts)
    active = []
    environment = os.environ.copy()
    environment["PYTHONPATH"] = ":".join(
        [
            str(ROOT),
            str(ROOT / "third_party/CenterNet2"),
            str(ROOT / "reproduction_tools"),
            environment.get("PYTHONPATH", ""),
        ]
    )

    def start(video_id: int, trace: Path):
        output = temporary / f"counterfactual_video_{video_id:08d}.jsonl"
        log_path = temporary / f"worker_{video_id:08d}.log"
        log = log_path.open("w", encoding="utf-8")
        command = [
            PYTHON,
            "-u",
            str(BUILDER),
            "--trace",
            str(trace),
            "--annotations",
            str(annotations),
            "--output",
            str(output),
            "--gmt-checkpoint",
            str(checkpoint),
            "--horizon",
            str(horizon),
        ]
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        return video_id, process, log, log_path, output

    def wait_one():
        video_id, process, log, log_path, output = active.pop(0)
        code = process.wait()
        log.close()
        if code:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"counterfactual worker {video_id} failed ({code}):\n{tail}")
        manifest = Path(str(output) + ".manifest.json")
        if not output.is_file() or not manifest.is_file():
            raise RuntimeError(f"counterfactual worker {video_id} produced incomplete output")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if payload.get("status") != "PASS":
            raise RuntimeError(f"counterfactual worker {video_id} did not PASS")
        outputs[video_id] = output

    while pending or active:
        while pending and len(active) < workers:
            active.append(start(*pending.pop(0)))
        wait_one()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gmt-checkpoint", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.horizon < 1 or args.workers < 1:
        raise ValueError("horizon and workers must be positive")

    source = args.trace.resolve()
    annotations = args.annotations.resolve()
    checkpoint = args.gmt_checkpoint.resolve()
    output = args.output.resolve()
    manifest_path = Path(str(output) + ".manifest.json")
    if output.exists() or manifest_path.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    if not source.is_file() or not annotations.is_file() or not checkpoint.is_file():
        raise FileNotFoundError("trace, annotations, and checkpoint must exist")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=output.stem + ".parts.", dir=output.parent) as directory_name:
        temporary = Path(directory_name)
        parts = split_trace(source, temporary)
        if not parts:
            raise ValueError("trace contains no video events")
        outputs = run_workers(
            parts,
            temporary,
            annotations,
            checkpoint,
            args.horizon,
            args.workers,
        )
        stats: Dict[str, int] = {}
        records = 0
        part_records = []
        partial_output = output.with_suffix(output.suffix + ".tmp")
        with partial_output.open("w", encoding="utf-8") as merged:
            for video_id, _ in parts:
                part = outputs[video_id]
                part_manifest = json.loads(
                    Path(str(part) + ".manifest.json").read_text(encoding="utf-8")
                )
                part_count = int(part_manifest.get("records", 0))
                records += part_count
                for question, count in part_manifest.get("records_by_question", {}).items():
                    stats[question] = stats.get(question, 0) + int(count)
                part_records.append({"video_id": video_id, "records": part_count})
                with part.open(encoding="utf-8") as reader:
                    shutil.copyfileobj(reader, merged)
        partial_output.replace(output)

    report = {
        "status": "PASS",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "trace": str(source),
        "trace_sha256": sha256(source),
        "annotations": str(annotations),
        "annotations_sha256": sha256(annotations),
        "gmt_checkpoint": str(checkpoint),
        "gmt_checkpoint_sha256": sha256(checkpoint),
        "horizon": args.horizon,
        "records": records,
        "records_by_question": stats,
        "uses_future_gt": True,
        "online_model_features": "trace state_feature_vector only",
        "labeling_note": "offline frozen-evidence rollout over deep-copied mutable GMT tracker containers; GT is used only for branch utility",
        "counterfactual_engine": "frozen_evidence_mutable_gmt_state_v1",
        "parallel_workers": args.workers,
        "video_parts": part_records,
    }
    manifest_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
