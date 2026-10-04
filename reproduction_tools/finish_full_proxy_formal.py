#!/usr/bin/env python3
"""Finish the model-4500 full formal replay after OFF inference exits.

This is a resumable screening pipeline.  It never changes the canonical
training job, never overwrites a shard, and records enough metadata to make
the resulting proxy artifacts distinguishable from final-checkpoint results.
"""

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/liuyeqiang/anaconda3/envs/GMT/bin/python"
DATASET = Path("/data/DATASETS/TRACKING/JDE/VisionTrack")
MODEL = Path("/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_4500.pth")
OUT = Path("/data1/liuyeqiang/WWW/outputs/research_v2/model_4500")
TRACE_ROOT = OUT / "traces"
CACHE_ROOT = OUT
LOG = OUT / "finish_full_proxy_formal.log"
STATUS = OUT / "finish_full_proxy_formal.status.json"


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def write_status(status: str, **extra: object) -> None:
    payload = {"status": status, "updated_utc": now(), **extra}
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(STATUS)


def log(message: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] {message}\n")


def run(args: Sequence[str], *, env: dict[str, str] | None = None, log_name: str | None = None) -> None:
    log_path = LOG if log_name is None else OUT / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    merged_env = os.environ.copy()
    merged_env.update(env or {})
    merged_env["PYTHONPATH"] = ":".join(
        [str(ROOT), str(ROOT / "third_party/CenterNet2"), str(ROOT / "reproduction_tools"), merged_env.get("PYTHONPATH", "")]
    )
    log(f"$ {' '.join(str(item) for item in args)}")
    with log_path.open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            [str(item) for item in args],
            cwd=ROOT,
            env=merged_env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(map(str, args))}")


def active_output(output: Path) -> bool:
    try:
        rows = subprocess.check_output(["ps", "-eo", "args="], text=True).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return False
    return any("run_isolated_test_net.py" in row and str(output) in row for row in rows)


def wait_for_inference(output: Path, dataset_name: str, trace: Path) -> None:
    result = output / f"inference_{dataset_name}/coco_instances_results.json"
    while True:
        if result.is_file() and not active_output(output):
            log(f"inference complete: {output}")
            return
        write_status("WAITING_FOR_OFF_INFERENCE", waiting_for=str(output), trace=str(trace))
        time.sleep(30)


def manifest_pass(path: Path, *, formal: bool = True) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if payload.get("status") != "PASS":
        return False
    if formal and (
        payload.get("association_backend") != "formal_gmt_transformer"
        or payload.get("formal_gmt_association_adapter") is not True
        or int(payload.get("skipped_events", -1)) != 0
    ):
        return False
    return True


def shard_groups(count: int, group_count: int = 4) -> list[list[int]]:
    groups = [[] for _ in range(group_count)]
    for index, video_id in enumerate(range(1, count + 1)):
        groups[index % group_count].append(video_id)
    return groups


def ensure_formal_shards(
    *, split: str, trace: Path, cache: Path, video_count: int, checkpoint: Path
) -> list[Path]:
    annotation = DATASET / "annotations" / f"{split}.json"
    root = OUT / f"formal_full_{split}_v2"
    root.mkdir(parents=True, exist_ok=True)
    devices = ("4", "5", "8", "9")
    pending = []
    outputs: list[Path] = []
    for shard_index, video_ids in enumerate(shard_groups(video_count)):
        output = root / f"shard_{shard_index}.jsonl"
        manifest = Path(str(output) + ".manifest.json")
        outputs.append(output)
        if output.is_file() and manifest_pass(manifest):
            continue
        if output.exists() or manifest.exists():
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            for path in (output, manifest):
                if path.exists():
                    shutil.move(str(path), str(path.with_name(path.name + ".incomplete." + stamp)))
        command = [
            PYTHON,
            "-u",
            str(ROOT / "reproduction_tools/build_jev_counterfactual_v2.py"),
            "--trace",
            str(trace),
            "--cache",
            str(cache),
            "--annotations",
            str(annotation),
            "--output",
            str(output),
            "--gmt-checkpoint",
            str(checkpoint),
            "--horizon",
            "8",
            "--association-backend",
            "formal_gmt_transformer",
            "--config-file",
            str(ROOT / "configs/VISION_test.yaml"),
            "--device",
            "cuda:0",
            "--view-num",
            "2",
            "--history-limit",
            "80",
            "--video-ids",
            *[str(value) for value in video_ids],
        ]
        log_path = root / f"shard_{shard_index}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = devices[shard_index]
        env["PYTHONPATH"] = ":".join(
            [str(ROOT), str(ROOT / "third_party/CenterNet2"), str(ROOT / "reproduction_tools"), env.get("PYTHONPATH", "")]
        )
        handle = log_path.open("a", encoding="utf-8")
        handle.write(f"[{now()}] videos={video_ids} gpu={devices[shard_index]}\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        pending.append((shard_index, output, manifest, process, handle))
    write_status("FORMAL_SHARDS_RUNNING", split=split, pending=[item[0] for item in pending])
    failures = []
    for shard_index, output, manifest, process, handle in pending:
        code = process.wait()
        handle.close()
        if code or not manifest_pass(manifest):
            failures.append({"shard": shard_index, "return_code": code, "manifest": str(manifest)})
    if failures:
        raise RuntimeError(f"formal shard failures: {failures}")
    return outputs


def merge_shards(split: str, shards: Iterable[Path]) -> Path:
    output = OUT / f"formal_{split}_full_v2.jsonl"
    manifest = Path(str(output) + ".manifest.json")
    if output.is_file() and manifest_pass(manifest, formal=False):
        return output
    if output.exists() or manifest.exists():
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        for path in (output, manifest):
            if path.exists():
                shutil.move(str(path), str(path.with_name(path.name + ".incomplete." + stamp)))
    run(
        [
            PYTHON,
            "-u",
            str(ROOT / "reproduction_tools/merge_jev_jsonl.py"),
            "--input",
            *[str(path) for path in shards],
            "--output",
            str(output),
        ],
        log_name=f"merge_formal_{split}.log",
    )
    if not manifest_pass(manifest, formal=False):
        raise RuntimeError(f"merged formal dataset failed validation: {output}")
    return output


def main() -> None:
    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)
    train_trace = TRACE_ROOT / "train_off_full_v2.jsonl"
    test_trace = TRACE_ROOT / "test_off_full_v2.jsonl"
    train_output = OUT / "inference_trace_train_full_v2"
    test_output = OUT / "inference_trace_test_full_v2"
    wait_for_inference(train_output, "VISION_train", train_trace)
    wait_for_inference(test_output, "VISION_test", test_trace)
    write_status("ALIGNMENT_VALIDATION")
    for split, trace, cache in (
        ("train", train_trace, OUT / "perception_cache_train"),
        ("test", test_trace, OUT / "perception_cache_test"),
    ):
        report = OUT / f"{split}_off_full_v2_alignment.json"
        if not (report.is_file() and json.loads(report.read_text(encoding="utf-8")).get("status") == "PASS"):
            run(
                [
                    PYTHON,
                    "-u",
                    str(ROOT / "reproduction_tools/validate_jev_trace_alignment.py"),
                    "--trace",
                    str(trace),
                    "--cache",
                    str(cache),
                    "--output",
                    str(report),
                ],
                log_name=f"validate_alignment_{split}.log",
            )
    train_shards = ensure_formal_shards(
        split="train",
        trace=train_trace,
        cache=OUT / "perception_cache_train",
        video_count=24,
        checkpoint=MODEL,
    )
    test_shards = ensure_formal_shards(
        split="test",
        trace=test_trace,
        cache=OUT / "perception_cache_test",
        video_count=22,
        checkpoint=MODEL,
    )
    train_full = merge_shards("train", train_shards)
    test_full = merge_shards("test", test_shards)
    write_status(
        "COMPLETE",
        train_formal=str(train_full),
        test_formal=str(test_full),
        proxy_checkpoint=str(MODEL),
        final_numbers_allowed=False,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # keep the failure resumable and inspectable
        write_status("FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
