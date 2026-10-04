#!/usr/bin/env python3
"""Resumable post-Stage2 GMT/JEV research pipeline.

The supervisor is intentionally conservative: every artifact is written to a
unique path, completed artifacts are validated by their own manifest, and an
incomplete artifact is renamed rather than deleted.  It therefore can be
restarted by the research monitor after a terminal disconnect or process
failure without silently mixing checkpoints, traces, or evaluation inputs.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from typing import Iterable, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/liuyeqiang/anaconda3/envs/GMT/bin/python"
DATASET = Path("/data/DATASETS/TRACKING/JDE/VisionTrack")
STAGE2 = ROOT / "outputs/stage2_single_gpu/model_20000.pth"
# Formal inference uses the already verified isolated fix while canonical
# Stage2 owns the checkout.  The identical one-line fix is applied to the
# main source only after training exits and is regression-tested.
FORMAL_SOURCE_OVERLAY = ROOT / "outputs/research_proxy/model_4500/isolated_source"
PIPE = ROOT / "outputs/research_pipeline"
MARKERS = PIPE / "markers"
LOG_PATH = PIPE / "pipeline.log"
POLL_SECONDS = 30
MODELS = (
    "fixed_threshold",
    "jev",
    "fixed_slot_mlp",
    "independent_mlp",
    "shared_heads",
    "logistic",
    "global_threshold",
    "state_threshold",
)
POLICIES = ("off",) + MODELS + ("oracle",)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Pipeline:
    def __init__(self) -> None:
        PIPE.mkdir(parents=True, exist_ok=True)
        MARKERS.mkdir(parents=True, exist_ok=True)
        self.log = LOG_PATH.open("a", encoding="utf-8", buffering=1)
        self.env = os.environ.copy()
        self.env.update(
            {
                "PYTHONPATH": ":".join(
                    [
                        str(ROOT),
                        str(ROOT / "third_party/CenterNet2"),
                        str(ROOT / "reproduction_tools"),
                        self.env.get("PYTHONPATH", ""),
                    ]
                ),
                "CUDA_VISIBLE_DEVICES": "0",
                "OMP_NUM_THREADS": "1",
                "GMT_DISTRIBUTED_BACKEND": "gloo",
                "GMT_CPU_COLLECTIVES": "1",
                "GMT_CHECKPOINT_BACKBONE": "1",
                "GMT_TRAIN_PROGRESS": "1",
            }
        )

    def close(self) -> None:
        self.log.close()

    def write_status(self, phase: str, status: str) -> None:
        payload = {
            "updated_utc": utc_now(),
            "phase": phase,
            "status": status,
            "stage2_checkpoint": str(STAGE2),
            "stage2_checkpoint_sha256": sha256(STAGE2) if STAGE2.is_file() else None,
        }
        temporary = PIPE / "pipeline_status.tmp"
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        temporary.replace(PIPE / "pipeline_status.json")

    def ensure_final_stage2_checkpoint(self) -> None:
        """Refuse to start formal inference from an unverified final file."""
        validation = ROOT / "outputs/stage2_single_gpu/validations/model_20000.json"
        valid = False
        if validation.is_file():
            try:
                payload = json.loads(validation.read_text(encoding="utf-8"))
                valid = (
                    payload.get("status") == "PASS"
                    and payload.get("iteration") == 20000
                    and payload.get("scheduler_last_epoch") == 20000
                    and payload.get("global_iteration") == 20000
                    and payload.get("checkpoint_reload") == "PASS"
                    and payload.get("optimizer_state") == "PRESENT"
                    and payload.get("model_finiteness") == "PASS"
                )
            except (OSError, ValueError):
                valid = False
        if not valid:
            self.write_status("stage2_final_checkpoint_validation", "RUNNING")
            self.command(
                [
                    PYTHON,
                    "-u",
                    str(ROOT / "reproduction_tools/validate_training_checkpoint.py"),
                    "--checkpoint",
                    str(STAGE2),
                    "--expected-iteration",
                    "20000",
                    "--expected-scheduler-iteration",
                    "20000",
                    "--output",
                    str(validation),
                ]
            )
            try:
                payload = json.loads(validation.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise RuntimeError("final Stage2 checkpoint validation output is unreadable") from exc
            if payload.get("status") != "PASS":
                raise RuntimeError("final Stage2 checkpoint validation did not PASS")
        self.write_status("stage2_final_checkpoint_validation", "COMPLETE")

    def command(self, args: Sequence[str]) -> None:
        self.log.write("$ " + shlex.join(str(item) for item in args) + "\n")
        self.log.flush()
        process = subprocess.Popen(
            [str(item) for item in args],
            cwd=ROOT,
            env=self.env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"command failed with exit code {return_code}: {shlex.join(map(str, args))}")

    def start_command(
        self,
        args: Sequence[str],
        *,
        cuda_visible_devices: str,
        log_name: str,
    ):
        """Start an independent GPU job on an explicitly selected card.

        The physical GPU is passed both through the environment and the
        inference wrapper's ``--gpu`` option.  The wrapper propagates that
        selection to its ``test_net.py`` child.  Separate logs keep concurrent
        jobs from interleaving their output in the resumable pipeline log.
        """
        self.log.write(
            "$ [gpu="
            + cuda_visible_devices
            + "] "
            + shlex.join(str(item) for item in args)
            + "\n"
        )
        self.log.flush()
        job_log = (PIPE / f"{log_name}.log").open("a", encoding="utf-8", buffering=1)
        env = self.env.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        if not cuda_visible_devices:
            # Eight independent policy trainers should share the host rather
            # than each opening a full BLAS thread pool.
            env["MKL_NUM_THREADS"] = "1"
            env["OPENBLAS_NUM_THREADS"] = "1"
            env["NUMEXPR_NUM_THREADS"] = "1"
        process = subprocess.Popen(
            [str(item) for item in args],
            cwd=ROOT,
            env=env,
            stdout=job_log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
        )
        return process, job_log

    @staticmethod
    def status(path: Path, expected: str) -> bool:
        if not path.is_file():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        return payload.get("status") == expected

    @staticmethod
    def active_for(target: Path) -> bool:
        """Detect an orphaned inference child before archiving its output."""
        try:
            result = subprocess.run(
                ["ps", "-eo", "args="],
                check=True,
                capture_output=True,
                text=True,
            )
        except OSError:
            return False
        needle = str(target)
        return any("test_net.py" in line and needle in line for line in result.stdout.splitlines())

    @staticmethod
    def archive(path: Path) -> None:
        if not path.exists() and not path.is_symlink():
            return
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        destination = path.with_name(path.name + ".incomplete." + stamp)
        counter = 1
        while destination.exists():
            destination = path.with_name(path.name + f".incomplete.{stamp}.{counter}")
            counter += 1
        path.rename(destination)

    def archive_incomplete(self, target: Path, *, trace: Optional[Path] = None) -> None:
        if self.active_for(target):
            self.log.write(f"[recover] active inference child found for {target}; waiting\n")
            self.log.flush()
            while self.active_for(target):
                time.sleep(POLL_SECONDS)
        self.archive(target)
        if trace is not None:
            self.archive(trace)

    def marker(self, name: str) -> Path:
        return MARKERS / f"{name}.done"

    def is_done(self, name: str) -> bool:
        return self.marker(name).is_file()

    def mark_done(self, name: str) -> None:
        self.marker(name).write_text(utc_now() + "\n", encoding="utf-8")

    @staticmethod
    def inference_complete(
        output: Path, dataset_name: str, trace: Optional[Path]
    ) -> bool:
        result = output / f"inference_{dataset_name}/coco_instances_results.json"
        manifest = output / "inference_manifest.json"
        return (
            Pipeline.status(manifest, "COMPLETE")
            and result.is_file()
            and (trace is None or trace.is_file())
        )

    @staticmethod
    def inference_args(
        dataset_name: str,
        output: Path,
        trace: Optional[Path],
        *,
        gpu: str,
        mode: Optional[str],
        controller: Optional[Path],
    ) -> list[str]:
        args = [
            PYTHON,
            "-u",
            str(ROOT / "reproduction_tools/run_formal_inference.py"),
            "--checkpoint",
            str(STAGE2),
            "--config",
            str(ROOT / "configs/VISION_test.yaml"),
            "--output",
            str(output),
            "--gpu",
            gpu,
            "--num-gpus",
            "1",
            "--dataset",
            dataset_name,
            "--source-overlay",
            str(FORMAL_SOURCE_OVERLAY),
        ]
        if mode is not None or controller is not None or trace is not None:
            effective_mode = mode or ("jev" if controller is not None else "off")
            args.extend(["--jev-mode", effective_mode])
            if controller is not None:
                args.extend(["--jev-controller", str(controller)])
            if trace is not None:
                args.extend(["--jev-trace", str(trace)])
        return args

    def ensure_inference(
        self,
        name: str,
        dataset_name: str,
        output: Path,
        trace: Optional[Path],
        *,
        mode: str = "off",
        controller: Optional[Path] = None,
    ) -> None:
        complete = self.inference_complete(output, dataset_name, trace)
        if self.is_done(name) and complete:
            return
        if not complete:
            self.archive_incomplete(output, trace=trace)
            self.write_status(name, "RUNNING")
            args = self.inference_args(
                dataset_name,
                output,
                trace,
                gpu="0",
                mode=mode,
                controller=controller,
            )
            self.command(args)
            complete = self.inference_complete(output, dataset_name, trace)
        if not complete:
            raise RuntimeError(f"inference artifact is incomplete: {output}")
        self.mark_done(name)
        self.write_status(name, "COMPLETE")

    def ensure_inference_parallel(self, jobs: Sequence[dict[str, object]]) -> None:
        """Run independent train/test OFF inference jobs concurrently.

        Each job has its own output and trace, so no artifact is shared while
        the processes are running.  The method waits for both children and
        validates both manifests before returning; a partial job remains
        resumable through the normal archive/marker logic.
        """
        pending = []
        for job in jobs:
            name = str(job["name"])
            dataset_name = str(job["dataset_name"])
            output = job["output"]
            trace = job["trace"]
            raw_mode = job.get("mode", "off")
            mode = None if raw_mode is None else str(raw_mode)
            controller = job.get("controller")
            gpu = str(job["gpu"])
            if not isinstance(output, Path) or (
                trace is not None and not isinstance(trace, Path)
            ):
                raise TypeError("parallel inference paths must be pathlib.Path values or None")
            if controller is not None and not isinstance(controller, Path):
                raise TypeError("parallel inference controller must be a pathlib.Path")
            complete = self.inference_complete(output, dataset_name, trace)
            if self.is_done(name) and complete:
                continue
            if complete:
                self.mark_done(name)
                continue
            self.archive_incomplete(output, trace=trace)
            self.write_status(name, "RUNNING")
            args = self.inference_args(
                dataset_name,
                output,
                trace,
                gpu=gpu,
                mode=mode,
                controller=controller,
            )
            process, job_log = self.start_command(
                args,
                cuda_visible_devices=gpu,
                log_name=name,
            )
            pending.append((job, process, job_log))

        failures = []
        for job, process, job_log in pending:
            return_code = process.wait()
            job_log.close()
            if return_code:
                failures.append((str(job["name"]), return_code))
        if failures:
            raise RuntimeError(f"parallel inference jobs failed: {failures}")

        for job in jobs:
            name = str(job["name"])
            dataset_name = str(job["dataset_name"])
            output = job["output"]
            trace = job["trace"]
            if not isinstance(output, Path) or (
                trace is not None and not isinstance(trace, Path)
            ):
                raise TypeError("parallel inference paths must be pathlib.Path values or None")
            if not self.inference_complete(output, dataset_name, trace):
                raise RuntimeError(f"inference artifact is incomplete: {output}")
            self.mark_done(name)
            self.write_status(name, "COMPLETE")

    def start_inference_job(
        self,
        name: str,
        dataset_name: str,
        output: Path,
        trace: Optional[Path],
        *,
        gpu: str,
        mode: Optional[str] = "off",
        controller: Optional[Path] = None,
    ):
        """Prepare and start one resumable inference job without waiting."""
        complete = self.inference_complete(output, dataset_name, trace)
        if self.is_done(name) and complete:
            return None
        if complete:
            self.mark_done(name)
            self.write_status(name, "COMPLETE")
            return None
        self.archive_incomplete(output, trace=trace)
        self.write_status(name, "RUNNING")
        args = self.inference_args(
            dataset_name,
            output,
            trace,
            gpu=gpu,
            mode=mode,
            controller=controller,
        )
        return self.start_command(
            args,
            cuda_visible_devices=gpu,
            log_name=name,
        )

    def finish_inference_job(
        self,
        name: str,
        dataset_name: str,
        output: Path,
        trace: Optional[Path],
        job,
    ) -> None:
        """Wait for a previously started inference job and validate it."""
        if job is not None:
            process, job_log = job
            return_code = process.wait()
            job_log.close()
            if return_code:
                raise RuntimeError(
                    f"inference job failed with exit code {return_code}: {name}"
                )
        if not self.inference_complete(output, dataset_name, trace):
            raise RuntimeError(f"inference artifact is incomplete: {output}")
        self.mark_done(name)
        self.write_status(name, "COMPLETE")

    def ensure_off_equivalence(self, traced_output: Path, native_output: Path) -> None:
        """Require traced OFF to preserve the native GMT prediction stream."""
        report_path = PIPE / "off_equivalence.json"
        if self.is_done("off_equivalence") and self.status(report_path, "PASS"):
            return
        traced_result = traced_output / "inference_VISION_test/coco_instances_results.json"
        native_result = native_output / "inference_VISION_test/coco_instances_results.json"
        traced_manifest = traced_output / "inference_manifest.json"
        native_manifest = native_output / "inference_manifest.json"
        for path in (traced_result, native_result, traced_manifest, native_manifest):
            if not path.is_file():
                raise RuntimeError(f"OFF equivalence input is missing: {path}")
        traced_meta = json.loads(traced_manifest.read_text(encoding="utf-8"))
        native_meta = json.loads(native_manifest.read_text(encoding="utf-8"))
        traced_payload = json.loads(traced_result.read_text(encoding="utf-8"))
        native_payload = json.loads(native_result.read_text(encoding="utf-8"))
        checkpoint_equal = (
            traced_meta.get("checkpoint_sha256") == native_meta.get("checkpoint_sha256")
        )
        config_equal = traced_meta.get("config_sha256") == native_meta.get("config_sha256")
        prediction_equal = traced_payload == native_payload
        report = {
            "status": "PASS" if checkpoint_equal and config_equal and prediction_equal else "FAIL",
            "traced_off_output": str(traced_result),
            "native_off_output": str(native_result),
            "traced_off_sha256": sha256(traced_result),
            "native_off_sha256": sha256(native_result),
            "traced_off_manifest": str(traced_manifest),
            "native_off_manifest": str(native_manifest),
            "checkpoint_sha256_equal": checkpoint_equal,
            "config_sha256_equal": config_equal,
            "prediction_json_equal": prediction_equal,
            "traced_prediction_count": len(traced_payload) if isinstance(traced_payload, list) else None,
            "native_prediction_count": len(native_payload) if isinstance(native_payload, list) else None,
            "traced_mode": traced_meta.get("jev_mode"),
            "native_mode": native_meta.get("jev_mode"),
            "interpretation": "JEV OFF trace path is accepted only when it produces the exact native GMT prediction stream",
        }
        temporary = report_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(report_path)
        if report["status"] != "PASS":
            raise RuntimeError("OFF equivalence gate failed; traced and native predictions differ")
        self.mark_done("off_equivalence")
        self.write_status("off_equivalence", "COMPLETE")

    def ensure_counterfactual(
        self, name: str, trace: Path, annotations: Path, output: Path
    ) -> None:
        manifest = Path(str(output) + ".manifest.json")
        complete = self.status(manifest, "PASS") and output.is_file()
        if self.is_done(name) and complete:
            return
        if not complete:
            self.archive(output)
            self.archive(manifest)
            self.write_status(name, "RUNNING")
            self.command(
                [
                    PYTHON,
                    "-u",
                    str(ROOT / "reproduction_tools/build_jev_counterfactual_parallel.py"),
                    "--trace",
                    str(trace),
                    "--annotations",
                    str(annotations),
                    "--gmt-checkpoint",
                    str(STAGE2),
                    "--output",
                    str(output),
                    "--horizon",
                    "32",
                    "--workers",
                    "4",
                ]
            )
            complete = self.status(manifest, "PASS") and output.is_file()
        if not complete:
            raise RuntimeError(f"counterfactual artifact is incomplete: {output}")
        self.mark_done(name)
        self.write_status(name, "COMPLETE")

    def ensure_policy(self, name: str, model: str, dataset: Path, output: Path) -> None:
        metrics = output / "metrics.json"
        complete = self.status(metrics, "COMPLETE") and (output / "model.pth").is_file()
        if self.is_done(name) and complete:
            return
        if not complete:
            self.archive(output)
            self.write_status(name, "RUNNING")
            epochs = "1" if model == "fixed_threshold" else "20"
            self.command(
                [
                    PYTHON,
                    "-u",
                    str(ROOT / "reproduction_tools/train_jev.py"),
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(output),
                    "--model",
                    model,
                    "--epochs",
                    epochs,
                    "--batch-size",
                    "1024",
                    "--hidden-dim",
                    "128",
                    "--lr",
                    "1e-3",
                    "--seed",
                    "20261003",
                    "--device",
                    "cpu",
                    "--fixed-threshold",
                    "0.2",
                ]
            )
            complete = self.status(metrics, "COMPLETE") and (output / "model.pth").is_file()
        if not complete:
            raise RuntimeError(f"policy artifact is incomplete: {output}")
        self.mark_done(name)
        self.write_status(name, "COMPLETE")

    def ensure_policies_parallel(self, dataset: Path, output_root: Path) -> None:
        """Train independent CPU policies concurrently on isolated outputs."""
        pending = []
        for model in MODELS:
            name = f"policy_{model}"
            output = output_root / model
            metrics = output / "metrics.json"
            complete = self.status(metrics, "COMPLETE") and (output / "model.pth").is_file()
            if self.is_done(name) and complete:
                continue
            if complete:
                self.mark_done(name)
                continue
            self.archive(output)
            self.write_status(name, "RUNNING")
            epochs = "1" if model == "fixed_threshold" else "20"
            args = [
                PYTHON,
                "-u",
                str(ROOT / "reproduction_tools/train_jev.py"),
                "--dataset",
                str(dataset),
                "--output",
                str(output),
                "--model",
                model,
                "--epochs",
                epochs,
                "--batch-size",
                "1024",
                "--hidden-dim",
                "128",
                "--lr",
                "1e-3",
                "--seed",
                "20261003",
                "--device",
                "cpu",
                "--fixed-threshold",
                "0.2",
            ]
            process, job_log = self.start_command(
                args,
                cuda_visible_devices="",
                log_name=name,
            )
            pending.append((name, model, output, process, job_log))

        failures = []
        for name, model, output, process, job_log in pending:
            return_code = process.wait()
            job_log.close()
            if return_code:
                failures.append((name, return_code))
        if failures:
            raise RuntimeError(f"parallel policy jobs failed: {failures}")

        for name, model, output, _, _ in pending:
            metrics = output / "metrics.json"
            if not self.status(metrics, "COMPLETE") or not (output / "model.pth").is_file():
                raise RuntimeError(f"policy artifact is incomplete: {output}")
            self.mark_done(name)
            self.write_status(name, "COMPLETE")

    def ensure_decision_stress(self, dataset: Path, output: Path) -> None:
        """Create real-data same-score/different-state diagnostics."""
        if self.is_done("decision_stress") and self.status(output, "PASS"):
            return
        if output.exists():
            self.archive(output)
        self.write_status("decision_stress", "RUNNING")
        policy_args = [
            f"{model}={PIPE / 'policies' / model}"
            for model in MODELS
        ]
        self.command(
            [
                PYTHON,
                "-u",
                str(ROOT / "reproduction_tools/analyze_jev_decisions.py"),
                "--dataset",
                str(dataset),
                "--output",
                str(output),
                *policy_args,
            ]
        )
        if not self.status(output, "PASS"):
            raise RuntimeError(f"decision stress artifact is incomplete: {output}")
        self.mark_done("decision_stress")
        self.write_status("decision_stress", "COMPLETE")

    def ensure_replay(
        self,
        name: str,
        predictions: Path,
        trace: Path,
        annotations: Path,
        output: Path,
        *,
        controller: Optional[Path] = None,
        oracle_dataset: Optional[Path] = None,
    ) -> None:
        manifest = Path(str(output) + ".manifest.json")
        complete = self.status(manifest, "PASS") and output.is_file()
        if self.is_done(name) and complete:
            return
        if not complete:
            self.archive(output)
            self.archive(manifest)
            self.write_status(name, "RUNNING")
            args = [
                PYTHON,
                "-u",
                str(ROOT / "reproduction_tools/replay_jev_policy.py"),
                "--predictions",
                str(predictions),
                "--trace",
                str(trace),
                "--annotations",
                str(annotations),
                "--output",
                str(output),
            ]
            if oracle_dataset is not None:
                args.extend(["--oracle-dataset", str(oracle_dataset)])
            elif controller is not None:
                args.extend(["--controller", str(controller)])
            else:
                raise ValueError("replay requires controller or oracle_dataset")
            self.command(args)
            complete = self.status(manifest, "PASS") and output.is_file()
        if not complete:
            raise RuntimeError(f"replay artifact is incomplete: {output}")
        self.mark_done(name)
        self.write_status(name, "COMPLETE")

    def ensure_prepare(self, name: str, split: str, predictions: Path, output: Path) -> None:
        manifest = output / "manifest.json"
        complete = self.status(manifest, "PASS")
        if self.is_done(name) and complete:
            return
        if not complete:
            self.archive(output)
            self.write_status(name, "RUNNING")
            self.command(
                [
                    PYTHON,
                    "-u",
                    str(ROOT / "reproduction_tools/prepare_visiontrack_predictions.py"),
                    "--predictions",
                    str(predictions),
                    "--dataset",
                    str(DATASET),
                    "--split",
                    split,
                    "--output",
                    str(output),
                ]
            )
            complete = self.status(manifest, "PASS")
        if not complete:
            raise RuntimeError(f"prepared artifact is incomplete: {output}")
        self.mark_done(name)
        self.write_status(name, "COMPLETE")

    def ensure_evaluation(self, name: str, prepared: Path, output: Path) -> None:
        metrics = output / "metrics.json"
        complete = self.status(metrics, "PASS")
        if self.is_done(name) and complete:
            return
        if not complete:
            self.archive(output)
            self.write_status(name, "RUNNING")
            self.command(
                [
                    PYTHON,
                    "-u",
                    str(ROOT / "reproduction_tools/evaluate_visiontrack.py"),
                    "--prepared",
                    str(prepared),
                    "--output",
                    str(output),
                    "--allow-duplicate-gt",
                ]
            )
            complete = self.status(metrics, "PASS")
        if not complete:
            raise RuntimeError(f"evaluation artifact is incomplete: {output}")
        self.mark_done(name)
        self.write_status(name, "COMPLETE")

    def ensure_crossview(self, name: str, prepared: Path, output: Path) -> None:
        metrics = output / "metrics.json"
        complete = self.status(metrics, "PASS")
        if self.is_done(name) and complete:
            return
        if not complete:
            self.archive(output)
            self.write_status(name, "RUNNING")
            self.command(
                [
                    PYTHON,
                    "-u",
                    str(ROOT / "reproduction_tools/evaluate_crossview_visiontrack.py"),
                    "--manifest",
                    str(prepared / "manifest.json"),
                    "--output",
                    str(output),
                    "--allow-duplicate-gt",
                ]
            )
            complete = self.status(metrics, "PASS")
        if not complete:
            raise RuntimeError(f"cross-view artifact is incomplete: {output}")
        self.mark_done(name)
        self.write_status(name, "COMPLETE")


def run() -> None:
    pipeline = Pipeline()
    try:
        if not STAGE2.is_file():
            pipeline.write_status("waiting_for_stage2", "BLOCKED")
            raise RuntimeError(f"Stage2 final checkpoint is unavailable: {STAGE2}")

        pipeline.ensure_final_stage2_checkpoint()

        pipeline.log.write(f"[pipeline] starting with {STAGE2}\n")
        pipeline.log.flush()
        (PIPE / "traces").mkdir(parents=True, exist_ok=True)
        (PIPE / "counterfactual").mkdir(parents=True, exist_ok=True)
        (PIPE / "policies").mkdir(parents=True, exist_ok=True)
        (PIPE / "replays").mkdir(parents=True, exist_ok=True)
        (PIPE / "prepared").mkdir(parents=True, exist_ok=True)
        (PIPE / "evaluations").mkdir(parents=True, exist_ok=True)
        (PIPE / "crossview").mkdir(parents=True, exist_ok=True)

        train_inference = PIPE / "inference_train_off"
        test_inference = PIPE / "inference_test_off"
        native_test_inference = PIPE / "inference_test_native_off"
        train_trace = PIPE / "traces/train_off.jsonl"
        test_trace = PIPE / "traces/test_off.jsonl"
        train_cf = PIPE / "counterfactual/train.jsonl"
        test_cf = PIPE / "counterfactual/test.jsonl"

        # The traced OFF runs use the same frozen Stage2 checkpoint but
        # independent datasets and output paths.  A third GPU runs native GMT
        # test OFF with JEV completely disabled so the OFF equivalence gate is
        # measured rather than assumed.
        pipeline.ensure_inference_parallel(
            [
                {
                    "name": "inference_train_off",
                    "dataset_name": "VISION_train",
                    "output": train_inference,
                    "trace": train_trace,
                    "gpu": "2",
                },
                {
                    "name": "inference_test_off",
                    "dataset_name": "VISION_test",
                    "output": test_inference,
                    "trace": test_trace,
                    "gpu": "3",
                },
                {
                    "name": "inference_test_native_off",
                    "dataset_name": "VISION_test",
                    "output": native_test_inference,
                    "trace": None,
                    "gpu": "5",
                    "mode": None,
                },
            ]
        )
        pipeline.ensure_off_equivalence(test_inference, native_test_inference)
        pipeline.ensure_counterfactual(
            "counterfactual_train",
            train_trace,
            DATASET / "annotations/train.json",
            train_cf,
        )
        pipeline.ensure_counterfactual(
            "counterfactual_test",
            test_trace,
            DATASET / "annotations/test.json",
            test_cf,
        )

        # All policy trainers are independent CPU jobs.  Running them
        # concurrently keeps the frozen-GMT post-processing path short while
        # preserving the same seed, data split, optimizer, and output per
        # policy.
        pipeline.ensure_policies_parallel(train_cf, PIPE / "policies")
        pipeline.ensure_decision_stress(train_cf, PIPE / "decision_stress.json")

        test_predictions = test_inference / "inference_VISION_test/coco_instances_results.json"

        actual_inference = PIPE / "inference_test_jev"
        actual_trace = PIPE / "traces/test_jev.jsonl"
        # Actual online-JEV inference is independent of the offline replay
        # evaluations below.  Start it on another idle physical GPU and let
        # the CPU evaluation chain proceed while it runs.
        actual_inference_job = pipeline.start_inference_job(
            "inference_test_jev_actual",
            "VISION_test",
            actual_inference,
            actual_trace,
            gpu="4",
            mode="jev",
            controller=PIPE / "policies/jev/model.pth",
        )

        for policy in POLICIES:
            prediction = test_predictions
            if policy == "oracle":
                replay = PIPE / "replays/oracle.json"
                pipeline.ensure_replay(
                    "replay_oracle",
                    prediction,
                    test_trace,
                    DATASET / "annotations/test.json",
                    replay,
                    oracle_dataset=test_cf,
                )
                prediction = replay
            elif policy != "off":
                replay = PIPE / f"replays/{policy}.json"
                pipeline.ensure_replay(
                    f"replay_{policy}",
                    prediction,
                    test_trace,
                    DATASET / "annotations/test.json",
                    replay,
                    controller=PIPE / f"policies/{policy}/model.pth",
                )
                prediction = replay

            prepared = PIPE / f"prepared/test_{policy}"
            evaluation = PIPE / f"evaluations/test_{policy}"
            crossview = PIPE / f"crossview/test_{policy}"
            pipeline.ensure_prepare(f"prepare_{policy}", "test", prediction, prepared)
            pipeline.ensure_evaluation(f"evaluate_{policy}", prepared, evaluation)
            pipeline.ensure_crossview(f"crossview_{policy}", prepared, crossview)

        pipeline.finish_inference_job(
            "inference_test_jev_actual",
            "VISION_test",
            actual_inference,
            actual_trace,
            actual_inference_job,
        )
        actual_prepared = PIPE / "prepared/test_actual_jev"
        actual_evaluation = PIPE / "evaluations/test_actual_jev"
        actual_crossview = PIPE / "crossview/test_actual_jev"
        pipeline.ensure_prepare(
            "prepare_actual_jev", "test", actual_inference / "inference_VISION_test/coco_instances_results.json", actual_prepared
        )
        pipeline.ensure_evaluation("evaluate_actual_jev", actual_prepared, actual_evaluation)
        pipeline.ensure_crossview("crossview_actual_jev", actual_prepared, actual_crossview)

        policy_args = [
            "off=" + str(PIPE / "evaluations/test_off"),
            "fixed_threshold=" + str(PIPE / "evaluations/test_fixed_threshold"),
            "jev_replay=" + str(PIPE / "evaluations/test_jev"),
            "fixed_slot_mlp=" + str(PIPE / "evaluations/test_fixed_slot_mlp"),
            "independent_mlp=" + str(PIPE / "evaluations/test_independent_mlp"),
            "shared_heads=" + str(PIPE / "evaluations/test_shared_heads"),
            "logistic=" + str(PIPE / "evaluations/test_logistic"),
            "global_threshold=" + str(PIPE / "evaluations/test_global_threshold"),
            "state_threshold=" + str(PIPE / "evaluations/test_state_threshold"),
            "oracle=" + str(PIPE / "evaluations/test_oracle"),
            "actual_jev=" + str(PIPE / "evaluations/test_actual_jev"),
        ]
        cross_args = [
            "off=" + str(PIPE / "crossview/test_off"),
            "fixed_threshold=" + str(PIPE / "crossview/test_fixed_threshold"),
            "jev_replay=" + str(PIPE / "crossview/test_jev"),
            "fixed_slot_mlp=" + str(PIPE / "crossview/test_fixed_slot_mlp"),
            "independent_mlp=" + str(PIPE / "crossview/test_independent_mlp"),
            "shared_heads=" + str(PIPE / "crossview/test_shared_heads"),
            "logistic=" + str(PIPE / "crossview/test_logistic"),
            "global_threshold=" + str(PIPE / "crossview/test_global_threshold"),
            "state_threshold=" + str(PIPE / "crossview/test_state_threshold"),
            "oracle=" + str(PIPE / "crossview/test_oracle"),
            "actual_jev=" + str(PIPE / "crossview/test_actual_jev"),
        ]
        policy_summary = PIPE / "test_policy_summary.json"
        if not pipeline.status(policy_summary, "PASS"):
            pipeline.archive(policy_summary)
            pipeline.command(
                [
                    PYTHON,
                    "-u",
                    str(ROOT / "reproduction_tools/summarize_policy_suite.py"),
                    "--output",
                    str(policy_summary),
                    *policy_args,
                ]
            )
        cross_summary = PIPE / "test_crossview_summary.json"
        if not pipeline.status(cross_summary, "PASS"):
            pipeline.archive(cross_summary)
            pipeline.command(
                [
                    PYTHON,
                    "-u",
                    str(ROOT / "reproduction_tools/summarize_crossview_suite.py"),
                    "--output",
                    str(cross_summary),
                    *cross_args,
                ]
            )

        oracle_gate = PIPE / "oracle_gate.json"
        if not pipeline.status(oracle_gate, "PASS"):
            pipeline.archive(oracle_gate)
            pipeline.command(
                [
                    PYTHON,
                    "-u",
                    str(ROOT / "reproduction_tools/summarize_oracle_gate.py"),
                    "--output",
                    str(oracle_gate),
                    "--counterfactual",
                    str(test_cf),
                    "--trace",
                    str(actual_trace),
                    "--off",
                    str(PIPE / "evaluations/test_off"),
                    "--oracle",
                    str(PIPE / "evaluations/test_oracle"),
                    "--threshold",
                    str(PIPE / "evaluations/test_fixed_threshold"),
                    "--jev",
                    str(PIPE / "evaluations/test_jev"),
                    "--off-cross",
                    str(PIPE / "crossview/test_off"),
                    "--oracle-cross",
                    str(PIPE / "crossview/test_oracle"),
                    "--threshold-cross",
                    str(PIPE / "crossview/test_fixed_threshold"),
                    "--jev-cross",
                    str(PIPE / "crossview/test_jev"),
                ]
            )

        pipeline.write_status("complete", "COMPLETE")
        complete = {
            "status": "COMPLETE",
            "finished_utc": utc_now(),
            "stage2_checkpoint": str(STAGE2),
            "stage2_checkpoint_sha256": sha256(STAGE2),
            "off_equivalence": str(PIPE / "off_equivalence.json"),
            "decision_stress": str(PIPE / "decision_stress.json"),
            "policy_summary": str(policy_summary),
            "crossview_summary": str(cross_summary),
            "oracle_gate": str(oracle_gate),
            "future_gt_online": False,
            "oracle_policy": "offline only; uses counterfactual test labels and is not an online result",
        }
        (PIPE / "PIPELINE_COMPLETE.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
        pipeline.log.write("[pipeline] COMPLETE\n")
        pipeline.log.flush()
    except Exception as exc:
        pipeline.write_status("failed", "FAILED")
        pipeline.log.write(f"[pipeline] FAILED: {type(exc).__name__}: {exc}\n")
        pipeline.log.flush()
        raise
    finally:
        pipeline.close()


if __name__ == "__main__":
    run()
