#!/usr/bin/env python3
"""Run the reviewer-proof JEV pipeline from the canonical Stage2 checkpoint.

The proxy worktree is used only as an isolated implementation/runtime source;
all final artifacts are keyed to the canonical ``model_20000.pth``.  Every
phase is resumable, refuses to overwrite evidence, and writes a small marker
only after its output has passed the corresponding validator.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = Path("/data1/liuyeqiang/WWW")
PYTHON = "/home/liuyeqiang/anaconda3/envs/GMT/bin/python"
DATASET = Path("/data/DATASETS/TRACKING/JDE/VisionTrack")
CHECKPOINT = CANONICAL / "outputs/stage2_single_gpu/model_20000.pth"
OUT = CANONICAL / "outputs/research_final_v2"
PIPE = CANONICAL / "outputs/research_pipeline"
LOG = OUT / "final_v2_pipeline.log"
POLL_SECONDS = 30
GPU_GROUPS = ("4", "5", "8", "9")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, Mapping) else None


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def archive(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = path.with_name(path.name + ".incomplete." + stamp)
    suffix = 1
    while destination.exists():
        destination = path.with_name(path.name + f".incomplete.{stamp}.{suffix}")
        suffix += 1
    path.rename(destination)


def base_env() -> dict[str, str]:
    environment = os.environ.copy()
    paths = [
        str(ROOT),
        str(ROOT / "third_party/CenterNet2"),
        str(ROOT / "reproduction_tools"),
        str(CANONICAL),
        str(CANONICAL / "third_party/CenterNet2"),
        str(CANONICAL / "reproduction_tools"),
    ]
    old = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = ":".join(paths + ([old] if old else []))
    environment["OMP_NUM_THREADS"] = "1"
    environment["GMT_DISTRIBUTED_BACKEND"] = "gloo"
    environment["GMT_CPU_COLLECTIVES"] = "1"
    environment["GMT_CHECKPOINT_BACKBONE"] = "1"
    environment["GMT_TRAIN_PROGRESS"] = "1"
    return environment


class FinalPipeline:
    def __init__(self) -> None:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "logs").mkdir(exist_ok=True)
        PIPE.mkdir(parents=True, exist_ok=True)
        self.handle = LOG.open("a", encoding="utf-8", buffering=1)

    def close(self) -> None:
        self.handle.close()

    def status(self, phase: str, state: str, **extra: Any) -> None:
        payload = {
            "updated_utc": now(),
            "phase": phase,
            "status": state,
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": sha256(CHECKPOINT) if CHECKPOINT.is_file() else None,
            **extra,
        }
        write_json(PIPE / "pipeline_status.json", payload)

    def marker(self, name: str) -> Path:
        return OUT / "markers" / f"{name}.done"

    def done(self, name: str) -> bool:
        return self.marker(name).is_file()

    def mark(self, name: str, **extra: Any) -> None:
        payload = {"status": "PASS", "created_utc": now(), **extra}
        write_json(self.marker(name), payload)

    def command(
        self,
        args: Sequence[str | Path],
        *,
        cwd: Path = ROOT,
        env: Mapping[str, str] | None = None,
        log_name: str,
    ) -> None:
        command = [str(item) for item in args]
        self.handle.write("$ " + " ".join(command) + "\n")
        self.handle.flush()
        log_path = OUT / "logs" / f"{log_name}.log"
        merged = base_env()
        merged.update(env or {})
        with log_path.open("a", encoding="utf-8") as log_handle:
            result = subprocess.run(
                command,
                cwd=cwd,
                env=merged,
                stdin=subprocess.DEVNULL,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode:
            raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}")

    def start(
        self,
        args: Sequence[str | Path],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log_name: str,
    ):
        command = [str(item) for item in args]
        self.handle.write("$ [async] " + " ".join(command) + "\n")
        self.handle.flush()
        log_path = OUT / "logs" / f"{log_name}.log"
        merged = base_env()
        merged.update(env)
        handle = log_path.open("a", encoding="utf-8", buffering=1)
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=merged,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        return process, handle

    @staticmethod
    def inference_complete(output: Path, dataset_name: str) -> bool:
        manifest = read_json(output / "inference_manifest.json")
        result = output / f"inference_{dataset_name}" / "coco_instances_results.json"
        return bool(
            manifest
            and manifest.get("status") == "COMPLETE"
            and manifest.get("checkpoint_sha256") == sha256(CHECKPOINT)
            and result.is_file()
        )

    def ensure_checkpoint(self) -> None:
        if not CHECKPOINT.is_file():
            self.status("stage2_checkpoint", "WAITING")
            raise RuntimeError(f"canonical final checkpoint is not present: {CHECKPOINT}")
        validation = CANONICAL / "outputs/stage2_single_gpu/validations/model_20000.json"
        payload = read_json(validation)
        valid = bool(
            payload
            and payload.get("status") == "PASS"
            and payload.get("iteration") == 20000
            and payload.get("scheduler_last_epoch") == 20000
            and payload.get("global_iteration") == 20000
            and payload.get("checkpoint_reload") == "PASS"
            and payload.get("model_finiteness") == "PASS"
        )
        if not valid:
            if validation.exists():
                archive(validation)
            self.status("stage2_checkpoint", "VALIDATING")
            self.command(
                [
                    PYTHON,
                    "-u",
                    ROOT / "reproduction_tools/validate_training_checkpoint.py",
                    "--checkpoint",
                    CHECKPOINT,
                    "--expected-iteration",
                    "20000",
                    "--expected-scheduler-iteration",
                    "20000",
                    "--output",
                    validation,
                ],
                log_name="validate_final_checkpoint",
                cwd=ROOT,
            )
            payload = read_json(validation)
            if not payload or payload.get("status") != "PASS":
                raise RuntimeError("final Stage2 checkpoint validation did not PASS")
        self.status("stage2_checkpoint", "PASS", validation=str(validation))
        self.mark("stage2_checkpoint", validation=str(validation), sha256=sha256(CHECKPOINT))

    def _inference_args(
        self,
        *,
        dataset: str,
        output: Path,
        gpu: str,
        trace: Path | None,
    ) -> list[str | Path]:
        args: list[str | Path] = [
            PYTHON,
            "-u",
            CANONICAL / "reproduction_tools/run_formal_inference.py",
            "--checkpoint",
            CHECKPOINT,
            "--config",
            ROOT / "configs/VISION_test.yaml",
            "--output",
            output,
            "--gpu",
            gpu,
            "--num-gpus",
            "1",
            "--dataset",
            dataset,
            "--jev-mode",
            "off",
            "--source-overlay",
            ROOT,
        ]
        if trace is not None:
            args.extend(["--jev-trace", trace])
        return args

    def ensure_off_inference(self) -> Mapping[str, Path]:
        root = OUT / "off"
        root.mkdir(exist_ok=True)
        jobs = [
            {
                "name": "train_off",
                "dataset": "VISION_train",
                "gpu": "2",
                "output": root / "inference_train",
                "trace": root / "traces/train_off.jsonl",
                "cache": root / "perception_cache_train",
            },
            {
                "name": "test_off",
                "dataset": "VISION_test",
                "gpu": "3",
                "output": root / "inference_test",
                "trace": root / "traces/test_off.jsonl",
                "cache": root / "perception_cache_test",
            },
            {
                "name": "test_native_off",
                "dataset": "VISION_test",
                "gpu": "6",
                "output": root / "inference_native_test",
                "trace": None,
                "cache": None,
            },
        ]
        pending = []
        for job in jobs:
            if self.inference_complete(job["output"], job["dataset"]):
                continue
            for path in (job["output"], job["trace"], job["cache"]):
                if isinstance(path, Path):
                    archive(path)
            environment = {"CUDA_VISIBLE_DEVICES": str(job["gpu"])}
            if job["cache"] is not None:
                environment["JEV_PERCEPTION_CACHE_PATH"] = str(job["cache"])
            process, handle = self.start(
                self._inference_args(
                    dataset=str(job["dataset"]),
                    output=job["output"],
                    gpu=str(job["gpu"]),
                    trace=job["trace"],
                ),
                cwd=CANONICAL,
                env=environment,
                log_name=str(job["name"]),
            )
            pending.append((job, process, handle))
        self.status("final_off_inference", "RUNNING", jobs=len(jobs))
        failures = []
        for job, process, handle in pending:
            code = process.wait()
            handle.close()
            if code:
                failures.append({"name": job["name"], "return_code": code})
        if failures:
            raise RuntimeError(f"final OFF inference failures: {failures}")
        for job in jobs:
            if not self.inference_complete(job["output"], job["dataset"]):
                raise RuntimeError(f"inference artifact incomplete: {job['output']}")
        validation_reports = {}
        for job in jobs[:2]:
            report = OUT / "off" / f"{job['name']}_cache_validation.json"
            if not (read_json(report) or {}).get("status") == "PASS":
                if report.exists():
                    archive(report)
                self.command(
                    [
                        PYTHON,
                        "-u",
                        ROOT / "reproduction_tools/validate_jev_perception_cache.py",
                        "--cache",
                        job["cache"],
                        "--annotations",
                        DATASET / "annotations" / ("train.json" if job["dataset"] == "VISION_train" else "test.json"),
                        "--output",
                        report,
                        "--require-complete",
                    ],
                    log_name=f"validate_{job['name']}_cache",
                )
            validation_reports[job["name"]] = report
            if (read_json(report) or {}).get("status") != "PASS":
                raise RuntimeError(f"cache validation failed: {report}")
        for job in jobs[:2]:
            report = OUT / "off" / f"{job['name']}_alignment.json"
            if not (read_json(report) or {}).get("status") == "PASS":
                if report.exists():
                    archive(report)
                self.command(
                    [
                        PYTHON,
                        "-u",
                        ROOT / "reproduction_tools/validate_jev_trace_alignment.py",
                        "--trace",
                        job["trace"],
                        "--cache",
                        job["cache"],
                        "--output",
                        report,
                    ],
                    log_name=f"validate_{job['name']}_alignment",
                )
            if (read_json(report) or {}).get("status") != "PASS":
                raise RuntimeError(f"trace/cache alignment failed: {report}")
        self.ensure_off_equivalence(jobs[1]["output"], jobs[2]["output"])
        self.status("final_off_inference", "PASS", cache_validations={k: str(v) for k, v in validation_reports.items()})
        self.mark("final_off_inference")
        return {
            "train_trace": jobs[0]["trace"],
            "test_trace": jobs[1]["trace"],
            "train_cache": jobs[0]["cache"],
            "test_cache": jobs[1]["cache"],
            "test_predictions": jobs[1]["output"] / "inference_VISION_test/coco_instances_results.json",
            "native_predictions": jobs[2]["output"] / "inference_VISION_test/coco_instances_results.json",
        }

    def ensure_off_equivalence(self, traced_output: Path, native_output: Path) -> None:
        report_path = OUT / "off" / "off_equivalence.json"
        if (read_json(report_path) or {}).get("status") == "PASS":
            return
        traced_manifest = read_json(traced_output / "inference_manifest.json") or {}
        native_manifest = read_json(native_output / "inference_manifest.json") or {}
        traced = json.loads((traced_output / "inference_VISION_test/coco_instances_results.json").read_text())
        native = json.loads((native_output / "inference_VISION_test/coco_instances_results.json").read_text())
        report = {
            "status": "PASS"
            if traced == native
            and traced_manifest.get("checkpoint_sha256") == native_manifest.get("checkpoint_sha256")
            and traced_manifest.get("config_sha256") == native_manifest.get("config_sha256")
            else "FAIL",
            "prediction_json_equal": traced == native,
            "checkpoint_sha256_equal": traced_manifest.get("checkpoint_sha256") == native_manifest.get("checkpoint_sha256"),
            "config_sha256_equal": traced_manifest.get("config_sha256") == native_manifest.get("config_sha256"),
            "traced_sha256": sha256(traced_output / "inference_VISION_test/coco_instances_results.json"),
            "native_sha256": sha256(native_output / "inference_VISION_test/coco_instances_results.json"),
        }
        write_json(report_path, report)
        if report["status"] != "PASS":
            raise RuntimeError("final OFF equivalence gate failed")

    @staticmethod
    def annotation_videos(path: Path) -> list[int]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return sorted(int(video["id"]) for video in payload["videos"])

    def formal_manifest_pass(self, path: Path, *, formal: bool = True) -> bool:
        payload = read_json(path)
        if not payload or payload.get("status") != "PASS":
            return False
        if formal and (
            payload.get("association_backend") != "formal_gmt_transformer"
            or payload.get("formal_gmt_association_adapter") is not True
            or int(payload.get("skipped_events", -1)) != 0
        ):
            return False
        return True

    def ensure_formal_split(self, split: str, trace: Path, cache: Path) -> Path:
        annotation = DATASET / "annotations" / f"{split}.json"
        split_root = OUT / "formal" / split
        split_root.mkdir(parents=True, exist_ok=True)
        merged = split_root / f"formal_{split}_full_v2.jsonl"
        merged_manifest = Path(str(merged) + ".manifest.json")
        shard_paths = [split_root / f"shard_{index}.jsonl" for index in range(len(GPU_GROUPS))]
        shard_manifests = [Path(str(path) + ".manifest.json") for path in shard_paths]
        if (
            self.formal_manifest_pass(merged_manifest, formal=False)
            and merged.is_file()
            and all(self.formal_manifest_pass(path) for path in shard_manifests)
        ):
            return merged
        video_ids = self.annotation_videos(annotation)
        groups = [[] for _ in GPU_GROUPS]
        for index, video_id in enumerate(video_ids):
            groups[index % len(groups)].append(video_id)
        pending = []
        for index, video_group in enumerate(groups):
            output = split_root / f"shard_{index}.jsonl"
            manifest = Path(str(output) + ".manifest.json")
            if self.formal_manifest_pass(manifest) and output.is_file():
                continue
            archive(output)
            archive(manifest)
            command = [
                PYTHON,
                "-u",
                ROOT / "reproduction_tools/build_jev_counterfactual_v2.py",
                "--trace",
                trace,
                "--cache",
                cache,
                "--annotations",
                annotation,
                "--output",
                output,
                "--gmt-checkpoint",
                CHECKPOINT,
                "--horizon",
                "8",
                "--association-backend",
                "formal_gmt_transformer",
                "--config-file",
                ROOT / "configs/VISION_test.yaml",
                "--device",
                "cuda:0",
                "--view-num",
                "2",
                "--history-limit",
                "80",
                "--video-ids",
                *[str(value) for value in video_group],
            ]
            process, handle = self.start(
                command,
                cwd=ROOT,
                env={"CUDA_VISIBLE_DEVICES": GPU_GROUPS[index]},
                log_name=f"formal_{split}_shard_{index}",
            )
            pending.append((index, process, handle, output, manifest))
        failures = []
        for index, process, handle, output, manifest in pending:
            code = process.wait()
            handle.close()
            if code or not self.formal_manifest_pass(manifest):
                failures.append({"shard": index, "code": code, "manifest": str(manifest)})
        if failures:
            raise RuntimeError(f"formal {split} shard failures: {failures}")
        shards = [split_root / f"shard_{index}.jsonl" for index in range(len(groups))]
        if not all(self.formal_manifest_pass(Path(str(path) + ".manifest.json")) for path in shards):
            raise RuntimeError(f"formal {split} shard validation failed before merge")
        if not (merged.is_file() and self.formal_manifest_pass(merged_manifest, formal=False)):
            archive(merged)
            archive(merged_manifest)
            self.command(
                [PYTHON, "-u", ROOT / "reproduction_tools/merge_jev_jsonl.py", "--input", *shards, "--output", merged],
                log_name=f"merge_formal_{split}",
            )
        if not self.formal_manifest_pass(merged_manifest, formal=False):
            raise RuntimeError(f"merged formal dataset failed validation: {merged}")
        self.mark(f"formal_{split}", dataset=str(merged), manifest=str(merged_manifest))
        return merged

    def ensure_formal_data(self, traces: Mapping[str, Path], caches: Mapping[str, Path]) -> Mapping[str, Path]:
        if self.done("formal_train") and self.done("formal_test"):
            return {
                "train": OUT / "formal/train/formal_train_full_v2.jsonl",
                "test": OUT / "formal/test/formal_test_full_v2.jsonl",
            }
        self.status("formal_counterfactual", "RUNNING")
        train = self.ensure_formal_split("train", traces["train"], caches["train"])
        test = self.ensure_formal_split("test", traces["test"], caches["test"])
        self.status("formal_counterfactual", "PASS")
        return {"train": train, "test": test}

    def ensure_policy_split(self, train: Path, test: Path) -> Path:
        output = OUT / "manifests/jev_policy_split_final.json"
        payload = read_json(output)
        if payload and payload.get("official_test_used_for_search") is False:
            return output
        if output.exists():
            archive(output)
        self.command(
            [
                PYTHON,
                "-u",
                ROOT / "reproduction_tools/create_jev_policy_split.py",
                "--input",
                train,
                "--official-test",
                test,
                "--output",
                output,
                "--seed",
                "20261003",
                "--val-fraction",
                "0.2",
            ],
            log_name="create_final_policy_split",
        )
        if not (read_json(output) or {}).get("official_test_used_for_search") is False:
            raise RuntimeError("final policy split gate failed")
        return output

    def ensure_policies(self, train: Path, split: Path) -> Mapping[str, Path]:
        specs = (
            ("fixed_threshold", 64),
            ("global_threshold", 64),
            ("state_threshold", 64),
            ("question_threshold", 70),
            ("independent_mlp", 32),
            ("shared_heads", 56),
            ("question_conditioned_mlp", 69),
            ("action_conditioned_no_question", 66),
            ("jev", 64),
        )
        root = OUT / "policies"
        root.mkdir(parents=True, exist_ok=True)
        pending = []
        for model, hidden in specs:
            output = root / model
            metrics = output / "metrics.json"
            if (read_json(metrics) or {}).get("status") == "COMPLETE" and (output / "model.pth").is_file():
                continue
            archive(output)
            command = [
                PYTHON,
                "-u",
                ROOT / "reproduction_tools/train_jev.py",
                "--dataset",
                train,
                "--output",
                output,
                "--model",
                model,
                "--epochs",
                "50",
                "--batch-size",
                "128",
                "--hidden-dim",
                str(hidden),
                "--lr",
                "1e-3",
                "--seed",
                "20261003",
                "--device",
                "cpu",
                "--fixed-threshold",
                "0.2",
                "--split-manifest",
                split,
            ]
            environment = {
                "CUDA_VISIBLE_DEVICES": "",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            }
            process, handle = self.start(command, cwd=ROOT, env=environment, log_name=f"policy_{model}")
            pending.append((model, process, handle, output))
        failures = []
        for model, process, handle, output in pending:
            code = process.wait()
            handle.close()
            if code or (read_json(output / "metrics.json") or {}).get("status") != "COMPLETE":
                failures.append({"model": model, "code": code})
        if failures:
            raise RuntimeError(f"final policy training failures: {failures}")
        outputs = {model: root / model for model, _hidden in specs}
        self.mark("policies", outputs={name: str(path) for name, path in outputs.items()})
        return outputs

    def ensure_calibration(self, policies: Mapping[str, Path], train: Path, split: Path) -> Path:
        output = OUT / "policies/jev/calibration_val_only.json"
        calibrated = OUT / "policies/jev/model_calibrated.pth"
        if (read_json(output) or {}).get("status") == "PASS" and calibrated.is_file():
            return calibrated
        if output.exists():
            archive(output)
        self.command(
            [
                PYTHON,
                "-u",
                ROOT / "reproduction_tools/calibrate_jev.py",
                "--dataset",
                train,
                "--checkpoint",
                policies["jev"] / "model.pth",
                "--policy-split",
                split,
                "--output",
                output,
                "--device",
                "cpu",
            ],
            log_name="calibrate_jev_final",
        )
        report = read_json(output)
        if not report or report.get("status") != "PASS":
            raise RuntimeError("final JEV calibration did not PASS")
        if calibrated.exists():
            archive(calibrated)
        import torch

        payload = torch.load(policies["jev"] / "model.pth", map_location="cpu")
        payload["temperature"] = float(report["temperature"])
        torch.save(payload, calibrated)
        return calibrated

    def ensure_lock(self, checkpoint: Path, split: Path) -> Path:
        lock = OUT / "manifests/FINAL_SELECTION_LOCK.json"
        payload = read_json(lock)
        if payload and payload.get("gmt_checkpoint_sha256") == "sha256:" + sha256(checkpoint):
            return lock
        if lock.exists():
            archive(lock)
        self.command(
            [
                PYTHON,
                "-u",
                ROOT / "reproduction_tools/create_final_selection_lock.py",
                "--checkpoint",
                checkpoint,
                "--policy-split",
                split,
                "--output",
                lock,
                "--state-schema-version",
                "2",
                "--counterfactual-engine-version",
                "cached_perception_mutable_association_v2",
                "--utility-definition",
                "future_correct_identity_duration - 0.5*future_identity_switches - 0.25*future_fragmentation - 0.5*future_collisions - memory_contamination; sample_weight=0 for uninformative futures",
                "--horizon",
                "8",
                "--jev-architecture",
                "jev",
                "--hidden-size",
                "64",
                "--calibration-method",
                "temperature_scaling_policy_val_only",
                "--threshold-baseline",
                "question_threshold_h70",
                "--mlp-baseline",
                "independent_mlp_h32",
                "--hyperparameters",
                json.dumps(
                    {
                        "dataset_scope": str(OUT / "formal/train/formal_train_full_v2.jsonl"),
                        "official_test_data_not_used_for_selection": True,
                        "epochs": 50,
                        "batch_size": 128,
                        "learning_rate": 1e-3,
                        "seed": 20261003,
                        "calibration_temperature_path": str(OUT / "policies/jev/calibration_val_only.json"),
                    },
                    sort_keys=True,
                ),
                "--code-commit",
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            ],
            log_name="create_final_selection_lock",
        )
        if not (read_json(lock) or {}).get("official_test_gate", {}).get("lock_created_before_official_test"):
            raise RuntimeError("final selection lock gate failed")
        return lock

    def ensure_stress(self, train: Path, policies: Mapping[str, Path]) -> Path:
        output = OUT / "decision_stress.json"
        if (read_json(output) or {}).get("status") == "PASS":
            return output
        if output.exists():
            archive(output)
        self.command(
            [
                PYTHON,
                "-u",
                ROOT / "reproduction_tools/analyze_jev_decisions.py",
                "--dataset",
                train,
                "--output",
                output,
                *[f"{name}={path}" for name, path in policies.items()],
            ],
            log_name="decision_stress_final",
        )
        if (read_json(output) or {}).get("status") != "PASS":
            raise RuntimeError("final decision stress test failed")
        return output

    def ensure_official(self, artifacts: Mapping[str, Path], test_data: Path, policies: Mapping[str, Path], calibrated: Path, lock: Path) -> Mapping[str, Path]:
        official = OUT / "official"
        official.mkdir(exist_ok=True)
        online_output = official / "online_jev"
        online_trace = official / "online_jev.jsonl"
        if not self.inference_complete(online_output, "VISION_test"):
            archive(online_output)
            archive(online_trace)
            self.command(
                self._inference_args(dataset="VISION_test", output=online_output, gpu="7", trace=online_trace)
                + ["--jev-controller", calibrated],
                cwd=CANONICAL,
                env={"CUDA_VISIBLE_DEVICES": "7"},
                log_name="official_online_jev",
            )
        base_predictions = artifacts["test_predictions"]
        trace = artifacts["test_trace"]
        annotation = DATASET / "annotations/test.json"
        controller_paths = {
            name: (calibrated if name == "jev" else path / "model.pth")
            for name, path in policies.items()
        }
        prediction_paths: dict[str, Path] = {"off": base_predictions, "online_jev": online_output / "inference_VISION_test/coco_instances_results.json"}
        for name, controller in controller_paths.items():
            output = official / "replays" / f"{name}.json"
            manifest = Path(str(output) + ".manifest.json")
            if not (output.is_file() and (read_json(manifest) or {}).get("status") == "PASS"):
                archive(output)
                archive(manifest)
                self.command(
                    [
                        PYTHON,
                        "-u",
                        ROOT / "reproduction_tools/replay_jev_policy.py",
                        "--predictions",
                        base_predictions,
                        "--trace",
                        trace,
                        "--annotations",
                        annotation,
                        "--controller",
                        controller,
                        "--output",
                        output,
                    ],
                    log_name=f"replay_{name}_final",
                )
            prediction_paths[name] = output
        oracle = official / "replays/oracle.json"
        oracle_manifest = Path(str(oracle) + ".manifest.json")
        if not (oracle.is_file() and (read_json(oracle_manifest) or {}).get("status") == "PASS"):
            archive(oracle)
            archive(oracle_manifest)
            self.command(
                [
                    PYTHON,
                    "-u",
                    ROOT / "reproduction_tools/replay_jev_policy.py",
                    "--predictions",
                    base_predictions,
                    "--trace",
                    trace,
                    "--annotations",
                    annotation,
                    "--oracle-dataset",
                    test_data,
                    "--output",
                    oracle,
                ],
                log_name="replay_oracle_final",
            )
        prediction_paths["oracle"] = oracle
        evaluations: dict[str, Path] = {}
        crossviews: dict[str, Path] = {}
        for name, prediction in prediction_paths.items():
            prepared = official / "prepared" / name
            evaluation = official / "evaluations" / name
            crossview = official / "crossview" / name
            if not (read_json(prepared / "manifest.json") or {}).get("status") == "PASS":
                archive(prepared)
                self.command(
                    [PYTHON, "-u", ROOT / "reproduction_tools/prepare_visiontrack_predictions.py", "--predictions", prediction, "--dataset", DATASET, "--split", "test", "--output", prepared],
                    cwd=ROOT,
                    log_name=f"prepare_{name}_final",
                )
            if not (read_json(evaluation / "metrics.json") or {}).get("status") == "PASS":
                archive(evaluation)
                self.command(
                    [PYTHON, "-u", ROOT / "reproduction_tools/evaluate_visiontrack.py", "--prepared", prepared, "--output", evaluation, "--allow-duplicate-gt"],
                    cwd=ROOT,
                    log_name=f"evaluate_{name}_final",
                )
            if not (read_json(crossview / "metrics.json") or {}).get("status") == "PASS":
                archive(crossview)
                self.command(
                    [PYTHON, "-u", ROOT / "reproduction_tools/evaluate_crossview_visiontrack.py", "--manifest", prepared / "manifest.json", "--output", crossview, "--allow-duplicate-gt"],
                    cwd=ROOT,
                    log_name=f"crossview_{name}_final",
                )
            evaluations[name] = evaluation
            crossviews[name] = crossview
        summary = official / "policy_summary.json"
        if not (read_json(summary) or {}).get("status") == "PASS":
            archive(summary)
            args = [PYTHON, "-u", ROOT / "reproduction_tools/summarize_policy_suite.py", "--output", summary]
            args.extend(f"{name}={path}" for name, path in evaluations.items())
            self.command(args, cwd=ROOT, log_name="summarize_final_policy_suite")
        cross_summary = official / "crossview_summary.json"
        if not (read_json(cross_summary) or {}).get("status") == "PASS":
            archive(cross_summary)
            args = [PYTHON, "-u", ROOT / "reproduction_tools/summarize_crossview_suite.py", "--output", cross_summary]
            args.extend(f"{name}={path}" for name, path in crossviews.items())
            self.command(args, cwd=ROOT, log_name="summarize_final_crossview_suite")
        oracle_gate = official / "oracle_gate.json"
        if not (read_json(oracle_gate) or {}).get("status") == "PASS":
            archive(oracle_gate)
            self.command(
                [
                    PYTHON,
                    "-u",
                    ROOT / "reproduction_tools/summarize_oracle_gate.py",
                    "--output",
                    oracle_gate,
                    "--counterfactual",
                    test_data,
                    "--trace",
                    trace,
                    "--off",
                    evaluations["off"],
                    "--oracle",
                    evaluations["oracle"],
                    "--threshold",
                    evaluations["question_threshold"],
                    "--jev",
                    evaluations["jev"],
                    "--off-cross",
                    crossviews["off"],
                    "--oracle-cross",
                    crossviews["oracle"],
                    "--threshold-cross",
                    crossviews["question_threshold"],
                    "--jev-cross",
                    crossviews["jev"],
                ],
                cwd=ROOT,
                log_name="summarize_final_oracle_gate",
            )
        report = {
            "status": "PASS",
            "created_utc": now(),
            "checkpoint": str(CHECKPOINT),
            "checkpoint_sha256": "sha256:" + sha256(CHECKPOINT),
            "selection_lock": str(lock),
            "future_gt_online": False,
            "counterfactual_engine": "cached_perception_mutable_association_v2",
            "official": {
                "policy_summary": str(summary),
                "crossview_summary": str(cross_summary),
                "oracle_gate": str(oracle_gate),
                "evaluations": {name: str(path) for name, path in evaluations.items()},
                "crossviews": {name: str(path) for name, path in crossviews.items()},
            },
            "go_no_go": "REVIEW_REQUIRED",
        }
        write_json(OUT / "FINAL_REPORT.json", report)
        return {"report": OUT / "FINAL_REPORT.json", "oracle_gate": oracle_gate, "policy_summary": summary}

    def run(self) -> None:
        self.ensure_checkpoint()
        artifacts = self.ensure_off_inference()
        formal = self.ensure_formal_data(
            {"train": artifacts["train_trace"], "test": artifacts["test_trace"]},
            {"train": artifacts["train_cache"], "test": artifacts["test_cache"]},
        )
        split = self.ensure_policy_split(formal["train"], formal["test"])
        policies = self.ensure_policies(formal["train"], split)
        calibrated = self.ensure_calibration(policies, formal["train"], split)
        self.ensure_stress(formal["train"], policies)
        lock = self.ensure_lock(CHECKPOINT, split)
        result = self.ensure_official(artifacts, formal["test"], policies, calibrated, lock)
        complete = {
            "status": "COMPLETE",
            "finished_utc": now(),
            "stage2_checkpoint": str(CHECKPOINT),
            "stage2_checkpoint_sha256": "sha256:" + sha256(CHECKPOINT),
            "final_report": str(result["report"]),
            "selection_lock": str(lock),
            "oracle_gate": str(result["oracle_gate"]),
            "future_gt_online": False,
            "go_no_go": "REVIEW_REQUIRED",
        }
        write_json(PIPE / "PIPELINE_COMPLETE.json", complete)
        self.status("complete", "COMPLETE", final_report=str(result["report"]))


def main() -> None:
    pipeline = FinalPipeline()
    try:
        pipeline.run()
    except Exception as exc:
        pipeline.status("failed", "FAILED", error=f"{type(exc).__name__}: {exc}")
        pipeline.handle.write(f"[FAILED] {type(exc).__name__}: {exc}\n")
        pipeline.handle.flush()
        raise
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
