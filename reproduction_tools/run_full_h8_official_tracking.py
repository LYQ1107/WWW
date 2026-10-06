"""Run the locked Full-H=8 controller comparison on official VISION_test.

This runner is deliberately separate from the Full-H=8 data builder.  It
never runs GMT OFF again: the OFF row is the frozen canonical baseline audit.
The three learned controllers are evaluated with the same Stage2 checkpoint,
same read-only TEST perception cache, same current repository code, and the
same output/evaluation pipeline.

The command is fail-closed.  It accepts no official TEST run until the Full
H=8 postprocess and the corrected video1 runtime/candidate parity gates are
PASS.  Official TEST annotations are consumed only by the final evaluation
steps, never by controller selection or runtime inference.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/liuyeqiang/anaconda3/envs/GMT/bin/python"
SEED = 20261003
RNG_MASTER_SEED = 20261006
METHODS = {
    "question_threshold": "Learnable Threshold",
    "question_conditioned_mlp": "Generic MLP",
    "jev": "Full JEV",
}
PRIMARY_METRICS = ("HOTA", "DetA", "AssA", "IDF1", "MOTA", "IDSW", "Frag")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def scalar(value: Any) -> float:
    if isinstance(value, list):
        if not value:
            return 0.0
        return sum(float(item) for item in value) / len(value)
    return float(value)


def load_frozen_baseline(path: Path, checkpoint: Path) -> dict[str, Any]:
    payload = read_json(path)
    frozen = payload.get("frozen_canonical_gmt_baseline")
    if not isinstance(frozen, Mapping):
        raise ValueError("baseline audit has no frozen_canonical_gmt_baseline section")
    expected = {
        "HOTA": 67.442,
        "DetA": 66.278,
        "AssA": 68.992,
        "IDF1": 82.239,
        "MOTA": 80.942,
        "IDSW": 3092,
        "Frag": 8004,
        "CVIDF1": 79.0248,
        "CVMA": 80.9276,
    }
    if frozen.get("rerun") is not False:
        raise ValueError("frozen baseline audit does not prove rerun=false")
    if frozen.get("checkpoint_sha256") != sha256(checkpoint):
        raise ValueError("frozen baseline checkpoint digest does not match Stage2")
    for key, value in expected.items():
        observed = float(frozen.get(key, float("nan")))
        if observed != float(value):
            raise ValueError(f"frozen baseline {key} changed: {observed} != {value}")
    if frozen.get("OFF_equivalence") != "PASS":
        raise ValueError("frozen OFF equivalence is not PASS")
    return {
        "label": "Frozen GMT baseline",
        "classification": "FROZEN_CANONICAL_BASELINE_REUSED_NO_RERUN",
        "audit": str(path.resolve()),
        "audit_sha256": sha256(path),
        "metrics": expected,
    }


def require_pass(path: Path, label: str) -> Mapping[str, Any]:
    payload = read_json(path)
    if payload.get("status") != "PASS":
        raise RuntimeError(f"{label} is not PASS: {path}")
    return payload


def validate_gates(
    runtime: Path,
    repo: Path,
    formal_gate: Path,
) -> dict[str, Any]:
    gate_files = {
        "formal_chunk_equivalence": formal_gate,
        "video1_provenance": repo / "reports/JEV_RNG_V4/CURRENT_HEAD_VIDEO01_PROVENANCE.json",
        "video1_feature_parity": repo / "reports/JEV_RNG_V4/RUNTIME_FEATURE_PARITY_VIDEO01_CURRENT_HEAD.json",
        "video1_candidate_parity": repo / "reports/JEV_RNG_V4/REACTIVATION_CANDIDATE_PARITY_CURRENT_HEAD_VIDEO01.json",
        "video1_stability": repo / "reports/JEV_RNG_V4/RUNTIME_FEATURE_PARITY_STABILITY_VIDEO01_CURRENT_HEAD.json",
        "video1_closed_loop": repo / "reports/JEV_RNG_V4/CURRENT_HEAD_VIDEO01_THREE_WAY_TRACKING.json",
        "full_h8_aftercare": repo / "reports/JEV_RNG_V4/FULL_H8_CURRENT_HEAD_AFTERCARE.json",
    }
    evidence: dict[str, Any] = {}
    for name, path in gate_files.items():
        payload = require_pass(path, name)
        evidence[name] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "status": payload.get("status"),
        }
    postprocess = runtime / "POSTPROCESS_MANIFEST.json"
    postprocess_payload = require_pass(postprocess, "Full H8 postprocess manifest")
    first_round = runtime / "FIRST_ROUND_REPORT.json"
    first_round_payload = require_pass(first_round, "Full H8 first-round report")
    if list(first_round_payload.get("seeds", ())) != [SEED]:
        raise ValueError("Full H8 first-round report is not the locked single-seed result")
    evidence["postprocess_manifest"] = {"path": str(postprocess), "sha256": sha256(postprocess)}
    evidence["first_round_report"] = {"path": str(first_round), "sha256": sha256(first_round)}
    return evidence


def validate_methods(runtime: Path) -> dict[str, dict[str, Any]]:
    methods_root = runtime / "FIRST_ROUND_METHODS"
    result: dict[str, dict[str, Any]] = {}
    expected_dataset = str((runtime / "FULL_H8_POLICY_COMPACT").resolve())
    for method, label in METHODS.items():
        root = methods_root / method
        manifest_path = root / "method_manifest.json"
        manifest = require_pass(manifest_path, f"{label} method manifest")
        if list(manifest.get("seeds", ())) != [SEED]:
            raise ValueError(f"{label} is not single-seed {SEED}")
        if str(Path(str(manifest.get("dataset", ""))).resolve()) != expected_dataset:
            raise ValueError(f"{label} was not trained from Full H8 compact data")
        members = manifest.get("members")
        if not isinstance(members, list) or len(members) != 1:
            raise ValueError(f"{label} does not have exactly one locked member")
        calibrated = Path(str(members[0].get("calibrated_checkpoint", ""))).resolve()
        calibration_report = Path(str(members[0].get("calibration_report", ""))).resolve()
        if not calibrated.is_file() or not calibration_report.is_file():
            raise FileNotFoundError(f"{label} calibrated checkpoint/calibration report")
        calibration = require_pass(calibration_report, f"{label} calibration")
        if calibration.get("official_test_read") is not False:
            raise ValueError(f"{label} calibration read official TEST")
        result[method] = {
            "label": label,
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256(manifest_path),
            "checkpoint": str(calibrated),
            "checkpoint_sha256": sha256(calibrated),
            "calibration": str(calibration_report),
            "calibration_sha256": sha256(calibration_report),
            "seed": SEED,
            "dataset": expected_dataset,
        }
    return result


def run_command(command: list[str | Path], *, cwd: Path, env: Mapping[str, str] | None = None) -> None:
    completed = subprocess.run(
        [str(item) for item in command],
        cwd=cwd,
        env=dict(env) if env is not None else None,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(map(str, command))}")


def ensure_inference(
    *,
    root: Path,
    checkpoint: Path,
    config: Path,
    cache: Path,
    output: Path,
    trace: Path,
    gpu: str,
) -> Path:
    result = output / "inference_VISION_test" / "coco_instances_results.json"
    manifest = output / "inference_manifest.json"
    if manifest.is_file() and result.is_file():
        payload = read_json(manifest)
        if payload.get("status") != "COMPLETE":
            raise RuntimeError(f"partial or failed inference output exists: {output}")
        return result
    if output.exists():
        raise RuntimeError(f"refusing to overwrite partial official inference output: {output}")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = ":".join(
        [str(root), str(root / "third_party/CenterNet2"), str(root / "reproduction_tools"), environment.get("PYTHONPATH", "")]
    )
    run_command(
        [
            PYTHON,
            "-u",
            root / "reproduction_tools/run_formal_inference.py",
            "--checkpoint",
            checkpoint,
            "--config",
            config,
            "--output",
            output,
            "--gpu",
            gpu,
            "--dataset",
            "VISION_test",
            "--jev-mode",
            "jev",
            "--jev-controller",
            checkpoint,
            "--jev-trace",
            trace,
            "--perception-cache-read",
            cache,
            "--trajectory-rng-master-seed",
            str(RNG_MASTER_SEED),
        ],
        cwd=root,
        env=environment,
    )
    if not result.is_file():
        raise RuntimeError(f"inference completed without predictions: {result}")
    return result


def ensure_evaluation(
    *,
    root: Path,
    predictions: Path,
    dataset: Path,
    output: Path,
) -> tuple[Path, Path]:
    prepared = output / "prepared"
    evaluation = output / "evaluation"
    crossview = output / "crossview"
    prepared_manifest = prepared / "manifest.json"
    evaluation_metrics = evaluation / "metrics.json"
    crossview_metrics = crossview / "metrics.json"
    if not prepared_manifest.is_file():
        if prepared.exists():
            raise RuntimeError(f"partial prepared output exists: {prepared}")
        run_command(
            [
                PYTHON,
                "-u",
                root / "reproduction_tools/prepare_visiontrack_predictions.py",
                "--predictions",
                predictions,
                "--dataset",
                dataset,
                "--split",
                "test",
                "--output",
                prepared,
            ],
            cwd=root,
        )
    require_pass(prepared_manifest, "official prepared evaluation")
    if not evaluation_metrics.is_file():
        if evaluation.exists():
            raise RuntimeError(f"partial evaluation output exists: {evaluation}")
        run_command(
            [
                PYTHON,
                "-u",
                root / "reproduction_tools/evaluate_visiontrack.py",
                "--prepared",
                prepared,
                "--output",
                evaluation,
                "--allow-duplicate-gt",
            ],
            cwd=root,
        )
    require_pass(evaluation_metrics, "official TrackEval evaluation")
    if not crossview_metrics.is_file():
        if crossview.exists():
            raise RuntimeError(f"partial cross-view output exists: {crossview}")
        run_command(
            [
                PYTHON,
                "-u",
                root / "reproduction_tools/evaluate_crossview_visiontrack.py",
                "--manifest",
                prepared_manifest,
                "--output",
                crossview,
                "--allow-duplicate-gt",
            ],
            cwd=root,
        )
    require_pass(crossview_metrics, "official cross-view evaluation")
    return evaluation_metrics, crossview_metrics


def tracking_metrics(evaluation: Path, crossview: Path) -> dict[str, Any]:
    report = read_json(evaluation)
    combined = report["combined_metrics"]
    metrics = {
        "HOTA": scalar(combined["HOTA"]["HOTA"]) * 100.0,
        "DetA": scalar(combined["HOTA"]["DetA"]) * 100.0,
        "AssA": scalar(combined["HOTA"]["AssA"]) * 100.0,
        "IDF1": scalar(combined["Identity"]["IDF1"]) * 100.0,
        "MOTA": scalar(combined["CLEAR"]["MOTA"]) * 100.0,
        "IDSW": scalar(combined["CLEAR"]["IDSW"]),
        "Frag": scalar(combined["CLEAR"]["Frag"]),
    }
    cross = read_json(crossview).get("reports", {})
    sequential = cross.get("cvidf1", {})
    metrics["CVIDF1"] = float(sequential["CVIDF1"])
    metrics["CVMA"] = float(sequential["CVMA"])
    return metrics


def trace_stats(path: Path) -> dict[str, Any]:
    opener = gzip.open if path.suffix == ".gz" else open
    questions: dict[str, int] = {}
    committed: dict[str, int] = {}
    proposed: dict[str, int] = {}
    off_actions: dict[str, int] = {}
    committed_by_question: dict[str, dict[str, int]] = {}
    proposed_by_question: dict[str, dict[str, int]] = {}
    total = 0
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            total += 1
            for field, counts in (
                ("question", questions),
                ("committed_action", committed),
                ("proposed_action", proposed),
                ("off_action", off_actions),
            ):
                value = row.get(field)
                if value is not None:
                    counts[str(value)] = counts.get(str(value), 0) + 1
            question = row.get("question")
            committed_action = row.get("committed_action")
            proposed_action = row.get("proposed_action")
            if question is not None and committed_action is not None:
                question_counts = committed_by_question.setdefault(str(question), {})
                action = str(committed_action)
                question_counts[action] = question_counts.get(action, 0) + 1
            if question is not None and proposed_action is not None:
                question_counts = proposed_by_question.setdefault(str(question), {})
                action = str(proposed_action)
                question_counts[action] = question_counts.get(action, 0) + 1
    def rates(counts: Mapping[str, int]) -> dict[str, float]:
        return {key: value / max(1, total) for key, value in sorted(counts.items())}
    def question_rates(counts: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, float]]:
        return {
            question: {
                action: value / max(1, sum(actions.values()))
                for action, value in sorted(actions.items())
            }
            for question, actions in sorted(counts.items())
        }
    return {
        "events": total,
        "question_counts": questions,
        "committed_action_counts": committed,
        "committed_action_rates": rates(committed),
        "committed_action_counts_by_question": committed_by_question,
        "committed_action_rates_by_question": question_rates(committed_by_question),
        "proposed_action_counts": proposed,
        "proposed_action_counts_by_question": proposed_by_question,
        "proposed_action_rates_by_question": question_rates(proposed_by_question),
        "off_action_counts": off_actions,
    }


def markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Full H=8 official controller comparison",
        "",
        "This report is generated only after the current-head runtime gates and Full H=8 aftercare pass.",
        "The GMT OFF row is the frozen canonical baseline; OFF inference is not rerun.",
        "",
        "## Primary tracking metrics",
        "",
        "| Method | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | Frag | CVIDF1 | CVMA |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, row in report["methods"].items():
        metrics = row["metrics"]
        lines.append(
            f"| {row['label']} | {metrics['HOTA']:.3f} | {metrics['DetA']:.3f} | "
            f"{metrics['AssA']:.3f} | {metrics['IDF1']:.3f} | {metrics['MOTA']:.3f} | "
            f"{metrics['IDSW']:.0f} | {metrics['Frag']:.0f} | {metrics['CVIDF1']:.4f} | {metrics['CVMA']:.4f} |"
        )
    lines += [
        "",
        "## Committed action rates",
        "",
        "| Method | MATCH accept | MATCH reassoc | MATCH new | MEMORY write | MEMORY skip | REACTIVATE old | REACTIVATE new |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    action_columns = (
        ("MATCH_DECISION", "ACCEPT_CURRENT"),
        ("MATCH_DECISION", "REASSOCIATE"),
        ("MATCH_DECISION", "START_NEW"),
        ("MEMORY_DECISION", "WRITE_MEMORY"),
        ("MEMORY_DECISION", "SKIP_MEMORY"),
        ("REACTIVATION_DECISION", "REACTIVATE_OLD"),
        ("REACTIVATION_DECISION", "START_NEW"),
    )
    for name, row in report["methods"].items():
        by_question = row.get("decision_trace", {}).get("committed_action_rates_by_question", {})
        values = []
        for question, action in action_columns:
            if question not in by_question:
                values.append("n/a")
            else:
                values.append(f"{100.0 * float(by_question[question].get(action, 0.0)):.2f}%")
        lines.append(f"| {row['label']} | " + " | ".join(values) + " |")
    lines += ["", "## Delta versus frozen GMT OFF", "", "| Method | ΔHOTA | ΔAssA | ΔIDF1 | ΔMOTA | ΔIDSW | ΔFrag |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, row in report["methods"].items():
        if name == "gmt_off":
            continue
        delta = row["delta_vs_frozen_gmt_off"]
        lines.append(
            f"| {row['label']} | {delta['HOTA']:+.3f} | {delta['AssA']:+.3f} | "
            f"{delta['IDF1']:+.3f} | {delta['MOTA']:+.3f} | {delta['IDSW']:+.0f} | {delta['Frag']:+.0f} |"
        )
    lines += [
        "",
        "## Interpretation policy",
        "",
        "The report records metrics and deltas without selecting a winner automatically. "
        "A paper claim requires reviewing association metrics, IDSW/Frag, action traces, "
        "and the frozen baseline provenance together.",
        "",
        f"- Status: `{report['status']}`",
        f"- Official TEST was read only after training: `{report['official_test_read']}`",
        f"- Full H=8 seed: `{report['seed']}`",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/VISION_test.yaml")
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("/data/DATASETS/TRACKING/JDE/VisionTrack"))
    parser.add_argument("--formal-gate-report", type=Path, required=True)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--gpus", default="4,8,9")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    runtime = args.runtime_root.resolve()
    repo = args.repo_root.resolve()
    checkpoint = args.checkpoint.resolve()
    config = args.config.resolve()
    cache = args.test_cache.resolve()
    dataset = args.dataset.resolve()
    output_root = (args.output_root or (runtime / "official_tracking")).resolve()
    if not checkpoint.is_file() or not config.is_file() or not dataset.is_dir():
        raise FileNotFoundError("checkpoint, config, and VisionTrack dataset are required")
    if not (cache / "index.jsonl").is_file():
        raise FileNotFoundError(cache / "index.jsonl")
    gate_evidence = validate_gates(runtime, repo, args.formal_gate_report.resolve())
    frozen = load_frozen_baseline(args.baseline_audit.resolve(), checkpoint)
    method_evidence = validate_methods(runtime)
    gpus = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if len(gpus) != len(METHODS):
        raise ValueError("exactly three GPUs are required for the three controller runs")

    jobs: dict[str, dict[str, Any]] = {}
    processes: list[tuple[str, subprocess.Popen, Path]] = []
    environment = os.environ.copy()
    environment["PYTHONPATH"] = ":".join(
        [str(repo), str(repo / "third_party/CenterNet2"), str(repo / "reproduction_tools"), environment.get("PYTHONPATH", "")]
    )
    output_root.mkdir(parents=True, exist_ok=True)
    for (method, evidence), gpu in zip(method_evidence.items(), gpus):
        method_root = output_root / method
        inference_root = method_root / "inference"
        trace = method_root / "online_decisions.jsonl.gz"
        log = method_root / "launcher.log"
        method_root.mkdir(parents=True, exist_ok=True)
        result_path = inference_root / "inference_VISION_test" / "coco_instances_results.json"
        inference_manifest = inference_root / "inference_manifest.json"
        if inference_manifest.is_file() and result_path.is_file():
            payload = read_json(inference_manifest)
            if payload.get("status") != "COMPLETE":
                raise RuntimeError(f"incomplete existing inference: {inference_root}")
            jobs[method] = {"evidence": evidence, "gpu": gpu, "result": str(result_path)}
            continue
        if inference_root.exists():
            raise RuntimeError(f"refusing to overwrite partial inference: {inference_root}")
        log_handle = log.open("w", encoding="utf-8")
        command = [
            PYTHON,
            "-u",
            repo / "reproduction_tools/run_formal_inference.py",
            "--checkpoint",
            checkpoint,
            "--config",
            config,
            "--output",
            inference_root,
            "--gpu",
            gpu,
            "--dataset",
            "VISION_test",
            "--jev-mode",
            "jev",
            "--jev-controller",
            evidence["checkpoint"],
            "--jev-trace",
            trace,
            "--perception-cache-read",
            cache,
            "--trajectory-rng-master-seed",
            str(RNG_MASTER_SEED),
        ]
        process = subprocess.Popen(
            [str(item) for item in command],
            cwd=repo,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_handle.close()
        processes.append((method, process, inference_root))
        jobs[method] = {"evidence": evidence, "gpu": gpu, "result": str(result_path), "pid": process.pid}

    failures = []
    for method, process, _output in processes:
        code = process.wait()
        if code:
            failures.append({"method": method, "return_code": code})
    if failures:
        raise RuntimeError(f"official controller inference failed: {failures}")

    frozen_metrics = dict(frozen["metrics"])
    jobs_report: dict[str, Any] = {
        "gmt_off": {"label": frozen["label"], "metrics": frozen_metrics, "provenance": frozen},
    }
    for method, evidence in method_evidence.items():
        method_root = output_root / method
        predictions = method_root / "inference/inference_VISION_test/coco_instances_results.json"
        evaluation, crossview = ensure_evaluation(
            root=repo,
            predictions=predictions,
            dataset=dataset,
            output=method_root,
        )
        metrics = tracking_metrics(evaluation, crossview)
        delta = {key: metrics[key] - frozen_metrics[key] for key in PRIMARY_METRICS + ("CVIDF1", "CVMA")}
        trace = method_root / "online_decisions.jsonl.gz"
        if not trace.is_file():
            raise FileNotFoundError(trace)
        jobs_report[method] = {
            "label": evidence["label"],
            "metrics": metrics,
            "delta_vs_frozen_gmt_off": delta,
            "provenance": {
                **evidence,
                "inference_manifest": str((method_root / "inference/inference_manifest.json").resolve()),
                "inference_manifest_sha256": sha256(method_root / "inference/inference_manifest.json"),
                "predictions": str(predictions.resolve()),
                "predictions_sha256": sha256(predictions),
                "test_cache": str(cache),
                "test_cache_index_sha256": sha256(cache / "index.jsonl"),
                "config": str(config),
                "config_sha256": sha256(config),
                "trajectory_rng_master_seed": RNG_MASTER_SEED,
                "decision_trace": str(trace.resolve()),
                "decision_trace_sha256": sha256(trace),
            },
            "decision_trace": trace_stats(trace),
            "evaluation": str(evaluation.resolve()),
            "crossview": str(crossview.resolve()),
        }

    report = {
        "schema_version": "jev_full_h8_official_tracking_v1",
        "status": "PASS",
        "created_utc": now(),
        "classification": "FULL_H8_CURRENT_HEAD_OFFICIAL_TEST_CLOSED_LOOP_COMPARISON",
        "official_test_read": True,
        "future_gt_online": False,
        "seed": SEED,
        "stage2_checkpoint": str(checkpoint),
        "stage2_checkpoint_sha256": sha256(checkpoint),
        "test_perception_cache": str(cache),
        "test_perception_cache_index_sha256": sha256(cache / "index.jsonl"),
        "gate_evidence": gate_evidence,
        "methods": jobs_report,
        "evaluation_policy": {
            "trackeval_allow_duplicate_gt": True,
            "crossview_time_axis_for_delta": "sequential",
            "frozen_off_rerun": False,
            "same_controller_runtime": "native GMT mutable state with online typed commits",
        },
        "go_no_go": "REVIEW_REQUIRED",
    }
    write_json(args.output_report.resolve(), report)
    args.output_markdown.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.resolve().write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"status": "PASS", "report": str(args.output_report.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
