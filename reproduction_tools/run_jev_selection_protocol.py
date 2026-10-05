"""Run the pre-lock JEV policy/horizon/seed selection protocol.

Only formal TRAIN counterfactual JSONL files are accepted.  The script never
opens TEST annotations or TEST counterfactual records and writes a PASS
manifest only after every required candidate, seed, and validation-only
calibration has completed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/liuyeqiang/anaconda3/envs/GMT/bin/python"
CANONICAL_CHECKPOINT_SHA256 = (
    "cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8"
)
HORIZONS = (1, 8, 16, 32)
SEEDS = (20261003, 20261004, 20261005)
MODELS = (
    # Fixed and learned scalar-threshold controls, including the nonlinear
    # full-state threshold required by the reviewer protocol.
    "fixed_threshold",
    "global_threshold",
    "state_threshold",
    "nonlinear_state_threshold",
    "question_threshold",
    # Generic MLP controls and the proposed typed/action-conditioned models.
    "fixed_slot_mlp",
    "independent_mlp",
    "shared_heads",
    "logistic",
    "question_conditioned_mlp",
    "action_conditioned_no_question",
    "question_conditioned_fixed_head",
    "jev",
)
THRESHOLD_MODELS = {
    "fixed_threshold",
    "global_threshold",
    "state_threshold",
    "nonlinear_state_threshold",
    "question_threshold",
}
MLP_MODELS = set(MODELS) - THRESHOLD_MODELS - {"jev"}
HIDDEN_DIM = 64
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = "1e-3"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, Mapping) else None


def archive(path: Path) -> None:
    if not path.exists():
        return
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = path.with_name(path.name + ".incomplete." + stamp)
    suffix = 1
    while destination.exists():
        destination = path.with_name(path.name + f".incomplete.{stamp}.{suffix}")
        suffix += 1
    path.rename(destination)


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join(
        [
            str(ROOT),
            str(ROOT / "third_party/CenterNet2"),
            str(ROOT / "third_party/detectron2"),
            str(ROOT / "reproduction_tools"),
            env.get("PYTHONPATH", ""),
        ]
    )
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return env


def parse_assignments(values: list[str], required: tuple[int, ...], label: str) -> dict[int, Path]:
    parsed: dict[int, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{label} must be H=PATH: {value}")
        key, raw_path = value.split("=", 1)
        horizon = int(key)
        if horizon in parsed:
            raise ValueError(f"duplicate {label} horizon: {horizon}")
        parsed[horizon] = Path(raw_path).resolve()
    missing = set(required) - set(parsed)
    extra = set(parsed) - set(required)
    if missing or extra:
        raise ValueError(f"{label} horizons missing={sorted(missing)} extra={sorted(extra)}")
    for horizon, path in parsed.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} H={horizon}: {path}")
    return parsed


def validate_split(path: Path) -> Mapping[str, Any]:
    payload = read_json(path)
    if not payload:
        raise ValueError(f"invalid policy split: {path}")
    if payload.get("official_test_used_for_search") is not False:
        raise ValueError("selection split is not official-test excluded")
    if payload.get("official_test_access") != "BLOCKED_BEFORE_FINAL_SELECTION_LOCK":
        raise ValueError("selection split does not block pre-lock TEST access")
    if payload.get("official_test_files"):
        raise ValueError("selection split contains official TEST files")
    train = set(payload.get("train_sequences", ()))
    val = set(payload.get("val_sequences", ()))
    if not train or not val or train & val:
        raise ValueError("selection split has invalid train/val sequences")
    return payload


def run(command: list[str | Path], *, log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(str(item) for item in command) + "\n")
        handle.flush()
        result = subprocess.run(
            [str(item) for item in command],
            cwd=ROOT,
            env=environment(),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {command}")


def val_metrics(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    history = metrics.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("policy metrics has no training history")
    final = history[-1]
    if not isinstance(final, Mapping) or not isinstance(final.get("val"), Mapping):
        raise ValueError("policy metrics has no final validation metrics")
    value = final["val"]
    if not value.get("weighted_records"):
        raise ValueError("policy validation has no weighted records")
    return value


def _state_digest(state: Mapping[str, Any]) -> str:
    raw = json.dumps(
        state, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _forbidden_state_keys(value: Any, path: str = "state") -> list[str]:
    forbidden = {
        "gt",
        "ground_truth",
        "future_gt",
        "future_ground_truth",
        "evaluator_feedback",
        "future_metrics",
    }
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in forbidden:
                found.append(f"{path}.{key}")
            found.extend(_forbidden_state_keys(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_state_keys(child, f"{path}[{index}]"))
    return found


def audit_features(path: Path, split_payload: Mapping[str, Any], horizon: int) -> Mapping[str, Any]:
    allowed_sequences = set(split_payload["train_sequences"]) | set(split_payload["val_sequences"])
    sequences: set[str] = set()
    dimensions: set[int] = set()
    record_count = 0
    weighted_count = 0
    checkpoint_hashes: set[str] = set()
    forbidden: list[str] = []
    digest_mismatches = 0
    invalid = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record_count += 1
            try:
                record = json.loads(line)
                sequence = str(record["sequence"])
                state = record["state"]
                features = state["feature_vector"]
                if not isinstance(features, list) or not features:
                    raise ValueError("feature_vector is empty")
                if not all(math.isfinite(float(value)) for value in features):
                    raise ValueError("feature_vector contains non-finite values")
                dimensions.add(len(features))
                sequences.add(sequence)
                checkpoint_hashes.add(str(record["gmt_checkpoint_sha256"]))
                if int(record["horizon"]) != int(horizon):
                    raise ValueError("record horizon mismatch")
                if sequence not in allowed_sequences:
                    raise ValueError("record sequence is outside policy split")
                if record.get("state_digest") != _state_digest(state):
                    digest_mismatches += 1
                forbidden.extend(_forbidden_state_keys(state, f"{path}:{line_number}.state"))
                if float(record.get("sample_weight", 1.0)) > 0:
                    weighted_count += 1
            except Exception:
                invalid += 1
    status = bool(
        record_count
        and not invalid
        and len(dimensions) == 1
        and digest_mismatches == 0
        and not forbidden
        and sequences == allowed_sequences
        and checkpoint_hashes == {CANONICAL_CHECKPOINT_SHA256}
    )
    return {
        "status": "PASS" if status else "FAIL",
        "dataset": str(path),
        "horizon": int(horizon),
        "records": record_count,
        "weighted_records": weighted_count,
        "sequences": len(sequences),
        "feature_dimensions": sorted(dimensions),
        "checkpoint_hashes": sorted(checkpoint_hashes),
        "state_digest_mismatches": digest_mismatches,
        "forbidden_state_fields": sorted(set(forbidden))[:20],
        "invalid_records": invalid,
        "policy_split_sequence_exact": sequences == allowed_sequences,
        "official_test_read": False,
    }


def calibrate(checkpoint: Path, dataset: Path, split: Path, output: Path, log: Path) -> Path:
    report = output / "calibration_val_only.json"
    calibrated = output / "model_calibrated.pth"
    if (read_json(report) or {}).get("status") != "PASS" or not calibrated.is_file():
        if output.exists():
            archive(output)
        output.mkdir(parents=True, exist_ok=True)
        run(
            [
                PYTHON,
                "-u",
                ROOT / "reproduction_tools/calibrate_jev.py",
                "--dataset",
                dataset,
                "--checkpoint",
                checkpoint,
                "--policy-split",
                split,
                "--output",
                report,
                "--device",
                "cpu",
            ],
            log=log,
        )
        import torch

        payload = torch.load(checkpoint, map_location="cpu")
        payload["temperature"] = float((read_json(report) or {})["temperature"])
        torch.save(payload, calibrated)
    report_payload = read_json(report)
    if not report_payload or report_payload.get("status") != "PASS":
        raise RuntimeError(f"calibration did not PASS: {report}")
    return calibrated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", action="append", required=True, help="H=TRAIN formal JSONL")
    parser.add_argument("--policy-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hidden-dim", type=int, default=HIDDEN_DIM)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()
    if args.hidden_dim < 1 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("hidden-dim, epochs and batch-size must be positive")
    datasets = parse_assignments(args.dataset, HORIZONS, "dataset")
    split = args.policy_split.resolve()
    split_payload = validate_split(split)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    log = output / "selection_protocol.log"
    feature_audits = {
        horizon: audit_features(datasets[horizon], split_payload, horizon)
        for horizon in HORIZONS
    }
    if any(report.get("status") != "PASS" for report in feature_audits.values()):
        raise RuntimeError(f"feature audit failed: {feature_audits}")
    candidates: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        dataset = datasets[horizon]
        for seed in SEEDS:
            for model in MODELS:
                run_root = output / "runs" / f"h{horizon}" / f"seed{seed}" / model
                metrics_path = run_root / "metrics.json"
                checkpoint = run_root / "model.pth"
                if (read_json(metrics_path) or {}).get("status") != "COMPLETE" or not checkpoint.is_file():
                    if run_root.exists():
                        archive(run_root)
                    run(
                        [
                            PYTHON,
                            "-u",
                            ROOT / "reproduction_tools/train_jev.py",
                            "--dataset",
                            dataset,
                            "--output",
                            run_root,
                            "--model",
                            model,
                            "--epochs",
                            str(args.epochs),
                            "--batch-size",
                            str(args.batch_size),
                            "--hidden-dim",
                            str(args.hidden_dim),
                            "--lr",
                            LEARNING_RATE,
                            "--seed",
                            str(seed),
                            "--device",
                            "cpu",
                            "--fixed-threshold",
                            "0.2",
                            "--split-manifest",
                            split,
                        ],
                        log=log,
                    )
                metrics = read_json(metrics_path)
                if not metrics or metrics.get("status") != "COMPLETE":
                    raise RuntimeError(f"missing completed policy metrics: {metrics_path}")
                validation = val_metrics(metrics)
                calibrated = calibrate(
                    checkpoint,
                    dataset,
                    split,
                    run_root / "calibration",
                    log,
                )
                candidates.append(
                    {
                        "horizon": horizon,
                        "seed": seed,
                        "model": model,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": sha256(checkpoint),
                        "calibrated_checkpoint": str(calibrated),
                        "calibrated_checkpoint_sha256": sha256(calibrated),
                        "metrics": dict(validation),
                    }
                )

    # Aggregate seeds before selecting a horizon/model.  The seed used for the
    # selected artifact is then the deterministic best validation run inside
    # that aggregate, never a test-dependent choice.
    aggregates: list[dict[str, Any]] = []
    for horizon in HORIZONS:
        for model in MODELS:
            rows = [row for row in candidates if row["horizon"] == horizon and row["model"] == model]
            nll = [float(row["metrics"]["nll"]) for row in rows]
            accuracy = [float(row["metrics"]["best_action_accuracy"]) for row in rows]
            aggregates.append(
                {
                    "horizon": horizon,
                    "model": model,
                    "seed_count": len(rows),
                    "mean_val_nll": sum(nll) / len(nll),
                    "mean_val_best_action_accuracy": sum(accuracy) / len(accuracy),
                    "seed_results": rows,
                }
            )
    selected_aggregate = min(
        aggregates,
        key=lambda row: (
            row["mean_val_nll"],
            -row["mean_val_best_action_accuracy"],
            row["horizon"],
            row["model"],
        ),
    )
    selected = min(
        selected_aggregate["seed_results"],
        key=lambda row: (float(row["metrics"]["nll"]), -float(row["metrics"]["best_action_accuracy"]), row["seed"]),
    )

    def best_control(names: set[str]) -> dict[str, Any]:
        options = [row for row in aggregates if row["model"] in names]
        if not options:
            raise RuntimeError("selection protocol control family is empty")
        return min(
            options,
            key=lambda row: (
                row["mean_val_nll"],
                -row["mean_val_best_action_accuracy"],
                row["horizon"],
                row["model"],
            ),
        )

    selected_threshold = best_control(THRESHOLD_MODELS)
    selected_mlp = best_control(MLP_MODELS)
    payload = {
        "status": "PASS",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "protocol": "policy_val_only_horizon_model_seed_selection_v1",
        "canonical_checkpoint_sha256": "sha256:" + CANONICAL_CHECKPOINT_SHA256,
        "official_test_used_for_search": False,
        "official_test_access": "BLOCKED_BEFORE_FINAL_SELECTION_LOCK",
        "policy_split": str(split),
        "policy_split_sha256": sha256(split),
        "policy_split_sequence_hash": split_payload["sequence_hash"],
        "horizons": list(HORIZONS),
        "seeds": list(SEEDS),
        "models": list(MODELS),
        "equal_supervision": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "learning_rate": float(LEARNING_RATE),
            "sample_weight_field": "record.sample_weight",
        },
        "calibration": {
            "method": "temperature_scaling_policy_val_only",
            "all_candidates_calibrated": True,
        },
        "feature_audits": feature_audits,
        "candidate_count": len(candidates),
        "aggregate_count": len(aggregates),
        "selected_aggregate": {
            "horizon": selected_aggregate["horizon"],
            "model": selected_aggregate["model"],
            "mean_val_nll": selected_aggregate["mean_val_nll"],
            "mean_val_best_action_accuracy": selected_aggregate["mean_val_best_action_accuracy"],
        },
        "selected_threshold_control": {
            "horizon": selected_threshold["horizon"],
            "model": selected_threshold["model"],
            "mean_val_nll": selected_threshold["mean_val_nll"],
            "mean_val_best_action_accuracy": selected_threshold["mean_val_best_action_accuracy"],
        },
        "selected_mlp_control": {
            "horizon": selected_mlp["horizon"],
            "model": selected_mlp["model"],
            "mean_val_nll": selected_mlp["mean_val_nll"],
            "mean_val_best_action_accuracy": selected_mlp["mean_val_best_action_accuracy"],
        },
        "selected": selected,
        "selection_rule": "minimize mean policy-val NLL; tie-break by mean best-action accuracy, horizon, model; then choose best seed by the same val rule",
        "candidates": candidates,
        "aggregates": aggregates,
        "no_official_test_read": True,
    }
    (output / "jev_selection_protocol_final.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "output": str(output / "jev_selection_protocol_final.json"), "selected": selected}, indent=2))


if __name__ == "__main__":
    main()
