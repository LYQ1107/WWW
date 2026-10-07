"""Validate the corrected controller bundle before video01 closed loop.

The report is bound to the completed video01 v2 records and manifest.  It
loads the calibrated single-seed controllers through the same runtime loader
used by closed-loop replay and performs one finite forward-pass smoke check.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import torch

from gtr.modeling.jev_runtime import build_controller_from_checkpoint


METHODS = (
    "question_threshold",
    "question_conditioned_mlp",
    "jev",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return value


def finite_tree(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tree(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item()) if value.is_floating_point() else True
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods-root", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    methods_root = args.methods_root.resolve()
    records = args.records.resolve()
    manifest_path = args.manifest.resolve()
    training_report = args.training_report.resolve()
    manifest = read_json(manifest_path)
    training = read_json(training_report)
    issues: list[str] = []
    if manifest.get("status") not in {"PASS", "COMPLETE"}:
        issues.append("v2 manifest is not complete")
    if manifest.get("source_commit") != args.expected_source_commit:
        issues.append("v2 manifest source commit mismatch")
    if not records.is_file():
        issues.append("v2 records are missing")
    if training.get("status") != "PASS":
        issues.append("corrected training report is not PASS")

    evidence: dict[str, Any] = {
        "methods_root": str(methods_root),
        "records": str(records),
        "records_sha256": sha256(records) if records.is_file() else None,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "manifest_source_commit": manifest.get("source_commit"),
        "expected_source_commit": args.expected_source_commit,
        "training_report": str(training_report),
        "training_report_sha256": sha256(training_report),
        "methods": {},
    }

    for method in METHODS:
        method_root = methods_root / method / "seed_20261003"
        method_manifest_path = methods_root / method / "method_manifest.json"
        calibration_report_path = method_root / "calibration" / "calibration_val_only.json"
        checkpoint = method_root / "calibration" / "model_calibrated.pth"
        method_item: dict[str, Any] = {
            "method": method,
            "method_manifest": str(method_manifest_path),
            "calibration_report": str(calibration_report_path),
            "checkpoint": str(checkpoint),
            "status": "FAIL",
        }
        try:
            method_manifest = read_json(method_manifest_path)
            calibration = read_json(calibration_report_path)
            method_item["method_manifest_status"] = method_manifest.get("status")
            method_item["dataset"] = method_manifest.get("dataset")
            method_item["seeds"] = method_manifest.get("seeds")
            method_item["calibration_status"] = calibration.get("status")
            if method_manifest.get("status") != "PASS":
                raise RuntimeError("method manifest is not PASS")
            if method_manifest.get("seeds") != [20261003]:
                raise RuntimeError("method seed policy is not [20261003]")
            if "small_h8_training_v4_canonical_features" not in str(method_manifest.get("dataset", "")):
                raise RuntimeError("method is not bound to corrected canonical-feature dataset")
            if calibration.get("status") != "PASS":
                raise RuntimeError("calibration report is not PASS")
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            payload = torch.load(str(checkpoint), map_location="cpu")
            if not isinstance(payload, Mapping):
                raise RuntimeError("checkpoint payload is not a mapping")
            method_item["checkpoint_sha256"] = sha256(checkpoint)
            method_item["checkpoint_model_name"] = payload.get("model_name")
            method_item["state_dim"] = int(payload.get("state_dim", -1))
            if int(payload.get("state_dim", -1)) != 64:
                raise RuntimeError("checkpoint state_dim is not 64")
            if not finite_tree(payload.get("model")):
                raise RuntimeError("checkpoint model contains non-finite values")
            controller = build_controller_from_checkpoint(checkpoint, device="cpu")
            controller.eval()
            features = torch.zeros((1, 64), dtype=torch.float32)
            questions = torch.zeros((1,), dtype=torch.long)
            legal_actions = torch.tensor([[0, 1, 2, -1, -1]], dtype=torch.long)
            with torch.no_grad():
                output = controller(features, questions, legal_actions)
            probs = output.get("probs") if isinstance(output, Mapping) else None
            if not torch.is_tensor(probs) or not bool(torch.isfinite(probs).all().item()):
                raise RuntimeError("runtime wrapper forward returned non-finite probabilities")
            method_item["wrapper_forward"] = "PASS"
            method_item["status"] = "PASS"
        except Exception as exc:  # noqa: BLE001 - report every method failure
            method_item["error"] = f"{type(exc).__name__}: {exc}"
            issues.append(f"controller_checkpoint_failed_{method}")
        evidence["methods"][method] = method_item

    report = {
        "schema_version": "jev_video01_v2_controller_wrapper_v1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "PASS" if not issues else "FAIL",
        "classification": "V2_BOUND_CONTROLLER_CHECKPOINT_PREFLIGHT",
        "issues": sorted(set(issues)),
        "evidence": evidence,
    }
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "issues": report["issues"]}, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
