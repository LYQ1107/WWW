"""Verify offline checkpoint and online JEVRuntimePolicy semantic parity.

This verifier consumes only the current online state vector, typed question,
and legal actions from held-out records.  Counterfactual outcomes are never
passed to the controller.  It catches action-ID/order or runtime-wrapper
changes before a tracking replay is trusted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, Mapping

import torch

from gtr.modeling.jev_runtime import JEVRuntimePolicy, build_controller_from_checkpoint
from jev_dataset_contract import validate_record


METHODS = {
    "question_threshold": "Learnable Threshold",
    "question_conditioned_mlp": "Generic MLP",
    "jev": "Full JEV",
}


def read_records(path: Path, limit: int):
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            validate_record(record, allow_future_gt=True)
            if record.get("uses_future_gt") is not True:
                raise ValueError(
                    f"expected frozen counterfactual record at line {line_number}"
                )
            if not isinstance(record["state"].get("feature_vector"), list):
                raise ValueError(f"missing feature vector at line {line_number}")
            records.append(record)
            if len(records) >= limit:
                break
    if len(records) < limit:
        raise ValueError(f"only {len(records)} records available; need {limit}")
    return records


def compare_method(checkpoint: Path, records) -> Dict[str, Any]:
    controller = build_controller_from_checkpoint(checkpoint, device="cpu")
    runtime = JEVRuntimePolicy("jev", controller)
    max_probability_error = 0.0
    mismatches = []
    for index, record in enumerate(records):
        feature = torch.tensor(
            record["state"]["feature_vector"], dtype=torch.float32
        )
        question = str(record["question_type"])
        legal = list(record["legal_actions"])
        with torch.no_grad():
            direct = controller(feature.unsqueeze(0), [question], [legal])
        direct_ids = direct["legal_actions"][0].detach().cpu().tolist()
        direct_probs = direct["probs"][0].detach().cpu().tolist()
        direct_map = {
            (name if isinstance(name, str) else None): float(probability)
            for name, probability in zip(
                [
                    ("ACCEPT_CURRENT", "REASSOCIATE", "START_NEW", "WRITE_MEMORY", "SKIP_MEMORY", "REACTIVATE_OLD")[action_id]
                    for action_id in direct_ids
                ],
                direct_probs,
            )
        }
        decision = runtime.decide(feature, question, legal, off_action=legal[0])
        error = max(
            abs(float(direct_map[name]) - float(decision.probabilities[name]))
            for name in legal
        )
        max_probability_error = max(max_probability_error, error)
        direct_action = max(direct_map, key=direct_map.get)
        if error > 1e-6 or direct_action != decision.proposed_action:
            mismatches.append(
                {
                    "index": index,
                    "question": question,
                    "legal_actions": legal,
                    "direct_action": direct_action,
                    "runtime_action": decision.proposed_action,
                    "max_probability_error": error,
                }
            )
    return {
        "status": "PASS" if not mismatches else "FAIL",
        "checkpoint": str(checkpoint),
        "records": len(records),
        "mismatches": mismatches,
        "mismatch_count": len(mismatches),
        "max_probability_error": max_probability_error,
        "future_gt_read_for_model": False,
        "model_input_fields": ["state.feature_vector", "question_type", "legal_actions"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("limit must be positive")
    records = read_records(args.records, args.limit)
    methods = {}
    for method, label in METHODS.items():
        checkpoint = args.checkpoint_dir / method / "model.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        result = compare_method(checkpoint, records)
        result["label"] = label
        methods[method] = result
    report = {
        "status": "PASS" if all(item["status"] == "PASS" for item in methods.values()) else "FAIL",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "classification": "SCREENING_PARITY_NOT_A_PAPER_RESULT",
        "records": len(records),
        "official_test_read": False,
        "methods": methods,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
