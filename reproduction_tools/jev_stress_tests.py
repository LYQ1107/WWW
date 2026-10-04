"""Structural stress tests for the typed JEV decision contract.

The script is deliberately usable before real rollout records exist.  It
checks invariants that must hold for every checkpoint, while clearly labeling
the resulting report as a contract test rather than a tracking result.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Dict

import torch

from gtr.modeling.jev_baselines import FixedThresholdPolicy
from gtr.modeling.jev_decision import JEVDecisionController, legal_actions_for
from gtr.modeling.jev_runtime import JEVRuntimePolicy


def ece(confidence, correct, bins=10):
    if not confidence:
        return 0.0
    total = len(confidence)
    value = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        selected = [
            i for i, score in enumerate(confidence)
            if lower <= score < upper
            or (index == bins - 1 and lower <= score <= upper)
        ]
        if selected:
            mean_conf = sum(confidence[i] for i in selected) / len(selected)
            mean_correct = sum(correct[i] for i in selected) / len(selected)
            value += len(selected) / total * abs(mean_conf - mean_correct)
    return value


def main():
    torch.manual_seed(20261003)
    model = JEVDecisionController(
        state_dim=8,
        hidden_dim=32,
        question_dim=16,
        action_dim=16,
        use_option_interaction=True,
    ).eval()
    same_score_different_state = torch.tensor(
        [[0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.5, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    legal = [legal_actions_for("MATCH_DECISION"), legal_actions_for("MATCH_DECISION")]
    result = model(same_score_different_state, ["MATCH_DECISION"] * 2, legal)
    state_sensitivity = float((result["probs"][0] - result["probs"][1]).abs().max())
    assert state_sensitivity > 1e-8

    permuted = model(
        same_score_different_state[:1],
        ["MATCH_DECISION"],
        [["START_NEW", "ACCEPT_CURRENT", "REASSOCIATE"]],
    )
    reference = result["probs"][0]
    reference_by_name = {name: reference[i].item() for i, name in enumerate(legal[0])}
    for index, name in enumerate(["START_NEW", "ACCEPT_CURRENT", "REASSOCIATE"]):
        assert abs(permuted["probs"][0, index].item() - reference_by_name[name]) < 1e-5

    masked = model(
        same_score_different_state[:1],
        ["REACTIVATION_DECISION"],
        [["START_NEW"]],
    )
    assert masked["probs"].shape == (1, 1) and abs(masked["probs"].item() - 1.0) < 1e-8

    fixed = FixedThresholdPolicy(0.5).eval()
    threshold_result = fixed(
        torch.tensor([[0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        ["MATCH_DECISION"],
        [legal_actions_for("MATCH_DECISION")],
    )
    assert threshold_result["probs"].argmax().item() == 2

    # Runtime source must stay free of evaluator/future-label imports.
    runtime_source = inspect.getsource(JEVRuntimePolicy)
    assert "ground_truth" not in runtime_source
    assert "import evaluator" not in runtime_source

    confidence = [float(row.max()) for row in result["probs"]]
    correctness = [1.0, 0.0]
    report: Dict[str, object] = {
        "status": "PASS",
        "contract_only": True,
        "same_score_different_state_max_probability_delta": state_sensitivity,
        "legal_mask_single_action": True,
        "action_permutation_equivariance": True,
        "fixed_threshold_boundary": True,
        "runtime_no_future_gt_import": True,
        "synthetic_ece": ece(confidence, correctness),
        "note": "No real tracking metric is claimed until Stage2 counterfactual records exist.",
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
