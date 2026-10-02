#!/usr/bin/env python3
"""Freeze the pre-registered offline gates into the final decision files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("decision_audit"))
    args = parser.parse_args()
    root = args.root
    coverage = load(root / "candidate_coverage.json")
    oracle = load(root / "oracle" / "ORACLE_CHOICE_HEADROOM.json")
    offline = load(root / "offline" / "offline_results.json")
    if coverage is None:
        raise RuntimeError("candidate_coverage.json is required")
    chosen = coverage.get("chosen_K")
    gates = {
        "candidate_recall_ge_95": bool(chosen is not None),
        "oracle_headroom": bool(oracle and oracle.get("gate", {}).get("AssA_plus_0.5_or_IDF1_plus_0.5") and oracle.get("gate", {}).get("MOTA_drop_at_most_1")),
        "set_choice_support": False,
        "comparison_against_linear_mlp": False,
    }
    comparison = {}
    if offline is not None:
        b2 = offline["models"]["B2_MLP"]["seeds"]
        b3 = offline["models"]["B3_SET_CHOICE"]["seeds"]
        improvements = []
        nll_better = []
        brier_better = []
        for seed in sorted(b2):
            delta = float(b3[seed]["dev_hard_recoverable"]["accuracy"] - b2[seed]["dev_hard_recoverable"]["accuracy"])
            improvements.append(delta)
            nll_better.append(b3[seed]["dev_all"]["nll"] < b2[seed]["dev_all"]["nll"])
            brier_better.append(b3[seed]["dev_all"]["brier"] < b2[seed]["dev_all"]["brier"])
        comparison = {
            "hard_accuracy_delta_setchoice_minus_mlp": improvements,
            "seeds_positive_over_mlp": int(sum(delta > 0 for delta in improvements)),
            "mean_hard_accuracy_delta": (sum(improvements) / len(improvements) if improvements else None),
            "nll_better_seeds": int(sum(nll_better)),
            "brier_better_seeds": int(sum(brier_better)),
        }
        gates["set_choice_support"] = bool(
            len(improvements) >= 3 and sum(delta >= 0.01 for delta in improvements) >= 2
            and (any(nll_better) or any(brier_better))
        )
        gates["comparison_against_linear_mlp"] = all(name in offline["models"] for name in ("B1_LINEAR", "B2_MLP"))

    if not gates["candidate_recall_ge_95"]:
        classification = "NO_DECISION_HEADROOM"
        reason = "K=16 candidate recall is below 95%; stop at candidate generation."
    elif not gates["oracle_headroom"]:
        classification = "NO_DECISION_HEADROOM"
        reason = "Oracle Choice does not pass the pre-registered practical tracking headroom gate."
    elif gates["set_choice_support"]:
        classification = "JEV_DECISION_GO"
        reason = "SetChoice passes the pre-registered hard-decision and probability-quality support gate."
    elif offline is not None:
        classification = "LEARNED_ASSOCIATION_ONLY"
        reason = "A learned association model may improve GMT, but SetChoice has no pre-registered advantage over independent MLP."
    else:
        classification = "NO_DECISION_HEADROOM"
        reason = "Offline decision models were not run because an earlier gate stopped the audit."

    payload = {
        "format": "gmt-decision-formulation-final-v1",
        "classification": classification,
        "reason": reason,
        "immutable_causal_state": {"STATE_CAUSAL_GO": "NO-GO", "MECHANISM_VALIDATED": "NO"},
        "gates": gates,
        "comparison": comparison,
        "coverage": coverage,
        "oracle": oracle,
        "offline_results_present": offline is not None,
        "visiontrack_test_used": False,
        "state_actions_implemented": False,
    }
    (root / "DECISION_FORMULATION_EVIDENCE_MATRIX.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Decision formulation evidence matrix", "",
        f"**Final classification: `{classification}`**", "", reason, "",
        "| gate | result |", "|---|---|",
        f"| candidate recall >=95% at frozen K | `{gates['candidate_recall_ge_95']}` |",
        f"| Oracle Choice practical headroom | `{gates['oracle_headroom']}` |",
        f"| SetChoice hard-decision support over MLP | `{gates['set_choice_support']}` |",
        f"| Linear/MLP comparison present | `{gates['comparison_against_linear_mlp']}` |", "",
        "Immutable prior causal state remains `STATE_CAUSAL_GO=NO-GO` and",
        "`MECHANISM_VALIDATED=NO`.  This audit implements choice and diagnostic",
        "verification only; no state action is enabled.  VisionTrack test GT was",
        "not used.", "",
    ]
    (root / "DECISION_FORMULATION_EVIDENCE_MATRIX.md").write_text("\n".join(lines))
    (root / "FINAL_DECISION.md").write_text("\n".join([
        "# Final decision", "", f"`{classification}`", "", reason, "",
        "Stop point: no base-GMT retraining, VisionTrack final test, DIVOTrack,",
        "state recovery, RLCD, or automatic next experiment is started.", "",
    ]))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
