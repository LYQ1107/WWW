"""Small val-only calibration smoke test."""

import json
from pathlib import Path
import tempfile

import torch

from calibrate_jev import fit_temperature, predictions
from gtr.modeling.jev_baselines import FixedSlotMLP


def main():
    records = []
    for index in range(4):
        records.append(
            {
                "state": {"feature_vector": [0.1, 0.2, 0.3, 0.4]},
                "question_type": "MATCH_DECISION",
                "legal_actions": ["ACCEPT_CURRENT", "REASSOCIATE", "START_NEW"],
                "target_probs": [0.8, 0.1, 0.1],
                "best_actions": ["ACCEPT_CURRENT"],
                "sample_weight": 1.0,
            }
        )
    torch.manual_seed(3)
    model = FixedSlotMLP(4, 8)
    temperature = fit_temperature(model, records)
    assert 0.05 <= temperature <= 20.0
    before = predictions(model, records)
    after = predictions(model, records, temperature)
    assert all(abs(sum(item["probabilities"]) - 1.0) < 1e-6 for item in after)
    assert before != after
    print("JEV calibration invariants: PASS")


if __name__ == "__main__":
    main()

