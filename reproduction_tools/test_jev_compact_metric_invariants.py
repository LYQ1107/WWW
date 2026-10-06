"""Regression tests for masked policy probability and calibration invariants."""

import numpy as np
import torch

from aggregate_jev_three_way_compact import _ece, _assert_probability_contract
from calibrate_jev_compact import assert_probability_contract as assert_calibration_contract
from train_jev_compact import assert_probability_contract as assert_training_contract


def main() -> None:
    probabilities = np.asarray(
        [[0.25, 0.75, 0.0], [0.5, 0.5, 0.0]], dtype=np.float64
    )
    legal = np.asarray([[0, 1, -1], [0, 2, -1]], dtype=np.int64)
    _assert_probability_contract(probabilities, legal)
    confidence = [(0.75, 1.0, 1.0), (0.5, 0.0, 1.0)]
    assert 0.0 <= _ece(confidence, 2.0) <= 1.0

    torch_probabilities = torch.tensor(probabilities, dtype=torch.float32)
    torch_legal = torch.tensor(legal, dtype=torch.long)
    weights = torch.ones(2, dtype=torch.float32)
    assert_training_contract(torch_probabilities, torch_legal, weights)
    assert_calibration_contract(torch_probabilities, torch_legal, weights)

    invalid = probabilities.copy()
    invalid[0, 0] = 0.6
    invalid[0, 1] = 0.6
    try:
        _assert_probability_contract(invalid, legal)
    except AssertionError:
        pass
    else:
        raise AssertionError("invalid probability mass was not rejected")
    print("JEV compact metric invariants: PASS")


if __name__ == "__main__":
    main()
