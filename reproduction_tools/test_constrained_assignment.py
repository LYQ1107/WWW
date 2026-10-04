"""Regression tests for reviewer-proof global REASSOCIATE semantics."""

import torch
from torch import nn

from gtr.modeling.jev_assignment import constrained_hungarian
from gtr.modeling.jev_runtime import JEVRuntimePolicy
from gtr.modeling.meta_arch.gtr_rcnn import GTRRCNN


class SequenceController(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, state, questions, legal_actions):
        names = list(legal_actions[0])
        if self.calls == 0 and "REASSOCIATE" in names:
            selected = "REASSOCIATE"
        else:
            selected = "ACCEPT_CURRENT" if "ACCEPT_CURRENT" in names else names[0]
        self.calls += 1
        logits = torch.tensor(
            [[5.0 if name == selected else -5.0 for name in names]],
            dtype=state.dtype,
            device=state.device,
        )
        ids = torch.tensor(
            [[
                {
                    "ACCEPT_CURRENT": 0,
                    "REASSOCIATE": 1,
                    "START_NEW": 2,
                    "WRITE_MEMORY": 3,
                    "SKIP_MEMORY": 4,
                    "REACTIVATE_OLD": 5,
                }[name]
                for name in names
            ]],
            dtype=torch.long,
            device=state.device,
        )
        return {
            "probs": torch.softmax(logits, dim=1),
            "legal_actions": ids,
            "legal_mask": torch.ones_like(ids, dtype=torch.bool),
        }


def make_tracker(policy):
    tracker = object.__new__(GTRRCNN)
    nn.Module.__init__(tracker)
    tracker.register_parameter("_test_device", nn.Parameter(torch.zeros(1), requires_grad=False))
    tracker.jev_policy = policy
    tracker.jev_state_dim = 64
    tracker.jev_max_reassociate = 1
    tracker.with_bank = False
    tracker.with_iou = True
    tracker.not_mult_thresh = True
    tracker._jev_context = {}
    return tracker


def main():
    direct = constrained_hungarian(
        torch.tensor(
            [
                [0.90, 0.05, 0.10],
                [0.85, 0.10, 0.80],
                [0.10, 0.79, 0.78],
            ]
        ),
        banned_edges={(0, 0)},
    )
    assert (0, 0) not in direct
    assert set(direct) == {(0, 2), (1, 0), (2, 1)}, direct

    scores = torch.tensor(
        [
            [0.90, 0.05, 0.10],
            [0.85, 0.10, 0.80],
            [0.10, 0.79, 0.78],
        ]
    )
    tracker = make_tracker(JEVRuntimePolicy("jev", SequenceController()))
    result = tracker._apply_jev_match_decisions(
        torch.tensor([10, 20, 30]),
        scores,
        torch.tensor([10, 20, 30]),
        [0, 1, 2],
        [0, 2, 1],
        0.05,
        track_lengths=torch.tensor([1.0, 1.0, 1.0]),
    )
    # The initial assignment is [0, 2, 1].  Rejecting (row 0, col 0) must
    # produce the global constrained assignment [2, 0, 1], not row 0's local
    # second-ranked column 1.
    assert torch.equal(result, torch.tensor([30, 10, 20])), result
    print("Constrained Hungarian/JEV reassociation invariants: PASS")


if __name__ == "__main__":
    main()
