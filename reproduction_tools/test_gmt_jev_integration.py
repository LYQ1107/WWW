"""CPU invariants for the GMT proposal/decision/commit integration."""

import torch
from torch import nn

from gtr.modeling.jev_runtime import JEVRuntimePolicy
from gtr.modeling.meta_arch.gtr_rcnn import GTRRCNN


class AlwaysStart(nn.Module):
    def forward(self, state, questions, legal_actions):
        ids = {
            "ACCEPT_CURRENT": 0,
            "REASSOCIATE": 1,
            "START_NEW": 2,
            "WRITE_MEMORY": 3,
            "SKIP_MEMORY": 4,
            "REACTIVATE_OLD": 5,
        }
        row = list(legal_actions[0])
        action_ids = torch.tensor([[ids[name] for name in row]], dtype=torch.long)
        logits = torch.tensor([[-5.0 if name != "START_NEW" else 5.0 for name in row]])
        return {
            "probs": torch.softmax(logits, dim=1),
            "legal_actions": action_ids,
            "legal_mask": torch.ones_like(action_ids, dtype=torch.bool),
        }


def make_tracker(policy=None):
    tracker = object.__new__(GTRRCNN)
    nn.Module.__init__(tracker)
    tracker.register_parameter("_test_device", nn.Parameter(torch.zeros(1), requires_grad=False))
    tracker.jev_policy = policy
    tracker.jev_state_dim = 64
    tracker.jev_max_reassociate = 1
    tracker.with_bank = False
    tracker.with_iou = True
    tracker.not_mult_thresh = True
    tracker._jev_context = {"video_id": 7, "frame": 3}
    return tracker


def main():
    scores = torch.tensor([[0.8, 0.1], [0.1, 0.2]])
    unique_ids = torch.tensor([10, 20], dtype=torch.long)
    match_i = [0, 1]
    match_j = [0, 1]
    original = torch.tensor([10, -1], dtype=torch.long)

    off = make_tracker()
    result = off._apply_jev_match_decisions(
        original.clone(), scores, unique_ids, match_i, match_j, 0.2
    )
    assert torch.equal(result, original), result

    shadow = make_tracker(JEVRuntimePolicy("shadow", AlwaysStart()))
    result = shadow._apply_jev_match_decisions(
        original.clone(), scores, unique_ids, match_i, match_j, 0.2
    )
    assert torch.equal(result, original), result

    jev = make_tracker(JEVRuntimePolicy("jev", AlwaysStart()))
    result = jev._apply_jev_match_decisions(
        original.clone(), scores, unique_ids, match_i, match_j, 0.2
    )
    assert torch.equal(result, torch.tensor([-1, -1])), result
    assert jev._jev_memory_action(
        score=0.7,
        threshold=0.2,
        track_count=2,
        memory_count=4,
        view=0,
        frame_index=3,
        window_length=8,
        track_id=10,
    ) == "WRITE_MEMORY"
    print("GMT/JEV integration invariants: PASS")


if __name__ == "__main__":
    main()
