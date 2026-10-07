"""Contract test for per-history-slice GMT association normalization."""

import torch
from torch import nn

from jev_counterfactual_v2 import (
    CachedPerceptionMutableAssociationV2, MutableGMTState,
    promote_stale_bank_candidates,
)
from jev_gmt_association_adapter import GMTAssociationTransformerAdapter


class FakeROIHeads:
    def _forward_transformer(self, instances, reid_features, view_num, query_frame,
                             target_box, target_time, target_inst_id,
                             trajectory_rng=None):
        self.last_history_ids = target_inst_id.tolist()
        # One current row and two historical columns.  Native inference has
        # already removed the current query columns before this return.
        return [torch.zeros((1, 2))], None, None, None, None, None

    @staticmethod
    def _activate_asso(outputs):
        return [
            torch.cat([output, output.new_zeros((output.shape[0], 1))], dim=1)
            .softmax(dim=1)[:, :-1]
            for output in outputs
        ]


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.roi_heads = FakeROIHeads()


def payload(frame, view):
    return {
        "video_id": 8,
        "frame": frame,
        "view": view,
        "pred_boxes": torch.zeros((1, 4)),
        "reid_features": torch.ones((1, 2)),
        "image_size": (32, 32),
    }


def test_native_bank_uses_one_joint_softmax():
    state = MutableGMTState(
        reactivation_bank={2: torch.tensor([6.0, 8.0]), 1: torch.tensor([3.0, 4.0])},
        reactivation_mode=True,
    )
    state.initialize_trajectory_rng(8)
    adapter = GMTAssociationTransformerAdapter(FakeModel(), view_num=2)
    result = adapter(payload(1, 0), (1, 2), state)
    # Native old_reids is one bank, with one unmatched dummy column.
    assert torch.allclose(result.scores, torch.full((1, 2), 1.0 / 3.0), atol=1e-7)
    assert adapter.model.roi_heads.last_history_ids == [2, 1]


def test_native_memory_preserves_raw_feature_magnitude():
    state = MutableGMTState(
        next_id=2, active_ids={1, 2},
        association_history=[
            {"perception": payload(0, 0), "assignments": {0: 1}},
            {"perception": payload(0, 1), "assignments": {0: 2}},
        ],
    )
    state.initialize_trajectory_rng(8)
    current = payload(1, 0)
    current["reid_features"] = torch.tensor([[3.0, 4.0]])
    adapter = GMTAssociationTransformerAdapter(FakeModel(), view_num=2)
    engine = CachedPerceptionMutableAssociationV2(association_fn=adapter)
    engine.step(current, state, actions={0: "START_NEW"}, memory_actions={0: "WRITE_MEMORY"})
    assert torch.equal(state.memory[3][0], torch.tensor([3.0, 4.0]))
    state.memory[3].append(torch.tensor([6.0, 8.0]))
    promote_stale_bank_candidates(state, bank_size=2, recent_ids=())
    assert torch.equal(state.reactivation_bank[3], torch.tensor([4.5, 6.0]))


def main():
    first = payload(0, 0)
    second = payload(0, 1)
    current = payload(1, 0)
    state = MutableGMTState(
        association_history=[
            {"perception": first, "assignments": {0: 1}},
            {"perception": second, "assignments": {0: 2}},
        ]
    )
    state.initialize_trajectory_rng(8)
    adapter = GMTAssociationTransformerAdapter(FakeModel(), view_num=2, history_limit=8)
    subset = dict(current)
    subset["source_detection_indices"] = [0]
    assert adapter._payload_key(current) != adapter._payload_key(subset)
    result = adapter(current, (1, 2), state)
    scores = result.scores
    # Each historical slice is softmaxed with its own dummy unmatched column:
    # 1 / (1 + 1), not one softmax over both slices (1 / (2 + 1)).
    assert scores.shape == (1, 2)
    assert torch.allclose(scores, torch.full((1, 2), 1.0 / 2.0), atol=1e-7)
    assert result.transformer_calls == 1
    assert result.rng_state_before == result.rng_state_after
    test_native_bank_uses_one_joint_softmax()
    test_native_memory_preserves_raw_feature_magnitude()
    print("GMT association adapter slice-normalization invariant: PASS")


if __name__ == "__main__":
    main()
