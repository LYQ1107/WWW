"""Contract test for per-history-slice GMT association normalization."""

import torch
from torch import nn

from jev_counterfactual_v2 import MutableGMTState
from jev_gmt_association_adapter import GMTAssociationTransformerAdapter


class FakeROIHeads:
    def _forward_transformer(self, instances, reid_features, view_num, query_frame,
                             target_box, target_time, target_inst_id,
                             trajectory_rng=None):
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


class RecordingROIHeads:
    def __init__(self):
        self.last_instances = None

    def _forward_transformer(self, instances, reid_features, view_num, query_frame,
                             target_box, target_time, target_inst_id,
                             trajectory_rng=None):
        self.last_instances = instances
        return [torch.zeros((1, 2))], None, None, None, None, None

    @staticmethod
    def _activate_asso(outputs):
        return [
            torch.cat([output, output.new_zeros((output.shape[0], 1))], dim=1)
            .softmax(dim=1)[:, :-1]
            for output in outputs
        ]


class RecordingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.roi_heads = RecordingROIHeads()


def payload(frame, view):
    return {
        "video_id": 8,
        "frame": frame,
        "view": view,
        "pred_boxes": torch.zeros((1, 4)),
        "reid_features": torch.ones((1, 2)),
        "image_size": (32, 32),
    }


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

    recording_model = RecordingModel()
    reactive_state = MutableGMTState(
        reactivation_mode=True,
        reactivation_bank={
            3: torch.tensor([1.0, 0.0]),
            5: torch.tensor([0.0, 1.0]),
        },
        reactivation_bank_boxes={
            3: torch.tensor([1.0, 2.0, 5.0, 6.0]),
            5: torch.tensor([7.0, 8.0, 9.0, 10.0]),
        },
        reactivation_bank_image_sizes={3: (32, 32), 5: (32, 32)},
    )
    reactive_state.initialize_trajectory_rng(8)
    reactive_adapter = GMTAssociationTransformerAdapter(
        recording_model, view_num=2, history_limit=8
    )
    reactive_adapter(current, (3, 5), reactive_state)
    seen = recording_model.roi_heads.last_instances
    assert seen is not None
    # Native old_reids is one historical Instances object containing every
    # stale ID, followed by the current unmatched-detection object.
    assert len(seen) == 2
    assert len(seen[0]) == 2
    assert seen[0].track_ids.tolist() == [3, 5]
    assert torch.equal(
        seen[0].pred_boxes.tensor,
        torch.tensor([[1.0, 2.0, 5.0, 6.0], [7.0, 8.0, 9.0, 10.0]]),
    )
    print("GMT association adapter slice-normalization invariant: PASS")
    print("GMT reactivation native old-reid geometry invariant: PASS")


if __name__ == "__main__":
    main()
