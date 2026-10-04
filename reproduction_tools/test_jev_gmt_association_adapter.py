"""Shape/contract test for the GMT association adapter without a detector."""

import torch
from torch import nn

from jev_counterfactual_v2 import CachedPerceptionMutableAssociationV2, MutableGMTState
from jev_gmt_association_adapter import GMTAssociationTransformerAdapter


class FakeROIHeads:
    def _forward_transformer(self, instances, reid_features, view_num, query_frame, *_):
        current = len(instances[-1])
        previous = sum(len(item) for item in instances[:-1])
        return [reid_features.new_ones((current, previous))], None, None, None, None, None

    @staticmethod
    def _activate_asso(outputs):
        return outputs


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.roi_heads = FakeROIHeads()


def payload(value):
    return {
        "pred_boxes": torch.tensor([[0.0, 0.0, 4.0, 4.0]]),
        "reid_features": torch.tensor([[value, 1.0]]),
        "image_size": (8, 8),
    }


def main():
    state = MutableGMTState(
        next_id=1,
        active_ids={1},
        track_embeddings={1: torch.tensor([1.0, 1.0])},
        track_hits={1: 1},
    )
    engine = CachedPerceptionMutableAssociationV2(
        association_fn=GMTAssociationTransformerAdapter(FakeModel())
    )
    engine.step(payload(1.0), state, actions={0: "ACCEPT_CURRENT"})
    scores = engine.score_matrix(payload(1.0), state)[1]
    assert scores.shape == (1, 1), scores.shape
    print("JEV GMT association-adapter invariants: PASS")


if __name__ == "__main__":
    main()

