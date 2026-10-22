import torch

from worldmot.tracking.query_lifecycle import constrained_assign


def test_one_query_can_claim_one_observation_per_camera():
    # Query 0 strongly explains both views; query 1 is a competing duplicate.
    logits = torch.tensor([[[8.0, 7.9, 0.0], [7.5, 7.4, 0.0]]])
    births = torch.tensor([[0.1, 0.1]])
    result = constrained_assign(logits, births, [1, 2], torch.ones(1, 2), null_margin=0.0)
    assert result.query_to_observations[0] == [0, 1]
    assert result.observation_to_query == {0: 0, 1: 0}


def test_same_camera_duplicate_is_rejected_and_can_birth():
    logits = torch.tensor([[[8.0, 7.9, 0.0]]])
    births = torch.tensor([[0.95, 0.95]])
    result = constrained_assign(logits, births, [1, 1], torch.ones(1, 1), null_margin=0.0)
    assert len(result.observation_to_query) == 1
    assert len(result.births) == 1
    assert result.duplicate_rejections >= 1

