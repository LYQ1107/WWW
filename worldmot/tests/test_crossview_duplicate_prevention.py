import torch

from worldmot.tracking.query_lifecycle import constrained_assign


def test_cross_view_duplicate_prevention_is_observation_unique():
    logits = torch.tensor([[[8.0, 7.0, 0.0], [7.9, 7.8, 0.0]]])
    result = constrained_assign(logits, torch.zeros(1, 2), [1, 2], torch.ones(1, 2))
    assert len(result.observation_to_query) == 2
    assert len(set(result.observation_to_query)) == 2

