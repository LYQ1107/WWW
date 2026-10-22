import torch

from worldmot.tracking.query_lifecycle import constrained_assign


def test_query_row_permutation_preserves_observation_partition():
    logits = torch.tensor([[[6.0, 1.0, 0.0], [1.0, 6.0, 0.0]]])
    births = torch.zeros(1, 2)
    a = constrained_assign(logits, births, [1, 2], torch.ones(1, 2))
    b = constrained_assign(logits[:, [1, 0]], births, [1, 2], torch.ones(1, 2))
    assert set(a.observation_to_query) == set(b.observation_to_query) == {0, 1}

