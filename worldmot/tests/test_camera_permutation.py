import torch

from worldmot.models.persistent_world_queries import PersistentWorldQueryTransformer
from worldmot.tracking.query_lifecycle import constrained_assign


def test_camera_labels_are_used_only_for_per_camera_uniqueness():
    logits = torch.tensor([[[7.0, 6.0, 0.0]]])
    births = torch.zeros(1, 2)
    result = constrained_assign(logits, births, [2, 1], torch.ones(1, 1))
    assert result.query_to_observations[0] == [0, 1]


def test_model_is_equivariant_to_observation_order_when_camera_ids_follow():
    torch.manual_seed(4)
    model = PersistentWorldQueryTransformer(hidden_dim=32, max_world_queries=4, heads=4, decoder_layers=1).eval()
    observations = torch.randn(1, 2, 340)
    geometry = torch.randn(1, 2, 18)
    ids = torch.tensor([[1, 2]])
    first = model(observations, geometry, camera_ids=ids)
    second = model(observations[:, [1, 0]], geometry[:, [1, 0]], camera_ids=ids[:, [1, 0]])
    assert torch.allclose(first["assignment_logits"][:, :, [1, 0, 2]], second["assignment_logits"], atol=1e-5)

