import torch

from worldmot.models.persistent_world_queries import PersistentWorldQueryTransformer
from worldmot.training.world_query_criterion import WorldQueryCriterion


def test_pwqt_forward_keeps_slot_and_global_id_state_separate():
    model = PersistentWorldQueryTransformer(
        observation_dim=340, hidden_dim=32, max_world_queries=4,
        cameras=2, heads=4, decoder_layers=1,
    )
    observations = torch.randn(1, 2, 340)
    geometry = torch.randn(1, 2, 18)
    state = model.init_state(1)
    state.global_ids[0, 0] = 17
    out = model(observations, geometry, state)
    assert out["assignment_logits"].shape == (1, 4, 3)
    assert out["state"].global_ids[0, 0].item() == 17
    assert out["state"].slot_ids.tolist() == [[0, 1, 2, 3]]


def test_multiview_assignment_target_has_a_finite_backward_pass():
    model = PersistentWorldQueryTransformer(
        observation_dim=340, hidden_dim=32, max_world_queries=4,
        cameras=2, heads=4, decoder_layers=1,
    )
    out = model(torch.randn(1, 2, 340), torch.randn(1, 2, 18))
    assignment = torch.zeros_like(out["assignment_logits"], dtype=torch.int64)
    assignment[:, :, -1] = 1
    assignment[:, 0, -1] = 0
    assignment[:, 0, 0] = 1
    assignment[:, 0, 1] = 1  # one query, two camera observations
    losses = WorldQueryCriterion()(out, {
        "assignment": assignment,
        "birth": torch.zeros(1, 2),
        "visibility": torch.zeros(1, 4, 2),
        "observation_valid": torch.ones(1, 2, dtype=torch.bool),
    })
    losses["total"].backward()
    assert torch.isfinite(losses["total"])
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

