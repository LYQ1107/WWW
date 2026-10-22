import io

import torch

from worldmot.models.query_model_profiles import build_profile
from worldmot.models.persistent_world_queries import PersistentWorldQueryTransformer


def test_large_profile_scales_capacity_without_changing_observation_or_id_contract():
    small = build_profile('small')
    large = build_profile('large')
    counts = [sum(p.numel() for p in model.parameters() if p.requires_grad)
              for model in (small, large)]
    assert counts[0] == 953365
    assert counts[1] > 7 * counts[0]
    for model in (small, large):
        assert model.observation_dim == 532
        assert model.max_world_queries == 128
        assert model.init_state(1).global_ids.eq(-1).all()
    assert len(large.attention.query_self) == 6


def test_large_profile_restores_config_and_carries_real_observation_gradient():
    torch.manual_seed(23)
    model = build_profile('large', max_queries=4).eval()
    first = torch.randn(1, 3, 532, requires_grad=True)
    geometry = torch.randn(1, 3, 18)
    output = model(first, geometry, camera_ids=torch.tensor([[1, 2, 1]]))
    state = model.accept_observations(output['state'], output, [{0: [0, 1]}])
    state.global_ids[0, 0] = 7
    second = model(torch.randn(1, 2, 532), torch.randn(1, 2, 18), state)
    second['assignment_logits'][0, 0, 0].backward()
    assert first.grad.abs().sum() > 0
    for branch in (model.rgb_projection, model.detector_projection, model.omega_projection):
        assert sum(p.grad.abs().sum() for p in branch.parameters() if p.grad is not None) > 0
    buffer = io.BytesIO()
    torch.save({'model': model.state_dict(), 'config': model.config}, buffer)
    buffer.seek(0)
    payload = torch.load(buffer, weights_only=True)
    restored = PersistentWorldQueryTransformer(**payload['config']).eval()
    restored.load_state_dict(payload['model'])
    with torch.no_grad():
        expected = model(first.detach(), geometry)['pair_logits']
        actual = restored(first.detach(), geometry)['pair_logits']
    assert torch.equal(expected, actual)
