import torch

from worldmot.models.multi_view_query_attention import MultiViewQueryAttention


def test_joint_attention_exposes_per_camera_observation_weights():
    module = MultiViewQueryAttention(hidden_dim=32, heads=4, layers=1, geometry_dim=18)
    out = module(torch.randn(1, 3, 32), torch.randn(1, 4, 32),
                 torch.randn(1, 4, 18), torch.randn(1, 3, 18),
                 torch.ones(1, 4, dtype=torch.bool))
    assert out.attention.shape == (1, 1, 4, 3, 4)
    assert out.geometry_bias.shape == (1, 1, 4, 3, 4)
    assert torch.isfinite(out.query_features).all()

