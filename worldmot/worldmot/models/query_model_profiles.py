"""Registered PWQT capacity profiles with identical observation/state contracts.

The large profile scales the existing learned association architecture. It is
not a reproduction of MOTIP or MeMOTR, and it does not import their models.
"""
from .persistent_world_queries import PersistentWorldQueryTransformer


MODEL_PROFILES = {
    'small': {'hidden_dim': 128, 'heads': 4, 'decoder_layers': 2},
    'large': {'hidden_dim': 256, 'heads': 8, 'decoder_layers': 6},
}


def build_profile(name, *, max_queries=128, ablation='joint'):
    if name not in MODEL_PROFILES:
        raise ValueError('unknown registered model profile: ' + name)
    return PersistentWorldQueryTransformer(
        observation_dim=532, max_world_queries=max_queries, ablation=ablation,
        **MODEL_PROFILES[name])
