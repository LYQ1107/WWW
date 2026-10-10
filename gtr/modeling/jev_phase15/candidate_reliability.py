"""History purity, selection safety and uncertainty have separate outputs."""
import torch
from torch import nn


class CandidateReliability(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.visual = nn.Linear(1024, d)
        self.head = nn.Sequential(nn.Linear(2*d + 20 + 16, d), nn.GELU(), nn.Linear(d, 3))

    def forward(self, x):
        b, q, k = x['legal'].shape
        history = (x['history_visual'] * x['history_mask'][..., None]).sum(2) / x['history_mask'].sum(2).clamp_min(1)[..., None]
        current = self.visual(x['detection_visual'])[:, :, None].expand(-1, -1, k, -1)
        past = self.visual(history)[:, None].expand(-1, q, -1, -1)
        z = self.head(torch.cat([current, past, x['commitment_features'], x['pair_evidence']], -1))
        return dict(purity_logits=z[..., 0], safety_logits=z[..., 1], uncertainty_logits=z[..., 2])
