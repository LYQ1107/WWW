"""Q4 reads previous commitment support, real candidates and current evidence."""
import torch
from torch import nn
from gtr.modeling.jev_stage2.model import Reader


class CommitmentQuestion(nn.Module):
    def __init__(self, variant='dynamic', d=256):
        super().__init__(); self.variant = variant
        self.visual = nn.Linear(1024, d); self.meta = nn.Linear(8, d)
        self.support = nn.Linear(20, d); self.score = nn.Linear(3, d)
        self.fixed = nn.Parameter(torch.randn(1, 1, d) * .02)
        self.reader = Reader(d) if variant == 'dynamic' else None
        self.pool = nn.Sequential(nn.Linear(d,4*d),nn.GELU(),nn.Linear(4*d,d),nn.LayerNorm(d)) if variant == 'pooled' else None

    def forward(self, x, who, trust, availability):
        b, q, k = x['legal'].shape
        det = self.visual(x['detection_visual']) + self.meta(x['detection_meta'])
        history = self.visual(x['history_visual'][:, :, 2])
        tokens = history[:, None] + self.support(x['commitment_features'])
        scores = torch.stack([who[:, :, :-1], trust, availability[:, :, None].expand(-1, -1, k)], -1)
        tokens = tokens + self.score(scores)
        if self.variant == 'fixed':
            # Only question is static. Every option still receives all observations.
            query = self.fixed.expand(b, q, -1)
        else: query = det
        if k and self.reader is not None:
            query = self.reader(query.reshape(b*q, 1, -1), tokens.reshape(b*q, k, -1),
                                x['legal'].reshape(b*q, k)).reshape(b, q, -1)
        elif self.pool is not None:
            context = (tokens*x['legal'][...,None]).sum(2)/x['legal'].sum(2).clamp_min(1)[...,None]
            query = self.pool(query+context)
        return query, tokens, det
