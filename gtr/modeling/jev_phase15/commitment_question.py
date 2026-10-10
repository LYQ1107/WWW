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
        self.reader = Reader(d)

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
        if k:
            query = self.reader(query.reshape(b*q, 1, -1), tokens.reshape(b*q, k, -1),
                                x['legal'].reshape(b*q, k)).reshape(b, q, -1)
        return query, tokens, det
