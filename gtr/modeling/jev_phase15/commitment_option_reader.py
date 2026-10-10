"""Candidate-specific values; the terminal is each row's legal DEFER action."""
import torch
from torch import nn
from gtr.modeling.jev_stage2.model import Reader


class CommitmentOptionReader(nn.Module):
    def __init__(self, variant='dynamic', d=256):
        super().__init__(); self.variant = variant
        self.reader = Reader(d)
        self.terminal = nn.Parameter(torch.randn(1, 1, 1, d) * .02)
        self.value = nn.Sequential(nn.Linear(3*d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, x, query, tokens, det):
        b, q, k = x['legal'].shape; d = det.shape[-1]
        options = torch.cat([tokens, self.terminal.expand(b, q, 1, d)], 2)
        mask = torch.cat([x['legal'], torch.ones(b, q, 1, dtype=torch.bool, device=det.device)], -1)
        if self.variant == 'set':
            options = self.reader(options.reshape(b*q, k+1, d), options.reshape(b*q, k+1, d),
                                  mask.reshape(b*q, k+1)).reshape(b, q, k+1, d)
        else:
            options = self.reader(options.reshape(b*q, k+1, d), query.reshape(b*q, 1, d)).reshape(b, q, k+1, d)
        return self.value(torch.cat([options, query[:, :, None].expand(-1, -1, k+1, -1),
                                    det[:, :, None].expand(-1, -1, k+1, -1)], -1)).squeeze(-1)
