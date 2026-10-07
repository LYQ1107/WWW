"""ID-free variable-cardinality MATCH candidate scorers.

The model scores candidate-local evidence under a legal-set mask.  Candidate
track IDs never enter the network; they are used only by the caller to bind a
selected row back to the current GMT proposal.
"""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int | None = None):
        super().__init__()
        output = hidden_dim if out_dim is None else out_dim
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output),
            nn.LayerNorm(output),
            nn.GELU(),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.network(value)


class CandidateConditionedScorer(nn.Module):
    """Shared state/candidate scorer with optional candidate interaction."""

    def __init__(
        self,
        state_dim: int,
        candidate_dim: int = 8,
        hidden_dim: int = 128,
        model_name: str = "candidate_jev",
    ):
        super().__init__()
        if model_name not in {"candidate_mlp", "candidate_jev"}:
            raise ValueError(f"unknown candidate model: {model_name}")
        if hidden_dim % 4 != 0 and model_name == "candidate_jev":
            raise ValueError("candidate_jev hidden_dim must be divisible by four")
        self.model_name = str(model_name)
        self.state_dim = int(state_dim)
        self.candidate_dim = int(candidate_dim)
        self.hidden_dim = int(hidden_dim)
        self.state_encoder = _MLP(self.state_dim, self.hidden_dim)
        self.candidate_encoder = _MLP(self.candidate_dim, self.hidden_dim)
        if model_name == "candidate_jev":
            layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=4,
                dim_feedforward=self.hidden_dim * 2,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.candidate_interaction = nn.TransformerEncoder(layer, num_layers=1)
        else:
            self.candidate_interaction = None
        self.scorer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(
        self,
        state_features: Tensor,
        candidate_features: Tensor,
        candidate_mask: Tensor,
    ) -> Dict[str, Tensor]:
        if state_features.ndim != 2:
            raise ValueError("state_features must have shape [B, D]")
        if candidate_features.ndim != 3:
            raise ValueError("candidate_features must have shape [B, K, C]")
        if candidate_mask.shape != candidate_features.shape[:2]:
            raise ValueError("candidate_mask must have shape [B, K]")
        if state_features.shape[0] != candidate_features.shape[0]:
            raise ValueError("state/candidate batch size mismatch")
        state = self.state_encoder(state_features.float())
        candidates = self.candidate_encoder(candidate_features.float())
        if self.candidate_interaction is not None:
            candidates = self.candidate_interaction(
                candidates, src_key_padding_mask=~candidate_mask.bool()
            )
        fused = torch.cat(
            [state.unsqueeze(1).expand(-1, candidates.shape[1], -1), candidates],
            dim=-1,
        )
        logits = self.scorer(fused).squeeze(-1)
        logits = logits.masked_fill(~candidate_mask.bool(), torch.finfo(logits.dtype).min)
        probabilities = F.softmax(logits, dim=-1)
        probabilities = probabilities.masked_fill(~candidate_mask.bool(), 0.0)
        return {"logits": logits, "probs": probabilities, "candidate_mask": candidate_mask.bool()}
