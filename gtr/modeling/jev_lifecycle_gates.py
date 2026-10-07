"""Strict MATCH-only gate controls for the Phase VI novelty audit.

These controllers have no tracker, annotation or evaluator dependency. Their
only transition is accepting the current assignment or requesting a new ID.
"""

import torch
from torch import nn

from .jev_baselines import _prepare_inputs, _result
from .jev_decision import ACTION_TO_INDEX, QUESTION_NAMES
from .jev_state import STATE_FEATURE_INDEX


class MatchThresholdGate(nn.Module):
    def __init__(self, state_dim=64, hidden_dim=None, initial_threshold=0.1):
        super().__init__()
        self.score_index = STATE_FEATURE_INDEX["raw_traj_score"]
        if state_dim <= self.score_index:
            raise ValueError("MATCH gate requires the canonical raw proposal score")
        self.base_threshold = nn.Parameter(torch.tensor(float(initial_threshold)))
        self.conditioner = None if hidden_dim is None else nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def threshold_for(self, features):
        state = features.float()
        if state.ndim == 1:
            state = state.unsqueeze(0)
        threshold = self.base_threshold.expand(state.shape[0])
        if self.conditioner is not None:
            threshold = threshold + self.conditioner(state).squeeze(-1)
        return threshold

    def forward(self, state_features, questions, legal_actions):
        state, questions, ids, mask = _prepare_inputs(state_features, questions, legal_actions)
        if bool((questions != QUESTION_NAMES.index("MATCH_DECISION")).any()):
            raise ValueError("Phase VI threshold gates control MATCH only")
        accept = ACTION_TO_INDEX["ACCEPT_CURRENT"]
        new = ACTION_TO_INDEX["START_NEW"]
        if bool((mask & (ids != accept) & (ids != new)).any()):
            raise ValueError("binary gate received a non-binary action")
        margin = state[:, self.score_index] - self.threshold_for(state)
        logits = torch.where(ids == accept, margin[:, None] / 2, -margin[:, None] / 2)
        result = _result(logits, ids, mask)
        result["threshold"] = self.threshold_for(state)
        return result

    @torch.no_grad()
    def decide_strict(self, features, legal_actions):
        """Apply the preregistered strict score > tau boundary, including ties."""
        if "ACCEPT_CURRENT" not in legal_actions:
            return "START_NEW"
        state = features.float().reshape(1, -1)
        return "ACCEPT_CURRENT" if bool(
            state[0, self.score_index] > self.threshold_for(state)[0]
        ) else "START_NEW"
