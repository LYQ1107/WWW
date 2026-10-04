"""Baselines that consume the same typed JEV state/action interface.

These policies are intentionally kept separate from the GMT tracker.  They
are useful for offline decision experiments and for the eventual SHADOW
adapter: every policy receives the same current-state tensor, typed question,
and runtime legal-action set, and returns masked action probabilities.  None
of the policies can create a track ID or mutate tracker state.

The first four state columns have a documented convention for threshold
baselines only (neural baselines may use the full state vector):

    0: current association/accept score
    1: reassociation score
    2: memory-write score
    3: reactivation score

An adapter may provide richer features, but it must preserve this prefix when
comparing against the fixed-threshold family.
"""

from typing import Dict, Mapping, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from .jev_decision import (
    ACTION_NAMES,
    ACTION_TO_INDEX,
    QUESTION_NAMES,
    JEVDecisionController,
    JEVStateEncoder,
    _validate_actions,
)


QuestionLike = Union[str, int]
ActionLike = Union[str, int]


def _prepare_inputs(
    state_features: Tensor,
    questions: Union[QuestionLike, Sequence[QuestionLike], Tensor],
    legal_actions: Union[
        Tensor,
        Sequence[ActionLike],
        Sequence[Sequence[ActionLike]],
    ],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    if state_features.ndim == 1:
        state_features = state_features.unsqueeze(0)
    if state_features.ndim != 2:
        raise ValueError("state_features must have shape [B, state_dim]")
    state_features = state_features.float()
    batch_size = state_features.shape[0]
    questions_tensor = JEVDecisionController._questions_tensor(
        questions, batch_size, state_features.device
    )
    action_ids, legal_mask = JEVDecisionController._action_tensor(
        legal_actions, batch_size, state_features.device
    )
    for row, question in enumerate(questions_tensor.tolist()):
        _validate_actions(question, action_ids[row][legal_mask[row]].tolist())
    return state_features, questions_tensor, action_ids, legal_mask


def _result(logits: Tensor, action_ids: Tensor, legal_mask: Tensor) -> Dict[str, Tensor]:
    logits = logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
    return {
        "logits": logits,
        "probs": torch.softmax(logits, dim=-1),
        "legal_actions": action_ids,
        "legal_mask": legal_mask,
    }


def _action_scores(
    state: Tensor,
    action_ids: Tensor,
    legal_mask: Tensor,
    score_indices: Mapping[str, int],
) -> Tensor:
    """Return a score for each semantic action slot.

    Scores for actions without an explicitly meaningful state feature remain
    zero; the typed legal mask decides whether they can be emitted.
    """

    scores = state.new_zeros((state.shape[0], action_ids.shape[1]))
    for name, column in score_indices.items():
        if name not in ACTION_TO_INDEX:
            raise ValueError(f"unknown score action: {name}")
        if column < 0 or column >= state.shape[1]:
            raise ValueError(f"score column out of range for {name}: {column}")
        scores = torch.where(
            (action_ids == ACTION_TO_INDEX[name]) & legal_mask,
            state[:, column : column + 1].expand_as(scores),
            scores,
        )
    return scores


class FixedThresholdPolicy(nn.Module):
    """The original-style fixed threshold decision rule.

    The rule has no learned parameters.  For MATCH it accepts the current
    proposal when its score clears the threshold, otherwise tries the supplied
    reassociation score if legal, and finally starts a new track.  MEMORY and
    REACTIVATION are binary threshold gates.  Legal action order never changes
    the semantic choice.
    """

    DEFAULT_SCORE_INDICES = {
        "ACCEPT_CURRENT": 0,
        "REASSOCIATE": 1,
        "WRITE_MEMORY": 2,
        "REACTIVATE_OLD": 3,
    }

    def __init__(
        self,
        threshold: float = 0.5,
        score_indices: Mapping[str, int] = DEFAULT_SCORE_INDICES,
    ):
        super().__init__()
        self.threshold = float(threshold)
        self.score_indices = dict(score_indices)

    def forward(self, state_features, questions, legal_actions):
        state, q, action_ids, legal_mask = _prepare_inputs(
            state_features, questions, legal_actions
        )
        chosen = []
        for row, question in enumerate(q.tolist()):
            names = [
                ACTION_NAMES[index]
                for index in action_ids[row][legal_mask[row]].tolist()
            ]
            values = {
                name: float(
                    state[row, self.score_indices[name]].detach().item()
                )
                for name in self.score_indices
                if name in names
            }
            q_name = QUESTION_NAMES[question]
            if q_name == "MATCH_DECISION":
                if values.get("ACCEPT_CURRENT", float("-inf")) > self.threshold:
                    preferred = "ACCEPT_CURRENT"
                elif values.get("REASSOCIATE", float("-inf")) > self.threshold:
                    preferred = "REASSOCIATE"
                else:
                    preferred = "START_NEW"
            elif q_name == "MEMORY_DECISION":
                preferred = (
                    "WRITE_MEMORY"
                    if values.get("WRITE_MEMORY", float("-inf")) > self.threshold
                    else "SKIP_MEMORY"
                )
            else:
                preferred = (
                    "REACTIVATE_OLD"
                    if values.get("REACTIVATE_OLD", float("-inf")) > self.threshold
                    else "START_NEW"
                )
            if preferred not in names:
                # A runtime legal mask may remove a branch.  Select the first
                # legal action rather than silently inventing one.
                preferred = names[0]
            chosen.append(ACTION_TO_INDEX[preferred])

        logits = state.new_full(action_ids.shape, -20.0)
        for row, action in enumerate(chosen):
            logits[row] = torch.where(
                action_ids[row] == action,
                state.new_tensor(20.0),
                logits[row],
            )
        return _result(logits, action_ids, legal_mask)


class GlobalLearnedThreshold(nn.Module):
    """One learned scalar threshold shared by all typed questions."""

    def __init__(
        self,
        initial_threshold: float = 0.5,
        score_indices: Mapping[str, int] = FixedThresholdPolicy.DEFAULT_SCORE_INDICES,
    ):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(float(initial_threshold)))
        self.score_indices = dict(score_indices)

    def forward(self, state_features, questions, legal_actions):
        state, q, action_ids, legal_mask = _prepare_inputs(
            state_features, questions, legal_actions
        )
        scores = _action_scores(state, action_ids, legal_mask, self.score_indices)
        threshold = self.threshold.to(dtype=state.dtype)
        logits = scores - threshold
        for row, question in enumerate(q.tolist()):
            if QUESTION_NAMES[question] == "MATCH_DECISION":
                current = state[row, self.score_indices["ACCEPT_CURRENT"]]
                reassoc = state[row, self.score_indices["REASSOCIATE"]]
                new_logit = threshold - torch.maximum(current, reassoc)
                logits[row] = torch.where(
                    action_ids[row] == ACTION_TO_INDEX["START_NEW"],
                    new_logit,
                    logits[row],
                )
        return _result(logits, action_ids, legal_mask)


class StateConditionedThreshold(nn.Module):
    """Threshold baseline whose scalar cutoff is conditioned on state."""

    def __init__(
        self,
        state_dim: int,
        initial_threshold: float = 0.5,
        score_indices: Mapping[str, int] = FixedThresholdPolicy.DEFAULT_SCORE_INDICES,
    ):
        super().__init__()
        self.base_threshold = nn.Parameter(torch.tensor(float(initial_threshold)))
        self.conditioner = nn.Linear(state_dim, 1)
        nn.init.zeros_(self.conditioner.weight)
        nn.init.zeros_(self.conditioner.bias)
        self.score_indices = dict(score_indices)

    def forward(self, state_features, questions, legal_actions):
        state, q, action_ids, legal_mask = _prepare_inputs(
            state_features, questions, legal_actions
        )
        scores = _action_scores(state, action_ids, legal_mask, self.score_indices)
        threshold = self.base_threshold + self.conditioner(state).squeeze(-1)
        logits = scores - threshold[:, None]
        for row, question in enumerate(q.tolist()):
            if QUESTION_NAMES[question] == "MATCH_DECISION":
                current = state[row, self.score_indices["ACCEPT_CURRENT"]]
                reassoc = state[row, self.score_indices["REASSOCIATE"]]
                logits[row] = torch.where(
                    action_ids[row] == ACTION_TO_INDEX["START_NEW"],
                    threshold[row] - torch.maximum(current, reassoc),
                    logits[row],
                )
        return _result(logits, action_ids, legal_mask)


class LogisticGate(nn.Module):
    """Single fixed-slot linear/logistic gate with a runtime legal mask."""

    def __init__(self, state_dim: int):
        super().__init__()
        self.gate = nn.Linear(state_dim, len(ACTION_NAMES))

    def forward(self, state_features, questions, legal_actions):
        state, _, action_ids, legal_mask = _prepare_inputs(
            state_features, questions, legal_actions
        )
        fixed_logits = self.gate(state)
        logits = fixed_logits.gather(1, action_ids.clamp_min(0))
        return _result(logits, action_ids, legal_mask)


class FixedSlotMLP(nn.Module):
    """One MLP emitting six semantic action slots, then masking illegal ones."""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(ACTION_NAMES)),
        )

    def forward(self, state_features, questions, legal_actions):
        state, _, action_ids, legal_mask = _prepare_inputs(
            state_features, questions, legal_actions
        )
        logits = self.network(state).gather(1, action_ids.clamp_min(0))
        return _result(logits, action_ids, legal_mask)


class IndependentMLPHeads(nn.Module):
    """Separate question-specific MLPs; intentionally a strong baseline."""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, len(ACTION_NAMES)),
                )
                for name in QUESTION_NAMES
            }
        )

    def forward(self, state_features, questions, legal_actions):
        state, q, action_ids, legal_mask = _prepare_inputs(
            state_features, questions, legal_actions
        )
        rows = []
        for row, question in enumerate(q.tolist()):
            fixed = self.heads[QUESTION_NAMES[question]](state[row : row + 1])
            rows.append(fixed.gather(1, action_ids[row : row + 1].clamp_min(0)))
        return _result(torch.cat(rows, dim=0), action_ids, legal_mask)


class SharedEncoderSeparateHeads(nn.Module):
    """Shared state encoder with separate question heads."""

    def __init__(self, state_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.encoder = JEVStateEncoder(state_dim, hidden_dim, num_layers)
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, len(ACTION_NAMES)),
                )
                for name in QUESTION_NAMES
            }
        )

    def forward(self, state_features, questions, legal_actions):
        state, q, action_ids, legal_mask = _prepare_inputs(
            state_features, questions, legal_actions
        )
        encoded = self.encoder(state)
        rows = []
        for row, question in enumerate(q.tolist()):
            fixed = self.heads[QUESTION_NAMES[question]](encoded[row : row + 1])
            rows.append(fixed.gather(1, action_ids[row : row + 1].clamp_min(0)))
        return _result(torch.cat(rows, dim=0), action_ids, legal_mask)
