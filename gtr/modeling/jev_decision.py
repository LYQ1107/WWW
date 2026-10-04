"""Typed state-transition decision primitives for GMT/JEV experiments.

This module deliberately does not know about detector proposals, track IDs, or
ground truth.  It only maps an online state representation, a typed question,
and the question's runtime legal action set to a masked action distribution.
The GMT adapter remains responsible for proposing and committing identities.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


QUESTION_NAMES: Tuple[str, ...] = (
    "MATCH_DECISION",
    "MEMORY_DECISION",
    "REACTIVATION_DECISION",
)

ACTION_NAMES: Tuple[str, ...] = (
    "ACCEPT_CURRENT",
    "REASSOCIATE",
    "START_NEW",
    "WRITE_MEMORY",
    "SKIP_MEMORY",
    "REACTIVATE_OLD",
)

QUESTION_TO_INDEX = {name: index for index, name in enumerate(QUESTION_NAMES)}
ACTION_TO_INDEX = {name: index for index, name in enumerate(ACTION_NAMES)}

QUESTION_LEGAL_ACTIONS: Mapping[str, Tuple[str, ...]] = {
    "MATCH_DECISION": (
        "ACCEPT_CURRENT",
        "REASSOCIATE",
        "START_NEW",
    ),
    "MEMORY_DECISION": (
        "WRITE_MEMORY",
        "SKIP_MEMORY",
    ),
    "REACTIVATION_DECISION": (
        "REACTIVATE_OLD",
        "START_NEW",
    ),
}


ActionLike = Union[str, int]
QuestionLike = Union[str, int]


@dataclass(frozen=True)
class ActionSpec:
    """A semantic action token; it is not a candidate identity."""

    name: str
    index: int


@dataclass
class TrackingDecisionState:
    """Transport object for an online decision.

    ``features`` must contain only information available at the current time.
    In particular, this class has no GT/future-frame field by design.
    """

    features: Tensor
    question: QuestionLike
    legal_actions: Sequence[ActionLike]


def question_index(question: QuestionLike) -> int:
    if isinstance(question, str):
        if question not in QUESTION_TO_INDEX:
            raise ValueError(f"Unknown JEV question: {question}")
        return QUESTION_TO_INDEX[question]
    question = int(question)
    if question < 0 or question >= len(QUESTION_NAMES):
        raise ValueError(f"Question index out of range: {question}")
    return question


def action_index(action: ActionLike) -> int:
    if isinstance(action, str):
        if action not in ACTION_TO_INDEX:
            raise ValueError(f"Unknown JEV action: {action}")
        return ACTION_TO_INDEX[action]
    action = int(action)
    if action < 0 or action >= len(ACTION_NAMES):
        raise ValueError(f"Action index out of range: {action}")
    return action


def legal_actions_for(
    question: QuestionLike,
    *,
    can_reassociate: bool = True,
    has_old_track: bool = True,
) -> List[str]:
    """Build a deterministic runtime legal action set.

    The state can remove actions, but it cannot add an action belonging to a
    different typed question.  In particular, reactivation is unavailable
    when no valid stale identity exists.
    """

    q_name = QUESTION_NAMES[question_index(question)]
    actions = list(QUESTION_LEGAL_ACTIONS[q_name])
    if q_name == "MATCH_DECISION" and not can_reassociate:
        actions.remove("REASSOCIATE")
    if q_name == "REACTIVATION_DECISION" and not has_old_track:
        actions.remove("REACTIVATE_OLD")
    if not actions:
        raise ValueError(f"No legal actions remain for {q_name}")
    return actions


def _validate_actions(question: int, actions: Sequence[int]) -> None:
    q_name = QUESTION_NAMES[question]
    allowed = {ACTION_TO_INDEX[name] for name in QUESTION_LEGAL_ACTIONS[q_name]}
    if len(set(actions)) != len(actions):
        raise ValueError("JEV legal action set contains duplicates")
    unknown = set(actions) - allowed
    if unknown:
        names = [ACTION_NAMES[index] for index in sorted(unknown)]
        raise ValueError(f"Illegal actions for {q_name}: {names}")


class JEVStateEncoder(nn.Module):
    """Small shared encoder for current online tracking state features."""

    def __init__(self, state_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        layers: List[nn.Module] = []
        in_dim = state_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                ]
            )
            in_dim = hidden_dim
        self.network = nn.Sequential(*layers)

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features)


class JEVDecisionController(nn.Module):
    """Shared typed/action-conditioned scorer with runtime masking.

    The output dimension is the number of supplied legal actions, not a fixed
    candidate-ID classifier.  Action embeddings are semantic tokens shared by
    MATCH, MEMORY, and REACTIVATION questions.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 128,
        question_dim: int = 32,
        action_dim: int = 32,
        num_layers: int = 2,
        temperature: float = 1.0,
        use_option_interaction: bool = False,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if use_option_interaction and action_dim % 4 != 0:
            raise ValueError("action_dim must be divisible by four for interaction")
        self.temperature = float(temperature)
        self.state_encoder = JEVStateEncoder(state_dim, hidden_dim, num_layers)
        self.question_embedding = nn.Embedding(len(QUESTION_NAMES), question_dim)
        self.action_embedding = nn.Embedding(len(ACTION_NAMES), action_dim)
        self.query = nn.Sequential(
            nn.Linear(hidden_dim + question_dim, action_dim),
            nn.LayerNorm(action_dim),
            nn.GELU(),
            nn.Linear(action_dim, action_dim),
        )
        self.key = nn.Sequential(
            nn.Linear(action_dim, action_dim),
            nn.LayerNorm(action_dim),
            nn.GELU(),
            nn.Linear(action_dim, action_dim),
        )
        self.use_option_interaction = bool(use_option_interaction)
        if self.use_option_interaction:
            layer = nn.TransformerEncoderLayer(
                d_model=action_dim,
                nhead=4,
                dim_feedforward=action_dim * 2,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.option_interaction = nn.TransformerEncoder(layer, num_layers=1)
        else:
            self.option_interaction = None

    @staticmethod
    def _questions_tensor(
        questions: Union[QuestionLike, Sequence[QuestionLike], Tensor],
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        if isinstance(questions, Tensor):
            values = questions.to(device=device, dtype=torch.long).view(-1)
        elif isinstance(questions, (str, int)):
            values = torch.tensor([question_index(questions)], device=device)
        else:
            values = torch.tensor(
                [question_index(value) for value in questions], device=device
            )
        if values.numel() == 1 and batch_size != 1:
            values = values.expand(batch_size)
        if values.numel() != batch_size:
            raise ValueError("Question batch size does not match state batch size")
        return values

    @staticmethod
    def _action_tensor(
        legal_actions: Union[Tensor, Sequence[ActionLike], Sequence[Sequence[ActionLike]]],
        batch_size: int,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        if isinstance(legal_actions, Tensor):
            ids = legal_actions.to(device=device, dtype=torch.long)
            if ids.ndim == 1:
                ids = ids.unsqueeze(0)
            if ids.ndim != 2 or ids.shape[0] != batch_size:
                raise ValueError("legal_actions tensor must have shape [B, K]")
            valid = ids >= 0
            if not torch.all(valid.any(dim=1)):
                raise ValueError("Every decision must have at least one legal action")
            if torch.any(ids[valid] >= len(ACTION_NAMES)):
                raise ValueError("legal_actions contains an unknown action index")
            return ids, valid

        if batch_size == 1 and (
            not legal_actions or isinstance(legal_actions[0], (str, int))
        ):
            rows = [legal_actions]  # type: ignore[list-item]
        else:
            rows = list(legal_actions)  # type: ignore[arg-type]
        if len(rows) != batch_size:
            raise ValueError("legal_actions batch size does not match state batch size")
        encoded = [[action_index(action) for action in row] for row in rows]
        if any(len(row) == 0 for row in encoded):
            raise ValueError("Every decision must have at least one legal action")
        width = max(len(row) for row in encoded)
        ids = torch.full((batch_size, width), -1, dtype=torch.long, device=device)
        valid = torch.zeros_like(ids, dtype=torch.bool)
        for row_index, row in enumerate(encoded):
            ids[row_index, : len(row)] = torch.tensor(row, device=device)
            valid[row_index, : len(row)] = True
        return ids, valid

    def forward(
        self,
        state_features: Tensor,
        questions: Union[QuestionLike, Sequence[QuestionLike], Tensor],
        legal_actions: Union[
            Tensor,
            Sequence[ActionLike],
            Sequence[Sequence[ActionLike]],
        ],
    ) -> Dict[str, Tensor]:
        if state_features.ndim == 1:
            state_features = state_features.unsqueeze(0)
        if state_features.ndim != 2:
            raise ValueError("state_features must have shape [B, state_dim]")
        batch_size = state_features.shape[0]
        state_features = state_features.float()
        questions_tensor = self._questions_tensor(
            questions, batch_size, state_features.device
        )
        action_ids, legal_mask = self._action_tensor(
            legal_actions, batch_size, state_features.device
        )
        for row, question in enumerate(questions_tensor.tolist()):
            row_actions = action_ids[row][legal_mask[row]].tolist()
            _validate_actions(question, row_actions)

        state = self.state_encoder(state_features)
        question_state = self.question_embedding(questions_tensor)
        query = self.query(torch.cat([state, question_state], dim=-1))
        options = self.action_embedding(action_ids.clamp_min(0))
        if self.option_interaction is not None:
            options = self.option_interaction(options, src_key_padding_mask=~legal_mask)
        keys = self.key(options)
        logits = torch.sum(query.unsqueeze(1) * keys, dim=-1)
        logits = logits / (self.temperature * (keys.shape[-1] ** 0.5))
        logits = logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
        probs = F.softmax(logits, dim=-1)
        return {
            "logits": logits,
            "probs": probs,
            "legal_actions": action_ids,
            "legal_mask": legal_mask,
        }

    @torch.no_grad()
    def decide(
        self,
        state_features: Tensor,
        question: Union[QuestionLike, Sequence[QuestionLike], Tensor],
        legal_actions: Union[
            Tensor,
            Sequence[ActionLike],
            Sequence[Sequence[ActionLike]],
        ],
    ) -> Dict[str, object]:
        result = self(state_features, question, legal_actions)
        choice = result["probs"].argmax(dim=-1)
        chosen_ids = result["legal_actions"].gather(1, choice[:, None]).squeeze(1)
        result["chosen_action_ids"] = chosen_ids
        result["chosen_actions"] = [ACTION_NAMES[index] for index in chosen_ids.tolist()]
        return result
