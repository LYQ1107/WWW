"""Online-safe JEV policy adapter and trace writer.

The adapter owns no GMT state and has no evaluator/GT dependency.  It chooses
an action proposal; only the caller's explicit commit callback is allowed to
mutate track or memory state.  ``oracle`` is rejected here by design and is
implemented only in offline labeler code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import hashlib
import json
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from .jev_baselines import (
    ActionConditionedScorerNoQuestion,
    FixedThresholdPolicy,
    FixedSlotMLP,
    GlobalLearnedThreshold,
    IndependentMLPHeads,
    LogisticGate,
    NonlinearStateConditionedThreshold,
    QuestionConditionedFixedHead,
    QuestionConditionedMLP,
    QuestionConditionedThreshold,
    SharedEncoderSeparateHeads,
    StateConditionedThreshold,
)
from .jev_decision import ACTION_NAMES, JEVDecisionController, action_index, question_index


ONLINE_MODES = ("off", "shadow", "jev")
CONTROLLER_MODELS = {
    "fixed_threshold",
    "jev",
    "fixed_slot_mlp",
    "independent_mlp",
    "shared_heads",
    "logistic",
    "global_threshold",
    "state_threshold",
    "nonlinear_state_threshold",
    "question_threshold",
    "question_conditioned_mlp",
    "action_conditioned_no_question",
    "question_conditioned_fixed_head",
}
QuestionLike = Union[str, int]
ActionLike = Union[str, int]


class SeedEnsembleController(nn.Module):
    """Average policy probabilities over the complete preselected seed set."""

    def __init__(self, controllers: Sequence[nn.Module]):
        super().__init__()
        if not controllers:
            raise ValueError("seed ensemble must contain at least one controller")
        self.controllers = nn.ModuleList(controllers)

    def forward(self, state_features, questions, legal_actions):
        outputs = [
            controller(state_features, questions, legal_actions)
            for controller in self.controllers
        ]
        reference = outputs[0]
        for output in outputs[1:]:
            if output["probs"].shape != reference["probs"].shape:
                raise ValueError("seed ensemble members returned different policy shapes")
            if not torch.equal(output["legal_actions"], reference["legal_actions"]):
                raise ValueError("seed ensemble members disagree on legal action IDs")
            if not torch.equal(output["legal_mask"], reference["legal_mask"]):
                raise ValueError("seed ensemble members disagree on legal action masks")
        result = dict(reference)
        result["probs"] = torch.stack(
            [output["probs"] for output in outputs], dim=0
        ).mean(dim=0)
        return result


class TemperatureScaledController(nn.Module):
    """Apply validation-only temperature scaling to any policy family."""

    def __init__(self, controller: nn.Module, temperature: float):
        super().__init__()
        if not float(temperature) > 0:
            raise ValueError("temperature must be positive")
        self.controller = controller
        self.temperature = float(temperature)

    def forward(self, state_features, questions, legal_actions):
        output = self.controller(state_features, questions, legal_actions)
        if self.temperature == 1.0:
            return output
        logits = output["logits"] / self.temperature
        logits = logits.masked_fill(
            ~output["legal_mask"], torch.finfo(logits.dtype).min
        )
        result = dict(output)
        result["logits"] = logits
        result["probs"] = torch.softmax(logits, dim=-1)
        return result


def build_controller_from_checkpoint(
    checkpoint: Union[str, Path],
    *,
    device: Union[str, torch.device] = "cpu",
) -> nn.Module:
    """Rebuild an offline-trained JEV/baseline policy for online use.

    The checkpoint is expected to be produced by ``train_jev.py``.  Loading a
    policy is intentionally explicit: the runtime never silently falls back
    to a randomly initialized controller.
    """

    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = torch.load(str(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise ValueError("JEV controller checkpoint must contain a mapping")
    if payload.get("checkpoint_type") == "seed_ensemble":
        members = payload.get("members")
        if not isinstance(members, Sequence) or isinstance(members, (str, bytes)) or not members:
            raise ValueError("seed ensemble checkpoint has no members")
        controllers = []
        member_model_names = set()
        for member in members:
            if not isinstance(member, Mapping) or not member.get("path"):
                raise ValueError("seed ensemble member must contain a path")
            member_path = Path(str(member["path"]))
            if not member_path.is_absolute():
                member_path = path.parent / member_path
            if not member_path.is_file():
                raise FileNotFoundError(member_path)
            expected_digest = member.get("sha256")
            if expected_digest:
                digest = hashlib.sha256(member_path.read_bytes()).hexdigest()
                if str(expected_digest) not in {digest, "sha256:" + digest}:
                    raise ValueError(f"seed ensemble member digest mismatch: {member_path}")
            member_payload = torch.load(str(member_path), map_location="cpu")
            if not isinstance(member_payload, Mapping):
                raise ValueError(f"invalid seed ensemble member: {member_path}")
            member_model_names.add(str(member_payload.get("model_name", "jev")))
            controllers.append(build_controller_from_checkpoint(member_path, device=device))
        expected_model = str(payload.get("model_name", ""))
        if len(member_model_names) != 1 or (expected_model and expected_model not in member_model_names):
            raise ValueError("seed ensemble members do not share the declared model")
        return SeedEnsembleController(controllers).to(device).eval()
    model_name = str(payload.get("model_name", "jev"))
    if model_name not in CONTROLLER_MODELS:
        raise ValueError(f"unsupported controller model in checkpoint: {model_name}")
    state_dim = int(payload.get("state_dim", 64))
    hidden_dim = int(payload.get("hidden_dim", 128))
    if state_dim < 4 or hidden_dim < 1:
        raise ValueError("invalid controller dimensions in checkpoint")
    if model_name == "jev":
        model = JEVDecisionController(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            question_dim=int(payload.get("question_dim", max(8, hidden_dim // 4))),
            action_dim=int(payload.get("action_dim", max(8, hidden_dim // 4))),
            num_layers=int(payload.get("num_layers", 2)),
            temperature=float(payload.get("temperature", 1.0)),
            use_option_interaction=bool(payload.get("use_option_interaction", False)),
        )
    elif model_name == "fixed_threshold":
        model = FixedThresholdPolicy(
            threshold=float(payload.get("threshold", 0.2))
        )
    elif model_name == "fixed_slot_mlp":
        model = FixedSlotMLP(state_dim, hidden_dim)
    elif model_name == "independent_mlp":
        model = IndependentMLPHeads(state_dim, hidden_dim)
    elif model_name == "shared_heads":
        model = SharedEncoderSeparateHeads(state_dim, hidden_dim)
    elif model_name == "logistic":
        model = LogisticGate(state_dim)
    elif model_name == "global_threshold":
        model = GlobalLearnedThreshold()
    elif model_name == "state_threshold":
        model = StateConditionedThreshold(state_dim)
    elif model_name == "nonlinear_state_threshold":
        model = NonlinearStateConditionedThreshold(state_dim, hidden_dim=hidden_dim)
    elif model_name == "question_threshold":
        model = QuestionConditionedThreshold(
            state_dim,
            hidden_dim=hidden_dim,
            question_dim=int(payload.get("question_dim", max(8, hidden_dim // 4))),
        )
    elif model_name == "question_conditioned_mlp":
        model = QuestionConditionedMLP(
            state_dim,
            hidden_dim=hidden_dim,
            question_dim=int(payload.get("question_dim", max(8, hidden_dim // 4))),
        )
    elif model_name == "action_conditioned_no_question":
        model = ActionConditionedScorerNoQuestion(
            state_dim,
            hidden_dim=hidden_dim,
            action_dim=int(payload.get("action_dim", max(8, hidden_dim // 4))),
        )
    elif model_name == "question_conditioned_fixed_head":
        model = QuestionConditionedFixedHead(
            state_dim,
            hidden_dim=hidden_dim,
            question_dim=int(payload.get("question_dim", max(8, hidden_dim // 4))),
        )
    else:
        raise ValueError(f"unsupported controller model in checkpoint: {model_name}")
    state = payload.get("model")
    if not isinstance(state, Mapping):
        # Permit a raw state_dict only when the caller explicitly labels it as
        # a JEV controller; arbitrary weights must not be guessed at runtime.
        if all(isinstance(key, str) for key in payload.keys()):
            state = payload
        else:
            raise ValueError("controller checkpoint has no model state_dict")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"controller checkpoint mismatch: missing={list(missing)}, "
            f"unexpected={list(unexpected)}"
        )
    model = model.to(device).eval()
    calibration_temperature = payload.get("temperature")
    if model_name != "jev" and calibration_temperature is not None:
        model = TemperatureScaledController(
            model, float(calibration_temperature)
        ).to(device).eval()
    return model


def _canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RuntimeDecision:
    question: str
    legal_actions: Tuple[str, ...]
    off_action: Optional[str]
    proposed_action: str
    committed_action: str
    probabilities: Mapping[str, float]
    state_digest: str
    mode: str
    state_features: Tuple[float, ...] = ()
    context: Mapping[str, object] = field(default_factory=dict)


class DecisionTraceWriter:
    """Append-only JSONL writer for online/shadow decision traces.

    A ``.gz`` suffix enables lossless streaming compression.  This is useful
    for long official online runs whose audit trace is much larger than the
    prediction artifact; plain ``.jsonl`` behavior remains unchanged.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        opener = gzip.open if self.path.suffix == ".gz" else open
        if self.path.exists():
            with opener(self.path, "rt", encoding="utf-8") as existing:
                self._event_order = sum(1 for line in existing if line.strip())
        else:
            self._event_order = 0
        self.handle = opener(self.path, "at", encoding="utf-8")

    def write(self, decision: RuntimeDecision) -> None:
        context = dict(decision.context)
        context.setdefault("event_order", self._event_order)
        self._event_order += 1
        payload = {
            "mode": decision.mode,
            "future_gt_access": False,
            "question": decision.question,
            "legal_actions": list(decision.legal_actions),
            "off_action": decision.off_action,
            "proposed_action": decision.proposed_action,
            "committed_action": decision.committed_action,
            "probabilities": dict(decision.probabilities),
            "state_digest": decision.state_digest,
            "state_feature_vector": list(decision.state_features),
            "context": context,
        }
        self.handle.write(json.dumps(payload, sort_keys=True) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class JEVRuntimePolicy:
    """Select an online action without owning or mutating tracker state."""

    def __init__(
        self,
        mode: str = "off",
        controller: Optional[nn.Module] = None,
        trace_writer: Optional[DecisionTraceWriter] = None,
    ):
        mode = str(mode).lower()
        if mode not in ONLINE_MODES:
            raise ValueError(f"unsupported online JEV mode: {mode}")
        if mode != "off" and controller is None:
            raise ValueError(f"{mode} mode requires a decision controller")
        self.mode = mode
        self.controller = controller.eval() if controller is not None else None
        self.trace_writer = trace_writer

    @staticmethod
    def _validate_action(action: Optional[ActionLike], legal: Sequence[str]) -> Optional[str]:
        if action is None:
            return None
        if isinstance(action, int):
            action = ACTION_NAMES[action_index(action)]
        action = str(action)
        if action not in legal:
            raise ValueError(f"action {action} is not legal in {list(legal)}")
        return action

    def decide(
        self,
        state_features: Tensor,
        question: QuestionLike,
        legal_actions: Sequence[ActionLike],
        *,
        off_action: Optional[ActionLike] = None,
        context: Optional[Mapping[str, object]] = None,
    ) -> RuntimeDecision:
        q_index = question_index(question)
        q_name = ("MATCH_DECISION", "MEMORY_DECISION", "REACTIVATION_DECISION")[q_index]
        legal_names = tuple(
            ACTION_NAMES[action_index(action)] if isinstance(action, int) else str(action)
            for action in legal_actions
        )
        if not legal_names or len(set(legal_names)) != len(legal_names):
            raise ValueError("legal_actions must be a non-empty set")
        old_action = self._validate_action(off_action, legal_names)

        if self.mode == "off":
            if old_action is None:
                raise ValueError("OFF mode requires the original GMT action")
            proposed = old_action
            probabilities = {name: float(name == old_action) for name in legal_names}
        else:
            assert self.controller is not None
            # Detectron2 constructs the controller before the parent GMT
            # model is moved to CUDA.  Keep the checkpoint-owned controller
            # device authoritative instead of assuming the incoming state
            # tensor already lives there.
            try:
                controller_device = next(self.controller.parameters()).device
            except StopIteration:
                controller_device = state_features.device
            controller_features = state_features.to(controller_device)
            with torch.no_grad():
                output = self.controller(controller_features, [q_name], [list(legal_names)])
            probs = output["probs"][0].detach().float().cpu().tolist()
            ids = output["legal_actions"][0].detach().cpu().tolist()
            mask = output["legal_mask"][0].detach().cpu().tolist()
            probabilities = {
                ACTION_NAMES[action_id]: float(probability)
                for action_id, probability, valid in zip(ids, probs, mask)
                if valid
            }
            proposed = max(probabilities, key=probabilities.get)
            if old_action is None:
                old_action = proposed

        committed = old_action if self.mode == "shadow" else proposed
        assert committed is not None
        state_values = state_features.detach().float().cpu().reshape(-1).tolist()
        digest = _canonical_digest(
            {
                "features": state_values,
                "question": q_name,
                "legal_actions": list(legal_names),
                "context": dict(context or {}),
            }
        )
        decision = RuntimeDecision(
            question=q_name,
            legal_actions=legal_names,
            off_action=old_action,
            proposed_action=proposed,
            committed_action=committed,
            probabilities=probabilities,
            state_digest=digest,
            mode=self.mode,
            state_features=tuple(float(value) for value in state_values),
            context=dict(context or {}),
        )
        if self.trace_writer is not None:
            self.trace_writer.write(decision)
        return decision


class JEVCommitAdapter:
    """The only runtime object allowed to apply an action to GMT state."""

    def commit(
        self,
        tracker_state: object,
        decision: RuntimeDecision,
        apply_action: Callable[[object, str], object],
    ) -> object:
        if decision.committed_action not in decision.legal_actions:
            raise ValueError("runtime decision contains an illegal committed action")
        return apply_action(tracker_state, decision.committed_action)
