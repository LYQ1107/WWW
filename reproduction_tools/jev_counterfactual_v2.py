"""Cached-perception, mutable-association counterfactual engine.

This module contains no detector, evaluator, or future-GT access.  A frozen
perception payload is shared by all branches; only the copied GMT-like track,
memory, and stale-bank state is mutated.  The production adapter can inject
the GMT association transformer as ``association_fn``.  The built-in cosine
backend is a deterministic contract-test backend, not a replacement for GMT.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import torch

from gtr.modeling.jev_assignment import constrained_hungarian
from gtr.modeling.jev_perception_cache import CACHE_VERSION


ENGINE_VERSION = "cached_perception_mutable_association_v2"


@dataclass
class MutableGMTState:
    """The branch-local mutable part of GMT state."""

    next_id: int = 0
    active_ids: Set[int] = field(default_factory=set)
    stale_ids: Set[int] = field(default_factory=set)
    track_embeddings: Dict[int, torch.Tensor] = field(default_factory=dict)
    track_hits: Dict[int, int] = field(default_factory=dict)
    memory: Dict[int, List[torch.Tensor]] = field(default_factory=dict)
    assignments: Dict[str, int] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=dict)
    association_history: List[Mapping[str, object]] = field(default_factory=list)

    def clone(self) -> "MutableGMTState":
        """Copy mutable tracker state while sharing frozen perception payloads.

        Counterfactual branches are intentionally isolated, but cache payloads
        are immutable evidence.  Deep-copying the recent history for every
        legal action duplicated large CPU tensors and dominated shard runtime;
        only the assignments and mutable GMT containers need branch-local
        copies.
        """
        history = []
        for item in self.association_history:
            copied = {
                key: (value if key == "perception" else copy.deepcopy(value))
                for key, value in item.items()
            }
            if "assignments" in item:
                copied["assignments"] = dict(item["assignments"])
            history.append(copied)
        return MutableGMTState(
            next_id=int(self.next_id),
            active_ids=set(self.active_ids),
            stale_ids=set(self.stale_ids),
            track_embeddings={
                int(key): value.detach().clone()
                for key, value in self.track_embeddings.items()
            },
            track_hits={int(key): int(value) for key, value in self.track_hits.items()},
            memory={
                int(key): [value.detach().clone() for value in values]
                for key, values in self.memory.items()
            },
            assignments=dict(self.assignments),
            counters={str(key): int(value) for key, value in self.counters.items()},
            association_history=history,
        )


@dataclass(frozen=True)
class AssociationProposal:
    track_ids: Tuple[int, ...]
    scores: torch.Tensor
    pairs: Mapping[int, int]
    banned_edges: Tuple[Tuple[int, int], ...] = ()


def _normalise_features(features: torch.Tensor) -> torch.Tensor:
    features = torch.as_tensor(features, dtype=torch.float32)
    features = features.reshape(features.shape[0], -1)
    if features.shape[1] == 0:
        return features
    return features / features.norm(dim=1, keepdim=True).clamp_min(1e-8)


class CachedPerceptionMutableAssociationV2:
    """Rerun association/state dynamics while reusing frozen perception."""

    def __init__(
        self,
        *,
        association_fn: Optional[
            Callable[[Mapping[str, object], Sequence[int], MutableGMTState], torch.Tensor]
        ] = None,
        acceptance_threshold: float = 0.2,
        history_limit: int = 16,
    ):
        self.association_fn = association_fn
        self.acceptance_threshold = float(acceptance_threshold)
        self.history_limit = max(1, int(history_limit))

    @staticmethod
    def _track_ids(state: MutableGMTState) -> Tuple[int, ...]:
        return tuple(sorted(int(value) for value in state.active_ids))

    def _cosine_scores(
        self,
        perception: Mapping[str, object],
        track_ids: Sequence[int],
        state: MutableGMTState,
    ) -> torch.Tensor:
        detections = _normalise_features(
            torch.as_tensor(perception["reid_features"], dtype=torch.float32)
        )
        if not track_ids:
            return detections.new_empty((detections.shape[0], 0))
        if detections.shape[1] == 0:
            return detections.new_zeros((detections.shape[0], len(track_ids)))
        prototypes = []
        for track_id in track_ids:
            prototype = state.track_embeddings.get(track_id)
            if prototype is None:
                prototype = detections.new_zeros((detections.shape[1],))
            prototypes.append(torch.as_tensor(prototype, dtype=torch.float32).reshape(-1))
        prototype_tensor = torch.stack(prototypes, dim=0)
        if prototype_tensor.shape[1] != detections.shape[1]:
            raise ValueError("cached ReID dimension does not match mutable track state")
        prototype_tensor = _normalise_features(prototype_tensor)
        return detections @ prototype_tensor.t()

    def score_matrix(
        self,
        perception: Mapping[str, object],
        state: MutableGMTState,
    ) -> Tuple[Tuple[int, ...], torch.Tensor]:
        track_ids = self._track_ids(state)
        if self.association_fn is None:
            scores = self._cosine_scores(perception, track_ids, state)
        else:
            scores = self.association_fn(perception, track_ids, state)
            scores = torch.as_tensor(scores, dtype=torch.float32)
        detection_count = int(torch.as_tensor(perception["pred_boxes"]).shape[0])
        expected = (detection_count, len(track_ids))
        if tuple(scores.shape) != expected:
            raise ValueError(f"association backend returned {tuple(scores.shape)}, expected {expected}")
        if not torch.isfinite(scores).all():
            raise ValueError("association backend returned non-finite scores")
        return track_ids, scores

    def propose(
        self,
        perception: Mapping[str, object],
        state: MutableGMTState,
        *,
        banned_edges: Iterable[Tuple[int, int]] = (),
    ) -> AssociationProposal:
        track_ids, scores = self.score_matrix(perception, state)
        banned = tuple(sorted((int(row), int(col)) for row, col in banned_edges))
        pairs = dict(constrained_hungarian(scores, banned))
        return AssociationProposal(track_ids, scores, pairs, banned)

    @staticmethod
    def _new_id(state: MutableGMTState) -> int:
        state.next_id = max(int(state.next_id), *(state.active_ids or {0})) + 1
        state.active_ids.add(state.next_id)
        state.track_hits.setdefault(state.next_id, 0)
        state.counters["new_ids"] = state.counters.get("new_ids", 0) + 1
        return state.next_id

    @staticmethod
    def _update_track(
        state: MutableGMTState,
        track_id: int,
        feature: torch.Tensor,
        *,
        write_memory: bool = False,
    ) -> None:
        feature = torch.as_tensor(feature, dtype=torch.float32).detach().cpu().clone()
        if feature.numel():
            previous = state.track_embeddings.get(track_id)
            hits = max(1, int(state.track_hits.get(track_id, 0)))
            state.track_embeddings[track_id] = (
                feature if previous is None else (previous * hits + feature) / (hits + 1)
            )
        state.active_ids.add(int(track_id))
        state.track_hits[int(track_id)] = int(state.track_hits.get(int(track_id), 0)) + 1
        if write_memory:
            state.memory.setdefault(int(track_id), []).append(feature)
            state.counters["memory_writes"] = state.counters.get("memory_writes", 0) + 1

    def _actions_for_proposal(
        self,
        proposal: AssociationProposal,
        *,
        threshold: Optional[float] = None,
    ) -> Dict[int, str]:
        threshold = self.acceptance_threshold if threshold is None else float(threshold)
        actions = {}
        for row in range(proposal.scores.shape[0]):
            col = proposal.pairs.get(row)
            if col is None or float(proposal.scores[row, col]) <= threshold:
                actions[row] = "START_NEW"
            else:
                actions[row] = "ACCEPT_CURRENT"
        return actions

    def step(
        self,
        perception: Mapping[str, object],
        state: MutableGMTState,
        *,
        actions: Optional[Mapping[int, str]] = None,
        memory_actions: Optional[Mapping[int, str]] = None,
        threshold: Optional[float] = None,
    ) -> Mapping[str, object]:
        """Commit one frame/view from an isolated mutable state.

        If one or more rows request REASSOCIATE, all rejected edges are
        masked and the complete matrix is solved once.  The result is then
        consumed by the semantic actions; no ranked-candidate fallback exists.
        """
        proposal = self.propose(perception, state)
        action_map = dict(actions or self._actions_for_proposal(proposal, threshold=threshold))
        for row in range(proposal.scores.shape[0]):
            action_map.setdefault(row, "START_NEW")
        reassociate_rows = [
            row for row, action in action_map.items() if action == "REASSOCIATE"
        ]
        second = proposal
        if reassociate_rows:
            banned = set(proposal.banned_edges)
            for row, action in action_map.items():
                if action == "START_NEW":
                    banned.update((row, col) for col in range(proposal.scores.shape[1]))
            for row in reassociate_rows:
                current = proposal.pairs.get(row)
                if current is not None:
                    banned.add((row, current))
            second = self.propose(perception, state, banned_edges=banned)
            state.counters["reassociation_calls"] = state.counters.get("reassociation_calls", 0) + 1

        features = _normalise_features(
            torch.as_tensor(perception["reid_features"], dtype=torch.float32)
        )
        committed: Dict[int, int] = {}
        for row in range(features.shape[0]):
            action = action_map[row]
            col = second.pairs.get(row)
            if action == "START_NEW" or col is None:
                track_id = self._new_id(state)
            else:
                track_id = int(second.track_ids[col])
                if action == "REASSOCIATE":
                    state.counters["reassociated_rows"] = state.counters.get("reassociated_rows", 0) + 1
            committed[row] = track_id
            self._update_track(
                state,
                track_id,
                features[row] if features.shape[1] else torch.empty(0),
                write_memory=False,
            )

        if memory_actions:
            for row, action in memory_actions.items():
                if action != "WRITE_MEMORY" or row not in committed:
                    continue
                track_id = committed[row]
                feature = features[row] if features.shape[1] else torch.empty(0)
                state.memory.setdefault(track_id, []).append(feature.detach().cpu().clone())
                state.counters["memory_writes"] = state.counters.get("memory_writes", 0) + 1

        history_item = {
            # FrozenPerceptionCache payloads are immutable evidence. Sharing
            # the mapping is exact: branch cloning already preserves the
            # perception object by identity, and all mutable state lives in
            # assignments/track containers. Deep-copying every cached payload
            # at every replay step dominated formal H=8 runtime.
            "perception": perception,
            "assignments": dict(committed),
        }
        state.association_history.append(history_item)
        if len(state.association_history) > self.history_limit:
            del state.association_history[:-self.history_limit]

        return {
            "engine_version": ENGINE_VERSION,
            "cache_version": str(perception.get("cache_version", CACHE_VERSION)),
            "initial_pairs": dict(proposal.pairs),
            "initial_track_ids": list(proposal.track_ids),
            "final_pairs": dict(second.pairs),
            "final_track_ids": list(second.track_ids),
            "committed_track_ids": committed,
            "initial_scores": proposal.scores.detach().cpu(),
            "final_scores": second.scores.detach().cpu(),
            "banned_edges": list(second.banned_edges),
        }

    def rollout(
        self,
        perceptions: Sequence[Mapping[str, object]],
        initial_state: MutableGMTState,
        *,
        action_provider: Optional[
            Callable[[int, AssociationProposal, MutableGMTState], Mapping[int, str]]
        ] = None,
    ) -> Tuple[MutableGMTState, List[Mapping[str, object]]]:
        """Roll out future cached perception using a branch-local state copy."""
        branch = initial_state.clone()
        steps = []
        for index, perception in enumerate(perceptions):
            proposal = self.propose(perception, branch)
            actions = action_provider(index, proposal, branch) if action_provider else None
            steps.append(self.step(perception, branch, actions=actions))
        return branch, steps


def run_counterfactual_branches(
    engine: CachedPerceptionMutableAssociationV2,
    perceptions: Sequence[Mapping[str, object]],
    initial_state: MutableGMTState,
    action_plans: Mapping[str, Mapping[int, Mapping[int, str]]],
) -> Mapping[str, Tuple[MutableGMTState, List[Mapping[str, object]]]]:
    """Run named action branches from one unchanged pre-action state."""
    source = initial_state.clone()
    source_signature = _state_signature(source)
    results = {}
    for name, plan in action_plans.items():
        branch = source.clone()
        steps = []
        for index, perception in enumerate(perceptions):
            steps.append(engine.step(perception, branch, actions=plan.get(index)))
        results[name] = (branch, steps)
        if _state_signature(initial_state) != source_signature:
            raise AssertionError(f"counterfactual branch mutated source state: {name}")
    return results


def _state_signature(state: MutableGMTState):
    embeddings = tuple(
        (int(key), tuple(float(value) for value in tensor.reshape(-1).tolist()))
        for key, tensor in sorted(state.track_embeddings.items())
    )
    memory = tuple(
        (
            int(key),
            tuple(tuple(float(value) for value in item.reshape(-1).tolist()) for item in values),
        )
        for key, values in sorted(state.memory.items())
    )
    return (
        int(state.next_id),
        tuple(sorted(int(value) for value in state.active_ids)),
        tuple(sorted(int(value) for value in state.stale_ids)),
        embeddings,
        tuple(sorted((int(key), int(value)) for key, value in state.track_hits.items())),
        memory,
        tuple(sorted((str(key), int(value)) for key, value in state.counters.items())),
    )
