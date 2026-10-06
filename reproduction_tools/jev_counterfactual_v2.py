"""Cached-perception, mutable-association counterfactual engine.

This module contains no detector, evaluator, or future-GT access.  A frozen
perception payload is shared by all branches; only the copied GMT-like track,
memory, and stale-bank state is mutated.  The production adapter can inject
the GMT association transformer as ``association_fn``.  The built-in cosine
backend is a deterministic contract-test backend, not a replacement for GMT.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import torch

from gtr.modeling.jev_assignment import constrained_hungarian
from gtr.modeling.jev_perception_cache import CACHE_VERSION
from gtr.modeling.jev_state import count_track_history, legacy_acceptance_threshold
from gtr.modeling.roi_heads.transformer import TrajectoryRandom


ENGINE_VERSION = "cached_perception_mutable_association_v2"
TRAJECTORY_RNG_POLICY = "branch_local_explicit_python_random_v1"
TRAJECTORY_RNG_MASTER_SEED = 20261006


def trajectory_rng_seed_for_video(
    video_id: int,
    *,
    master_seed: int = TRAJECTORY_RNG_MASTER_SEED,
) -> int:
    """Derive a stable per-video stream without using process state.

    The formal replay protocol deliberately keeps this mapping simple and
    auditable: changing worker/GPU/PID or action evaluation order cannot change
    a video's initial trajectory RNG stream.
    """

    return int(master_seed) + int(video_id)


def _empty_trajectory_mapping_digest() -> str:
    payload = json.dumps([], separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def proposal_state_for_output(state: object) -> object:
    """Convert Python ``random`` state tuples to JSON-safe provenance."""

    if state is None:
        return None
    return json.loads(json.dumps(state))


def subset_perception_payload(
    payload: Mapping[str, object], rows: Sequence[int]
) -> Mapping[str, object]:
    """Return an immutable-cache payload restricted to detection ``rows``.

    Native GMT's stale-bank pass receives only the detections that failed the
    ordinary association in the current view.  Passing the complete current
    frame changes the transformer query set and therefore changes every stale
    candidate score.  Keep this operation in the shared counterfactual module
    so the formal builder and runtime replay cannot implement different row
    filtering rules.
    """

    selected = [int(row) for row in rows]
    total = int(torch.as_tensor(payload["pred_boxes"]).shape[0])
    if any(row < 0 or row >= total for row in selected):
        raise IndexError(f"payload row is outside detection range: {selected}")
    indices = torch.as_tensor(selected, dtype=torch.long)
    result = dict(payload)
    for name in ("pred_boxes", "detection_scores", "reid_features"):
        if name in payload:
            value = torch.as_tensor(payload[name])
            result[name] = value.index_select(0, indices)
    metadata = dict(payload.get("proposal_metadata", {}) or {})
    for name, value in list(metadata.items()):
        if isinstance(value, list) and len(value) == total:
            metadata[name] = [value[row] for row in selected]
    result["proposal_metadata"] = metadata
    result["source_detection_indices"] = selected
    return result


def reactivation_candidates(
    state: "MutableGMTState", *, bank_size: int = 10
) -> Tuple[List[int], Set[int]]:
    """Advance the mutable mirror of native ``poss_ids``/``old_reids``.

    A completed online memory bank first becomes eligible in ``poss_ids`` and
    is moved to the persistent stale bank only after the identity is outside
    the current association window.  The averaged bank entry then persists
    until successful reactivation.  This function is deliberately free of
    detector, GT, and future-frame access.
    """

    recent_ids: Set[int] = set()
    for item in state.association_history:
        for value in dict(item.get("assignments", {})).values():
            recent_ids.add(int(value))
    state.memory_bank_size = max(1, int(bank_size))
    for track_id, values in state.memory.items():
        track_id = int(track_id)
        if (
            len(values) >= state.memory_bank_size
            and track_id not in state.reactivation_bank
        ):
            state.possible_memory_ids.add(track_id)

    for track_id in sorted(tuple(state.possible_memory_ids)):
        track_id = int(track_id)
        if track_id in recent_ids:
            continue
        values = state.memory.get(track_id, ())
        if len(values) < state.memory_bank_size:
            continue
        recent_values = values[-state.memory_bank_size :]
        state.reactivation_bank[track_id] = torch.stack(
            [torch.as_tensor(value, dtype=torch.float32) for value in recent_values],
            dim=0,
        ).mean(dim=0).detach().cpu().clone()
        state.possible_memory_ids.discard(track_id)

    return sorted(int(track_id) for track_id in state.reactivation_bank), recent_ids


def build_reactivation_proposal(
    engine,
    payload: Mapping[str, object],
    state: "MutableGMTState",
    candidate_ids: Sequence[int],
    proposal: Optional["AssociationProposal"],
):
    """Run the formal GMT stale-bank association on an isolated state clone."""

    if not candidate_ids:
        return None
    probe = state.clone()
    if proposal is not None and proposal.rng_state_after is not None:
        probe.trajectory_rng_state = proposal.rng_state_after
    probe.active_ids = set(int(value) for value in candidate_ids)
    probe.reactivation_bank = {
        int(track_id): state.reactivation_bank[int(track_id)].detach().cpu().clone()
        for track_id in candidate_ids
    }
    probe.reactivation_mode = True
    return engine.propose(payload, probe)


@dataclass(frozen=True)
class AssociationScoreResult:
    """Association scores plus the explicit RNG provenance for one proposal."""

    scores: torch.Tensor
    rng_state_before: object = None
    rng_state_after: object = None
    trajectory_slot_mapping_digest: Optional[str] = None
    transformer_calls: int = 0


@dataclass
class MutableGMTState:
    """The branch-local mutable part of GMT state."""

    next_id: int = 0
    active_ids: Set[int] = field(default_factory=set)
    stale_ids: Set[int] = field(default_factory=set)
    # These two containers mirror native GMT's process-local ``poss_ids``
    # and persistent ``old_reids`` stale bank.  A candidate is promoted only
    # after its online memory reaches the configured bank size, then remains
    # in the old-reid bank until a successful reactivation removes it.
    possible_memory_ids: Set[int] = field(default_factory=set)
    reactivation_bank: Dict[int, torch.Tensor] = field(default_factory=dict)
    memory_bank_size: int = 10
    reactivation_mode: bool = False
    track_embeddings: Dict[int, torch.Tensor] = field(default_factory=dict)
    track_hits: Dict[int, int] = field(default_factory=dict)
    memory: Dict[int, List[torch.Tensor]] = field(default_factory=dict)
    assignments: Dict[str, int] = field(default_factory=dict)
    counters: Dict[str, int] = field(default_factory=dict)
    association_history: List[Mapping[str, object]] = field(default_factory=list)
    trajectory_rng_state: Optional[object] = None
    trajectory_rng_seed: Optional[int] = None
    trajectory_rng_calls: int = 0

    def initialize_trajectory_rng(
        self,
        video_id: int,
        *,
        master_seed: int = TRAJECTORY_RNG_MASTER_SEED,
    ) -> "MutableGMTState":
        """Initialize the video's branch-local trajectory-slot RNG."""

        seed = trajectory_rng_seed_for_video(video_id, master_seed=master_seed)
        rng = TrajectoryRandom(seed)
        self.trajectory_rng_seed = int(seed)
        self.trajectory_rng_state = copy.deepcopy(rng.getstate())
        self.trajectory_rng_calls = 0
        return self

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
            possible_memory_ids=set(self.possible_memory_ids),
            reactivation_bank={
                int(key): value.detach().clone()
                for key, value in self.reactivation_bank.items()
            },
            memory_bank_size=int(self.memory_bank_size),
            reactivation_mode=bool(self.reactivation_mode),
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
            trajectory_rng_state=copy.deepcopy(self.trajectory_rng_state),
            trajectory_rng_seed=(
                None
                if self.trajectory_rng_seed is None
                else int(self.trajectory_rng_seed)
            ),
            trajectory_rng_calls=int(self.trajectory_rng_calls),
        )


def seed_production_state_from_payload(
    payload: Mapping[str, object],
    seed_key: Sequence[int],
) -> Tuple[MutableGMTState, Mapping[str, object]]:
    """Seed GMT exactly as the native production replay does.

    The first-frame view with the most detections is already committed by GMT
    before policy decisions begin.  Counterfactual data generation must start
    from that same tracker/history state; treating the seed view as an empty
    association step produces tied labels and a train/runtime mismatch.
    """

    detection_count = int(torch.as_tensor(payload["pred_boxes"]).shape[0])
    if detection_count < 1:
        raise RuntimeError(f"production seed view is empty: {tuple(seed_key)}")
    state = MutableGMTState(
        next_id=detection_count,
        active_ids=set(range(1, detection_count + 1)),
        track_hits={track_id: 1 for track_id in range(1, detection_count + 1)},
        track_embeddings={
            track_id: torch.as_tensor(payload["reid_features"][track_id - 1])
            .detach()
            .cpu()
            .clone()
            for track_id in range(1, detection_count + 1)
        },
    )
    state.initialize_trajectory_rng(int(seed_key[0]))
    state.association_history.append(
        {
            "perception": payload,
            "assignments": {row: row + 1 for row in range(detection_count)},
        }
    )
    return state, payload


@dataclass(frozen=True)
class AssociationProposal:
    track_ids: Tuple[int, ...]
    scores: torch.Tensor
    pairs: Mapping[int, int]
    banned_edges: Tuple[Tuple[int, int], ...] = ()
    rng_state_before: object = None
    rng_state_after: object = None
    trajectory_slot_mapping_digest: Optional[str] = None
    transformer_calls: int = 0
    proposal_reused: bool = False


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
        not_mult_thresh: bool = False,
        history_limit: int = 16,
    ):
        self.association_fn = association_fn
        self.acceptance_threshold = float(acceptance_threshold)
        self.not_mult_thresh = bool(not_mult_thresh)
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
        """Return the public legacy pair while preserving formal provenance."""

        result = self._score_matrix_with_provenance(perception, state)
        return self._track_ids(state), result.scores

    def _score_matrix_with_provenance(
        self,
        perception: Mapping[str, object],
        state: MutableGMTState,
    ) -> AssociationScoreResult:
        track_ids = self._track_ids(state)
        if self.association_fn is None:
            scores = self._cosine_scores(perception, track_ids, state)
            result = AssociationScoreResult(
                scores=scores,
                rng_state_before=copy.deepcopy(state.trajectory_rng_state),
                rng_state_after=copy.deepcopy(state.trajectory_rng_state),
                trajectory_slot_mapping_digest=_empty_trajectory_mapping_digest(),
                transformer_calls=0,
            )
        else:
            if getattr(self.association_fn, "formal_gmt_association_adapter", False):
                if state.trajectory_rng_state is None or state.trajectory_rng_seed is None:
                    raise RuntimeError(
                        "formal replay requires an initialized trajectory_rng_state and trajectory_rng_seed"
                    )
            raw_result = self.association_fn(perception, track_ids, state)
            if isinstance(raw_result, AssociationScoreResult):
                result = raw_result
            elif all(
                hasattr(raw_result, name)
                for name in (
                    "scores",
                    "rng_state_before",
                    "rng_state_after",
                    "trajectory_slot_mapping_digest",
                    "transformer_calls",
                )
            ):
                result = AssociationScoreResult(
                    scores=raw_result.scores,
                    rng_state_before=raw_result.rng_state_before,
                    rng_state_after=raw_result.rng_state_after,
                    trajectory_slot_mapping_digest=raw_result.trajectory_slot_mapping_digest,
                    transformer_calls=int(raw_result.transformer_calls),
                )
            else:
                if getattr(self.association_fn, "formal_gmt_association_adapter", False):
                    raise RuntimeError(
                        "formal counterfactual association must return explicit RNG provenance"
                    )
                result = AssociationScoreResult(
                    scores=raw_result,
                    rng_state_before=copy.deepcopy(state.trajectory_rng_state),
                    rng_state_after=copy.deepcopy(state.trajectory_rng_state),
                    trajectory_slot_mapping_digest=_empty_trajectory_mapping_digest(),
                    transformer_calls=0,
                )
            if getattr(self.association_fn, "formal_gmt_association_adapter", False):
                if result.rng_state_before is None or result.rng_state_after is None:
                    raise RuntimeError(
                        "formal replay association result is missing RNG state provenance"
                    )
                if result.trajectory_slot_mapping_digest is None:
                    raise RuntimeError(
                        "formal replay association result is missing trajectory mapping digest"
                    )
            result = AssociationScoreResult(
                scores=torch.as_tensor(result.scores, dtype=torch.float32),
                rng_state_before=copy.deepcopy(result.rng_state_before),
                rng_state_after=copy.deepcopy(result.rng_state_after),
                trajectory_slot_mapping_digest=result.trajectory_slot_mapping_digest,
                transformer_calls=int(result.transformer_calls),
            )
        scores = result.scores
        detection_count = int(torch.as_tensor(perception["pred_boxes"]).shape[0])
        expected = (detection_count, len(track_ids))
        if tuple(scores.shape) != expected:
            raise ValueError(f"association backend returned {tuple(scores.shape)}, expected {expected}")
        if not torch.isfinite(scores).all():
            raise ValueError("association backend returned non-finite scores")
        return result

    def propose(
        self,
        perception: Mapping[str, object],
        state: MutableGMTState,
        *,
        banned_edges: Iterable[Tuple[int, int]] = (),
    ) -> AssociationProposal:
        track_ids = self._track_ids(state)
        score_result = self._score_matrix_with_provenance(perception, state)
        scores = score_result.scores
        banned = tuple(sorted((int(row), int(col)) for row, col in banned_edges))
        pairs = dict(constrained_hungarian(scores, banned))
        return AssociationProposal(
            track_ids,
            scores,
            pairs,
            banned,
            rng_state_before=copy.deepcopy(score_result.rng_state_before),
            rng_state_after=copy.deepcopy(score_result.rng_state_after),
            trajectory_slot_mapping_digest=score_result.trajectory_slot_mapping_digest,
            transformer_calls=int(score_result.transformer_calls),
        )

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

    @staticmethod
    def _update_memory_eligibility(state: MutableGMTState) -> None:
        """Promote completed online memory banks to native ``poss_ids``."""

        bank_size = max(1, int(state.memory_bank_size))
        for track_id, values in state.memory.items():
            track_id = int(track_id)
            if len(values) >= bank_size and track_id not in state.reactivation_bank:
                state.possible_memory_ids.add(track_id)

    def _actions_for_proposal(
        self,
        proposal: AssociationProposal,
        state: MutableGMTState,
        *,
        threshold: Optional[float] = None,
    ) -> Dict[int, str]:
        threshold = self.acceptance_threshold if threshold is None else float(threshold)
        actions = {}
        for row in range(proposal.scores.shape[0]):
            col = proposal.pairs.get(row)
            track_length = (
                count_track_history(state.association_history, proposal.track_ids[col])
                if col is not None
                else 1
            )
            legacy_threshold = legacy_acceptance_threshold(
                threshold, track_length, self.not_mult_thresh
            )
            if col is None or float(proposal.scores[row, col]) <= legacy_threshold:
                actions[row] = "START_NEW"
            else:
                actions[row] = "ACCEPT_CURRENT"
        return actions

    def resolve_actions(
        self,
        perception: Mapping[str, object],
        state: MutableGMTState,
        *,
        actions: Optional[Mapping[int, str]] = None,
        threshold: Optional[float] = None,
        proposal: Optional[AssociationProposal] = None,
    ) -> Mapping[str, object]:
        """Resolve semantic association actions without mutating tracker state.

        This mirrors the association-resolution prefix of step() and is used
        by closed-loop runtimes that must decide memory writes only after the
        final existing-track assignment is known. Rows that will create a new
        identity are returned with None as their existing track id.
        """

        proposal = proposal or self.propose(perception, state)
        action_map = dict(
            self._actions_for_proposal(proposal, state, threshold=threshold)
            if actions is None
            else actions
        )
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
            # REASSOCIATE is a constrained solve over the exact proposal that
            # produced the legal actions.  Re-running the transformer here
            # would consume a different trajectory-slot RNG stream and would
            # make action semantics depend on evaluation order.
            second = AssociationProposal(
                track_ids=proposal.track_ids,
                scores=proposal.scores,
                pairs=dict(constrained_hungarian(proposal.scores, banned)),
                banned_edges=tuple(sorted(banned)),
                rng_state_before=copy.deepcopy(proposal.rng_state_before),
                rng_state_after=copy.deepcopy(proposal.rng_state_after),
                trajectory_slot_mapping_digest=proposal.trajectory_slot_mapping_digest,
                transformer_calls=0,
                proposal_reused=True,
            )

        existing_track_ids: Dict[int, Optional[int]] = {}
        for row in range(proposal.scores.shape[0]):
            action = action_map[row]
            col = second.pairs.get(row)
            if action == "START_NEW" or col is None:
                existing_track_ids[row] = None
            else:
                # GTRRCNN validates the second constrained-Hungarian proposal
                # against the legacy scaled threshold before committing it.
                # The first proposal keeps the historical explicit-action
                # semantics; this check is only needed when reassociation
                # caused the second solve.
                if reassociate_rows:
                    track_id = int(second.track_ids[col])
                    track_length = count_track_history(
                        state.association_history, track_id
                    )
                    candidate_score = float(second.scores[row, col].item())
                    if candidate_score <= legacy_acceptance_threshold(
                        self.acceptance_threshold,
                        track_length,
                        self.not_mult_thresh,
                    ):
                        existing_track_ids[row] = None
                        continue
                existing_track_ids[row] = int(second.track_ids[col])

        return {
            "initial_proposal": proposal,
            "final_proposal": second,
            "actions": action_map,
            "reassociate_rows": tuple(reassociate_rows),
            "existing_track_ids": existing_track_ids,
        }
    def step(
        self,
        perception: Mapping[str, object],
        state: MutableGMTState,
        *,
        actions: Optional[Mapping[int, str]] = None,
        memory_actions: Optional[Mapping[int, str]] = None,
        reactivation_assignments: Optional[Mapping[int, int]] = None,
        threshold: Optional[float] = None,
        proposal: Optional[AssociationProposal] = None,
        reactivation_proposal: Optional[AssociationProposal] = None,
    ) -> Mapping[str, object]:
        """Commit one frame/view from an isolated mutable state.

        If one or more rows request REASSOCIATE, all rejected edges are
        masked and the complete matrix is solved once.  The result is then
        consumed by the semantic actions; no ranked-candidate fallback exists.
        """
        supplied_proposal = proposal
        resolution = self.resolve_actions(
            perception,
            state,
            actions=actions,
            threshold=threshold,
            proposal=proposal,
        )
        proposal = resolution["initial_proposal"]
        second = resolution["final_proposal"]
        action_map = dict(resolution["actions"])
        reactivation_assignments = {
            int(row): int(track_id)
            for row, track_id in (reactivation_assignments or {}).items()
        }
        reassociate_rows = list(resolution["reassociate_rows"])
        if supplied_proposal is not None and supplied_proposal.scores is not proposal.scores:
            raise AssertionError("step did not reuse the supplied association proposal")
        if resolution["final_proposal"].proposal_reused and (
            resolution["final_proposal"].scores is not proposal.scores
        ):
            raise AssertionError("REASSOCIATE proposal does not reuse the original score matrix")
        if reassociate_rows:
            state.counters["reassociation_calls"] = state.counters.get("reassociation_calls", 0) + 1

        # ``propose`` and ``resolve_actions`` are observational.  Only a
        # committed step advances the branch-local RNG provenance, and it does
        # so once for the initial transformer call (never for constrained
        # Hungarian re-solves).
        if resolution["initial_proposal"].rng_state_after is not None:
            state.trajectory_rng_state = copy.deepcopy(
                resolution["initial_proposal"].rng_state_after
            )
            state.trajectory_rng_calls = int(state.trajectory_rng_calls) + int(
                resolution["initial_proposal"].transformer_calls
            )
        if reactivation_proposal is not None and reactivation_proposal.rng_state_after is not None:
            # The native memory-bank path performs a second association pass
            # for stale identities.  Keep that pass explicit and branch-local
            # when a runtime replay supplies a reactivation proposal.
            state.trajectory_rng_state = copy.deepcopy(
                reactivation_proposal.rng_state_after
            )
            state.trajectory_rng_calls = int(state.trajectory_rng_calls) + int(
                reactivation_proposal.transformer_calls
            )

        features = _normalise_features(
            torch.as_tensor(perception["reid_features"], dtype=torch.float32)
        )
        committed: Dict[int, int] = {}
        for row in range(features.shape[0]):
            action = action_map[row]
            col = second.pairs.get(row)
            if row in reactivation_assignments:
                track_id = int(reactivation_assignments[row])
                state.active_ids.add(track_id)
                state.stale_ids.discard(track_id)
                # Native ``run_memory_tracker`` removes a successfully
                # reactivated ID from ``old_reids`` and returns it to
                # ``poss_ids`` for future bank updates.
                state.reactivation_bank.pop(track_id, None)
                if len(state.memory.get(track_id, ())) >= max(
                    1, int(state.memory_bank_size)
                ):
                    state.possible_memory_ids.add(track_id)
                state.counters["reactivated_rows"] = state.counters.get(
                    "reactivated_rows", 0
                ) + 1
            elif action == "START_NEW" or col is None:
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

        self._update_memory_eligibility(state)

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
        # Native GMT forms the next proposal from the current sliding window,
        # not from every ID ever created in the video.  Keep ``active_ids`` as
        # that window-derived set while retaining ``next_id``/hit/memory
        # provenance for identities that have moved to the stale bank.
        state.active_ids = {
            int(track_id)
            for item in state.association_history
            for track_id in dict(item.get("assignments", {})).values()
        }

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
            "proposal_provenance": {
                "rng_state_before": proposal_state_for_output(
                    proposal.rng_state_before
                ),
                "rng_state_after": proposal_state_for_output(
                    proposal.rng_state_after
                ),
                "trajectory_slot_mapping_digest": resolution[
                    "initial_proposal"
                ].trajectory_slot_mapping_digest,
                "transformer_calls": int(
                    resolution["initial_proposal"].transformer_calls
                ),
                "reassociate_reused_score_matrix": bool(
                    second.proposal_reused
                ),
            },
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
            steps.append(
                self.step(
                    perception,
                    branch,
                    actions=actions,
                    proposal=proposal,
                )
            )
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
    branches = {name: source.clone() for name in action_plans}
    steps_by_name = {name: [] for name in action_plans}
    for index, perception in enumerate(perceptions):
        # At the shared starting decision, all named branches consume the
        # exact same immutable proposal.  Once a branch has committed an
        # action, later proposals legitimately depend on that branch's
        # mutated history and are therefore computed independently.
        shared_proposal = None
        if index == 0 and branches:
            shared_proposal = engine.propose(perception, source)
        for name, plan in action_plans.items():
            branch = branches[name]
            proposal = shared_proposal if shared_proposal is not None else engine.propose(perception, branch)
            steps_by_name[name].append(
                engine.step(perception, branch, actions=plan.get(index), proposal=proposal)
            )
            if _state_signature(initial_state) != source_signature:
                raise AssertionError(f"counterfactual branch mutated source state: {name}")
    results = {
        name: (branches[name], steps_by_name[name]) for name in action_plans
    }
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
        tuple(
            (
                (
                    int(item["perception"]["video_id"]),
                    int(item["perception"]["frame"]),
                    int(item["perception"]["view"]),
                )
                if isinstance(item.get("perception"), Mapping)
                and all(
                    key in item["perception"] for key in ("video_id", "frame", "view")
                )
                else id(item.get("perception")),
                tuple(
                    sorted(
                        (int(key), int(value))
                        for key, value in dict(item.get("assignments", {})).items()
                    )
                ),
            )
            for item in state.association_history
        ),
        repr(state.trajectory_rng_state),
        None if state.trajectory_rng_seed is None else int(state.trajectory_rng_seed),
        int(state.trajectory_rng_calls),
    )
