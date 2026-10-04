"""Mutation-isolated frozen-evidence GMT tracker-state rollouts.

The runner consumes the online decision trace produced by the real GMT
adapter.  It restores JSON-serializable GMT containers (track assignments,
hit/memory counters, stale-bank membership) for each branch and replays the
same future decision evidence.  Future GT is accepted only by the offline
utility callback.  Detector and association networks are never called here,
so the manifest must identify this as a frozen-evidence branch run.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple


MAIN_SCOPES = {"match", "memory", "reactivation"}
MATCH_QUESTIONS = {"MATCH_DECISION", "REACTIVATION_DECISION"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def digest_state(value: Any) -> str:
    payload = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def is_main_decision(event: Mapping[str, Any]) -> bool:
    return str(event.get("context", {}).get("decision_scope", "")) in MAIN_SCOPES


def event_token(event: Mapping[str, Any], position: int) -> str:
    context = event.get("context", {})
    video = int(context.get("video_id", -1))
    order = int(context.get("event_order", position))
    return f"{video}:{order}:{event.get('question', '')}"


def frame_token(event: Mapping[str, Any]) -> str:
    context = event.get("context", {})
    return "{}:{}:{}".format(
        int(context.get("video_id", -1)),
        int(context.get("frame", 0)),
        int(context.get("view", 0)),
    )


@dataclass
class TraceTrackerState:
    """A deep-copyable offline representation of GMT commit containers."""

    next_id: int = 0
    assignments: Dict[str, int] = field(default_factory=dict)
    used_ids_by_frame: Dict[str, set] = field(default_factory=dict)
    active_ids: set = field(default_factory=set)
    track_hits: Dict[int, int] = field(default_factory=dict)
    memory_lengths: Dict[int, int] = field(default_factory=dict)
    memory_targets: Dict[int, int] = field(default_factory=dict)
    track_targets: Dict[int, int] = field(default_factory=dict)
    target_tracks: Dict[int, int] = field(default_factory=dict)
    possible_memory_ids: set = field(default_factory=set)
    stale_ids: set = field(default_factory=set)
    id_switches: int = 0
    fragments: int = 0
    collisions: int = 0
    memory_contamination: int = 0
    memory_writes: int = 0
    new_ids: int = 0
    reactivations: int = 0

    @classmethod
    def from_event(cls, event: Mapping[str, Any]) -> "TraceTrackerState":
        context = event.get("context", {})
        raw = context.get("tracker_state_before", {})
        state = cls()
        if isinstance(raw, Mapping):
            state.next_id = int(raw.get("id_count", 0) or 0)
            state.active_ids = {int(value) for value in raw.get("active_track_ids", ())}
            state.possible_memory_ids = {
                int(value) for value in raw.get("possible_memory_ids", ())
            }
            state.stale_ids = {int(value) for value in raw.get("stale_ids", ())}
            state.track_hits = {
                int(key): int(value)
                for key, value in dict(raw.get("track_hits", {})).items()
            }
            state.memory_lengths = {
                int(key): int(value)
                for key, value in dict(raw.get("memory_lengths", {})).items()
            }
        for value in (
            context.get("proposal_track_id"),
            context.get("alternate_track_id"),
            context.get("track_id"),
        ):
            if value is not None:
                state.next_id = max(state.next_id, int(value))
        return state

    def snapshot(self) -> Dict[str, Any]:
        return {
            "next_id": int(self.next_id),
            "assignments": dict(self.assignments),
            "used_ids_by_frame": {
                key: sorted(int(value) for value in values)
                for key, values in self.used_ids_by_frame.items()
            },
            "active_ids": sorted(int(value) for value in self.active_ids),
            "track_hits": {str(key): int(value) for key, value in self.track_hits.items()},
            "memory_lengths": {
                str(key): int(value) for key, value in self.memory_lengths.items()
            },
            "memory_targets": {
                str(key): int(value) for key, value in self.memory_targets.items()
            },
            "track_targets": {
                str(key): int(value) for key, value in self.track_targets.items()
            },
            "target_tracks": {
                str(key): int(value) for key, value in self.target_tracks.items()
            },
            "possible_memory_ids": sorted(int(value) for value in self.possible_memory_ids),
            "stale_ids": sorted(int(value) for value in self.stale_ids),
            "id_switches": int(self.id_switches),
            "fragments": int(self.fragments),
            "collisions": int(self.collisions),
            "memory_contamination": int(self.memory_contamination),
            "memory_writes": int(self.memory_writes),
            "new_ids": int(self.new_ids),
            "reactivations": int(self.reactivations),
        }


class FrozenEvidenceGMTBranchRunner:
    """Run typed actions against isolated mutable GMT trace state."""

    def __init__(
        self,
        events: Sequence[Mapping[str, Any]],
        target_for_event: Callable[[Mapping[str, Any]], Optional[int]],
    ):
        self.events = list(events)
        self.target_for_event = target_for_event

    @staticmethod
    def _new_id(state: TraceTrackerState) -> int:
        state.next_id = max(
            [int(state.next_id), *[int(value) for value in state.active_ids], *state.track_targets.keys(), 0]
        ) + 1
        state.new_ids += 1
        return state.next_id

    @staticmethod
    def _action_target(
        state: TraceTrackerState,
        event: Mapping[str, Any],
        action: str,
    ) -> Optional[int]:
        context = event.get("context", {})
        if action == "ACCEPT_CURRENT":
            value = context.get("proposal_track_id")
        elif action == "REASSOCIATE":
            value = context.get("alternate_track_id")
        elif action == "REACTIVATE_OLD":
            value = context.get("track_id")
        elif action == "START_NEW":
            value = None
        else:
            return None
        if value is None or action == "START_NEW":
            return FrozenEvidenceGMTBranchRunner._new_id(state)
        return int(value)

    @staticmethod
    def _pre_action_identity_utility(
        state: TraceTrackerState,
        event: Mapping[str, Any],
        action: str,
        target: Optional[int],
    ) -> float:
        """Score the current decision using state *before* applying it.

        The old implementation applied the branch first and then looked up
        ``track_targets``.  That made an already-wrong ACCEPT/REACTIVATE look
        correct because the GT mapping had just been overwritten.  A new ID
        is only immediately correct when the target has not already been
        assigned to another persistent track; an unknown existing ID is
        intentionally not awarded positive utility.
        """

        if target is None:
            return 0.0
        context = event.get("context", {})
        if action == "START_NEW":
            existing = state.target_tracks.get(int(target))
            return 1.0 if existing is None else -1.0
        if action == "ACCEPT_CURRENT":
            identifier = context.get("proposal_track_id")
        elif action == "REASSOCIATE":
            identifier = context.get("alternate_track_id")
        elif action == "REACTIVATE_OLD":
            identifier = context.get("track_id")
        else:
            return 0.0
        if identifier is None:
            return -1.0
        previous = state.track_targets.get(int(identifier))
        if previous is None:
            return 0.0
        return 1.0 if int(previous) == int(target) else -1.0

    def apply(
        self,
        state: TraceTrackerState,
        event: Mapping[str, Any],
        action: str,
        *,
        target: Optional[int] = None,
        position: int = 0,
    ) -> Optional[int]:
        question = str(event.get("question", ""))
        context = event.get("context", {})
        if question in MATCH_QUESTIONS and is_main_decision(event):
            identifier = self._action_target(state, event, action)
            token = event_token(event, position)
            frame = frame_token(event)
            used = state.used_ids_by_frame.setdefault(frame, set())
            previous = state.assignments.get(token)
            if previous is not None and previous != identifier:
                state.id_switches += 1
            if identifier in used and previous != identifier:
                state.collisions += 1
            used.add(identifier)
            state.assignments[token] = int(identifier)
            state.active_ids.add(int(identifier))
            state.track_hits[int(identifier)] = state.track_hits.get(int(identifier), 0) + 1
            if question == "REACTIVATION_DECISION" and action == "REACTIVATE_OLD":
                state.reactivations += 1
            if target is not None:
                old_target = state.track_targets.get(int(identifier))
                if old_target is not None and old_target != int(target):
                    state.id_switches += 1
                prior_identifier = state.target_tracks.get(int(target))
                if prior_identifier is not None and prior_identifier != int(identifier):
                    state.fragments += 1
                state.track_targets[int(identifier)] = int(target)
                state.target_tracks[int(target)] = int(identifier)
            return int(identifier)

        if question == "MEMORY_DECISION" and is_main_decision(event):
            track = context.get("track_id")
            if track is not None and action == "WRITE_MEMORY":
                track = int(track)
                state.memory_writes += 1
                state.memory_lengths[track] = state.memory_lengths.get(track, 0) + 1
                if target is not None:
                    old_target = state.memory_targets.get(track)
                    if old_target is not None and old_target != int(target):
                        state.memory_contamination += 1
                    state.memory_targets[track] = int(target)
            return None
        return None

    def _future_events(self, position: int, current: Mapping[str, Any], horizon: int):
        current_frame = int(current.get("context", {}).get("frame", 0))
        for future_position in range(position + 1, len(self.events)):
            event = self.events[future_position]
            if int(event.get("context", {}).get("frame", 0)) > current_frame + horizon:
                break
            if is_main_decision(event) and str(event.get("question")) in {
                "MATCH_DECISION",
                "MEMORY_DECISION",
                "REACTIVATION_DECISION",
            }:
                yield future_position, event

    def _identity_utility(
        self,
        branch: TraceTrackerState,
        position: int,
        current: Mapping[str, Any],
        action: str,
        horizon: int,
    ) -> Dict[str, float]:
        current_target = self.target_for_event(current)
        before_switches = branch.id_switches
        before_fragments = branch.fragments
        before_collisions = branch.collisions
        # Evaluate this metric from the pre-action state.  Applying the
        # action first would overwrite a wrong track->GT mapping and create
        # an immediate-correctness label leak.
        immediate = self._pre_action_identity_utility(
            branch, current, action, current_target
        )
        assigned = self.apply(
            branch,
            current,
            action,
            target=current_target,
            position=position,
        )
        correct = 0
        total = 0
        for future_position, future in self._future_events(position, current, horizon):
            target = self.target_for_event(future)
            off_action = str(future.get("off_action") or future.get("proposed_action"))
            assigned = self.apply(
                branch,
                future,
                off_action,
                target=target,
                position=future_position,
            )
            if target is None:
                continue
            total += 1
            correct += int(assigned is not None and branch.track_targets.get(assigned) == target)
        consistency = correct / total if total else 0.0
        switches = branch.id_switches - before_switches
        fragments = branch.fragments - before_fragments
        collisions = branch.collisions - before_collisions
        return {
            "utility": float(immediate + consistency - 0.5 * switches - 0.25 * fragments - 0.5 * collisions),
            "sample_weight": 1.0,
            "informative": True,
            "immediate_identity": float(immediate),
            "future_identity_consistency": float(consistency),
            "future_identity_switches": float(switches),
            "future_fragments": float(fragments),
            "future_collisions": float(collisions),
            "future_events": float(total),
        }

    def _memory_utility(
        self,
        branch: TraceTrackerState,
        position: int,
        current: Mapping[str, Any],
        action: str,
        horizon: int,
    ) -> Dict[str, float]:
        track = current.get("context", {}).get("track_id")
        target = self.target_for_event(current)
        before_contamination = branch.memory_contamination
        before_writes = branch.memory_writes
        self.apply(branch, current, action, target=target, position=position)
        consistent = 0
        total = 0
        for future_position, future in self._future_events(position, current, horizon):
            if str(future.get("question")) != "MATCH_DECISION" or track is None:
                continue
            context = future.get("context", {})
            if int(track) not in {
                int(context["proposal_track_id"])
                if context.get("proposal_track_id") is not None
                else -1,
                int(context["alternate_track_id"])
                if context.get("alternate_track_id") is not None
                else -1,
            }:
                continue
            future_target = self.target_for_event(future)
            if future_target is None:
                continue
            total += 1
            memory_target = branch.memory_targets.get(int(track))
            # ``None`` means that this branch has no informative memory
            # evidence yet.  Treating it as correct systematically favors
            # SKIP_MEMORY and is not a valid future-consistency label.
            if memory_target is None:
                total -= 1
            off_action = str(future.get("off_action") or future.get("proposed_action"))
            self.apply(
                branch,
                future,
                off_action,
                target=future_target,
                position=future_position,
            )
            if memory_target is None:
                continue
            consistent += int(memory_target == future_target)
        informative = total > 0
        ratio = consistent / total if informative else 0.0
        contamination = branch.memory_contamination - before_contamination
        return {
            # Uninformative WRITE/SKIP branches are an explicit tie with zero
            # training weight, not a hidden SKIP_MEMORY preference.
            "utility": float(ratio - contamination) if informative else 0.0,
            "sample_weight": 1.0 if informative else 0.0,
            "informative": bool(informative),
            "future_memory_consistency": float(ratio),
            "memory_contamination": float(contamination),
            "memory_writes": float(branch.memory_writes - before_writes),
            "future_events": float(total),
        }

    def run(
        self,
        base_state: TraceTrackerState,
        position: int,
        event: Mapping[str, Any],
        legal_actions: Sequence[str],
        horizon: int,
    ) -> Dict[str, Dict[str, Any]]:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if not legal_actions:
            raise ValueError("legal action set cannot be empty")
        source_digest = digest_state(base_state.snapshot())
        results: Dict[str, Dict[str, Any]] = {}
        for action in legal_actions:
            branch = copy.deepcopy(base_state)
            branch_before = digest_state(branch.snapshot())
            if str(event.get("question")) == "MEMORY_DECISION":
                metrics = self._memory_utility(branch, position, event, action, horizon)
            else:
                metrics = self._identity_utility(branch, position, event, action, horizon)
            results[action] = {
                **metrics,
                "branch_before_digest": branch_before,
                "rollout_digest": digest_state(branch.snapshot()),
            }
            if digest_state(base_state.snapshot()) != source_digest:
                raise AssertionError(f"counterfactual action mutated source state: {action}")
        return results
