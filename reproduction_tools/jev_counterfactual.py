"""Mutation-isolated counterfactual branch runner for offline JEV labels."""

import copy
import hashlib
import json
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence


def state_digest(state: Any) -> str:
    """Stable digest for JSON-like tracker snapshots used by tests/logs."""
    payload = json.dumps(state, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


class CounterfactualBranchRunner:
    """Run every action from an identical deep-copied online state.

    ``apply_action`` and ``rollout`` are injected by the GMT-specific adapter.
    The runner never selects an identity and never imports an evaluator; it
    only enforces branch isolation and records the supplied utility.
    """

    def __init__(self, apply_action: Callable[[Any, str], Any], rollout: Callable[[Any, int], Any], utility: Callable[[Any], Mapping[str, float]]):
        self.apply_action = apply_action
        self.rollout = rollout
        self.utility = utility

    def run(self, state: Any, legal_actions: Sequence[str], horizon: int) -> Dict[str, Dict[str, Any]]:
        if not legal_actions:
            raise ValueError("counterfactual branch requires at least one legal action")
        if horizon < 1:
            raise ValueError("horizon must be positive")
        original_digest = state_digest(state)
        results: Dict[str, Dict[str, Any]] = {}
        for action in legal_actions:
            branch = copy.deepcopy(state)
            branch_before = state_digest(branch)
            next_state = self.apply_action(branch, action)
            rollout_state = self.rollout(next_state, horizon)
            metrics = dict(self.utility(rollout_state))
            results[action] = {
                "utility": metrics,
                "rollout_digest": state_digest(rollout_state),
                "branch_before_digest": branch_before,
            }
            if state_digest(state) != original_digest:
                raise AssertionError(f"counterfactual action mutated source state: {action}")
        return results

