"""Contract tests for frozen-perception mutable-association branches."""

import copy

import torch

from jev_counterfactual_v2 import (
    ENGINE_VERSION,
    CachedPerceptionMutableAssociationV2,
    MutableGMTState,
    _state_signature,
    run_counterfactual_branches,
)


def perception(features):
    return {
        "cache_version": "frozen_perception_cache_v2",
        "pred_boxes": torch.zeros((len(features), 4)),
        "detection_scores": torch.ones(len(features)),
        "reid_features": torch.tensor(features, dtype=torch.float32),
        "image_size": (32, 32),
    }


def main():
    engine = CachedPerceptionMutableAssociationV2(acceptance_threshold=-1.0)
    initial = MutableGMTState(
        next_id=2,
        active_ids={1, 2},
        track_embeddings={
            1: torch.tensor([1.0, 0.0]),
            2: torch.tensor([0.0, 1.0]),
        },
        track_hits={1: 1, 2: 1},
    )
    perceptions = [
        perception([[0.99, 0.01], [0.01, 0.99]]),
        perception([[0.01, 0.99], [0.99, 0.01]]),
    ]
    before = copy.deepcopy(initial)
    result = run_counterfactual_branches(
        engine,
        perceptions,
        initial,
        {
            "accept": {0: {0: "ACCEPT_CURRENT", 1: "ACCEPT_CURRENT"}},
            "new": {0: {0: "START_NEW", 1: "START_NEW"}},
        },
    )
    assert initial.next_id == before.next_id
    assert initial.active_ids == before.active_ids
    assert torch.equal(initial.track_embeddings[1], before.track_embeddings[1])
    accept_state, accept_steps = result["accept"]
    new_state, new_steps = result["new"]
    assert accept_steps[0]["engine_version"] == ENGINE_VERSION
    assert accept_state.counters.get("new_ids", 0) < new_state.counters.get("new_ids", 0)
    assert accept_state.track_hits != new_state.track_hits

    frozen = perception([[1.0, 0.0]])
    history_state = MutableGMTState(
        association_history=[{"perception": frozen, "assignments": {0: 4}}]
    )
    history_clone = history_state.clone()
    assert history_clone.association_history[0]["perception"] is frozen
    history_clone.association_history[0]["assignments"][0] = 9
    assert history_state.association_history[0]["assignments"][0] == 4

    # Non-mutating resolution must predict existing-track outcomes without
    # consuming IDs, updating counters, or changing history.
    resolution_state = initial.clone()
    resolution_before = _state_signature(resolution_state)
    resolution = engine.resolve_actions(
        perception([[0.70, 0.70], [0.99, 0.01]]),
        resolution_state,
        actions={0: "REASSOCIATE", 1: "ACCEPT_CURRENT"},
    )
    assert _state_signature(resolution_state) == resolution_before
    assert resolution["existing_track_ids"][1] in {1, 2}
    reassoc_state = initial.clone()
    reassoc = engine.step(
        perception([[0.70, 0.70], [0.99, 0.01]]),
        reassoc_state,
        actions={0: "REASSOCIATE", 1: "ACCEPT_CURRENT"},
    )
    assert reassoc["banned_edges"]
    assert reassoc_state.counters["reassociation_calls"] == 1

    # Candidate-conditioned MATCH selection binds a concrete legal native
    # track without falling back to the legacy threshold or re-running GMT.
    candidate_state = initial.clone()
    candidate_perception = perception([[0.99, 0.01], [0.01, 0.99]])
    candidate_proposal = engine.propose(candidate_perception, candidate_state)
    candidate_resolution = engine.resolve_actions(
        candidate_perception,
        candidate_state,
        actions={0: "ACCEPT_CURRENT", 1: "ACCEPT_CURRENT"},
        candidate_assignments={0: 2},
        proposal=candidate_proposal,
    )
    assert candidate_resolution["candidate_assignments"] == {0: 2}
    assert candidate_resolution["existing_track_ids"][0] == 2
    assert candidate_resolution["final_proposal"].proposal_reused is True
    candidate_step = engine.step(
        candidate_perception,
        candidate_state,
        actions={0: "ACCEPT_CURRENT", 1: "ACCEPT_CURRENT"},
        candidate_assignments={0: 2},
        proposal=candidate_proposal,
    )
    assert candidate_step["committed_track_ids"][0] == 2
    assert candidate_step["candidate_assignments"] == {0: 2}
    try:
        engine.resolve_actions(
            candidate_perception,
            initial.clone(),
            actions={0: "START_NEW", 1: "ACCEPT_CURRENT"},
            candidate_assignments={0: 2},
            proposal=candidate_proposal,
        )
    except ValueError as exc:
        assert "ACCEPT_CURRENT" in str(exc)
    else:
        raise AssertionError("candidate assignment bypassed its action binding")
    print("JEV cached-perception v2 invariants: PASS")


if __name__ == "__main__":
    main()
