"""Hard gates for explicit branch-local counterfactual RNG semantics."""

from collections import OrderedDict
import random

import torch

from gtr.modeling.roi_heads.transformer import (
    TrajectoryRandom,
    _trajectory_slot_mapping,
)
from jev_counterfactual_v2 import (
    AssociationScoreResult,
    CachedPerceptionMutableAssociationV2,
    MutableGMTState,
    TRAJECTORY_RNG_MASTER_SEED,
    TRAJECTORY_RNG_POLICY,
    _state_signature,
)


class CountingFormalAssociation:
    formal_gmt_association_adapter = True

    def __init__(self):
        self.calls = 0

    def __call__(self, perception, track_ids, state):
        self.calls += 1
        rng = TrajectoryRandom(0)
        rng.setstate(state.trajectory_rng_state)
        before = rng.getstate()
        # Preserve the same explicit sample/choices API used by the GMT
        # trajectory mapping.  The generated score matrix is fixed so this
        # test isolates provenance and proposal semantics from model output.
        rng.sample(range(400), min(3, len(track_ids)))
        scores = torch.tensor(
            [[0.90, 0.80, 0.10], [0.10, 0.90, 0.80]], dtype=torch.float32
        )[:, : len(track_ids)]
        return AssociationScoreResult(
            scores=scores,
            rng_state_before=before,
            rng_state_after=rng.getstate(),
            trajectory_slot_mapping_digest=rng.trajectory_mapping_digest(),
            transformer_calls=1,
        )


def make_perception():
    return {
        "cache_version": "frozen_perception_cache_v2",
        "video_id": 8,
        "frame": 1,
        "view": 0,
        "pred_boxes": torch.zeros((2, 4)),
        "reid_features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "image_size": (32, 32),
    }


def make_state():
    state = MutableGMTState(
        next_id=3,
        active_ids={1, 2, 3},
        track_embeddings={
            1: torch.tensor([1.0, 0.0]),
            2: torch.tensor([0.0, 1.0]),
            3: torch.tensor([1.0, 1.0]),
        },
        track_hits={1: 1, 2: 1, 3: 1},
    )
    state.initialize_trajectory_rng(8)
    return state


def assert_proposal_equal(left, right):
    assert left.track_ids == right.track_ids
    assert left.pairs == right.pairs
    assert torch.equal(left.scores, right.scores)
    assert left.rng_state_before == right.rng_state_before
    assert left.rng_state_after == right.rng_state_after
    assert left.trajectory_slot_mapping_digest == right.trajectory_slot_mapping_digest


def main():
    assert TRAJECTORY_RNG_POLICY == "branch_local_explicit_python_random_v1"
    assert TRAJECTORY_RNG_MASTER_SEED == 20261006

    # The explicit wrapper preserves the released sample/choices stream; it
    # only makes the RNG object injectable and records provenance.
    cues = list(range(401))
    legacy_rng = random.Random(17)
    explicit_rng = TrajectoryRandom(17)
    assert _trajectory_slot_mapping(cues, legacy_rng) == _trajectory_slot_mapping(
        cues, explicit_rng
    )
    assert explicit_rng.trajectory_mapping_digest().startswith("sha256:")

    association = CountingFormalAssociation()
    engine = CachedPerceptionMutableAssociationV2(
        association_fn=association,
        acceptance_threshold=-1.0,
    )
    perception = make_perception()
    source = make_state()
    source_before = _state_signature(source)

    # Clone isolation and repeatability: every branch starts with an identical
    # RNG state and therefore receives identical scores/provenance.
    branch_a = source.clone()
    branch_b = source.clone()
    proposal_a = engine.propose(perception, branch_a)
    proposal_b = engine.propose(perception, branch_b)
    assert_proposal_equal(proposal_a, proposal_b)
    assert association.calls == 2
    assert _state_signature(source) == source_before
    assert branch_a.trajectory_rng_calls == 0
    assert branch_b.trajectory_rng_calls == 0

    # Candidate evaluation order is also a gate: each legal action starts
    # from the same pre-decision state, so its committed result is invariant
    # to the order in which the caller enumerates actions.
    order_association = CountingFormalAssociation()
    order_engine = CachedPerceptionMutableAssociationV2(
        association_fn=order_association,
        acceptance_threshold=-1.0,
    )

    def evaluate_in_order(order):
        outcomes = {}
        for action in order:
            branch = source.clone()
            proposal = order_engine.propose(perception, branch)
            result = order_engine.step(
                perception,
                branch,
                actions={0: action, 1: "ACCEPT_CURRENT"},
                proposal=proposal,
            )
            outcomes[action] = {
                "committed": dict(result["committed_track_ids"]),
                "pairs": dict(result["final_pairs"]),
                "scores": result["final_scores"].clone(),
                "state": _state_signature(branch),
            }
        return outcomes

    candidates_a = evaluate_in_order(
        ("ACCEPT_CURRENT", "REASSOCIATE", "START_NEW")
    )
    candidates_b = evaluate_in_order(
        ("START_NEW", "ACCEPT_CURRENT", "REASSOCIATE")
    )
    for action in candidates_a:
        assert candidates_a[action]["committed"] == candidates_b[action]["committed"]
        assert candidates_a[action]["pairs"] == candidates_b[action]["pairs"]
        assert torch.equal(candidates_a[action]["scores"], candidates_b[action]["scores"])
        assert candidates_a[action]["state"] == candidates_b[action]["state"]

    # The action map insertion order cannot change the constrained proposal.
    actions_one = OrderedDict(((0, "REASSOCIATE"), (1, "ACCEPT_CURRENT")))
    actions_two = OrderedDict(((1, "ACCEPT_CURRENT"), (0, "REASSOCIATE")))
    resolution_one = engine.resolve_actions(
        perception, branch_a, actions=actions_one, proposal=proposal_a
    )
    resolution_two = engine.resolve_actions(
        perception, branch_b, actions=actions_two, proposal=proposal_b
    )
    assert association.calls == 2, "resolve_actions called the transformer again"
    assert resolution_one["existing_track_ids"] == resolution_two["existing_track_ids"]
    assert resolution_one["final_proposal"].proposal_reused is True
    assert resolution_one["final_proposal"].transformer_calls == 0
    assert resolution_one["final_proposal"].scores is proposal_a.scores
    assert torch.equal(
        resolution_one["final_proposal"].scores, proposal_a.scores
    )
    assert _state_signature(branch_a) == _state_signature(branch_b)

    # A supplied proposal is reused by step(), so the complete
    # propose -> resolve -> commit path makes exactly one transformer call.
    committed = engine.step(
        perception,
        branch_a,
        actions=actions_one,
        proposal=proposal_a,
    )
    assert association.calls == 2, "step recomputed the supplied proposal"
    assert committed["proposal_provenance"]["transformer_calls"] == 1
    assert committed["proposal_provenance"]["reassociate_reused_score_matrix"] is True
    assert branch_a.trajectory_rng_calls == 1
    assert branch_a.trajectory_rng_state == proposal_a.rng_state_after
    assert _state_signature(source) == source_before

    # A fresh worker/process with the same stable video seed reproduces the
    # same stream regardless of the caller's process-global RNG state.
    worker_one = make_state()
    worker_two = make_state()
    worker_one_clone = worker_one.clone()
    worker_two_clone = worker_two.clone()
    first = engine.propose(perception, worker_one_clone)
    second = engine.propose(perception, worker_two_clone)
    assert_proposal_equal(first, second)

    print("JEV explicit branch-local RNG isolation: PASS")


if __name__ == "__main__":
    main()
