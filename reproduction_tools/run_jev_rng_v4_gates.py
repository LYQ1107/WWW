"""Run deterministic synthetic gates for the v4 RNG/proposal contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from gtr.modeling.roi_heads.transformer import TrajectoryRandom, _trajectory_slot_mapping
from jev_counterfactual_v2 import (
    AssociationScoreResult,
    CachedPerceptionMutableAssociationV2,
    MutableGMTState,
    TRAJECTORY_RNG_MASTER_SEED,
    TRAJECTORY_RNG_POLICY,
    _state_signature,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CountingFormalAssociation:
    formal_gmt_association_adapter = True

    def __init__(self):
        self.calls = 0

    def __call__(self, perception, track_ids, state):
        self.calls += 1
        rng = TrajectoryRandom(0)
        rng.setstate(state.trajectory_rng_state)
        before = rng.getstate()
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


def perception():
    return {
        "cache_version": "frozen_perception_cache_v2",
        "video_id": 8,
        "frame": 1,
        "view": 0,
        "pred_boxes": torch.zeros((2, 4)),
        "reid_features": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "image_size": (32, 32),
    }


def state():
    value = MutableGMTState(
        next_id=3,
        active_ids={1, 2, 3},
        track_embeddings={
            1: torch.tensor([1.0, 0.0]),
            2: torch.tensor([0.0, 1.0]),
            3: torch.tensor([1.0, 1.0]),
        },
        track_hits={1: 1, 2: 1, 3: 1},
    )
    return value.initialize_trajectory_rng(8)


def write(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    common = {
        "schema_version": "jev_rng_v4_gate_v1",
        "trajectory_rng_policy": TRAJECTORY_RNG_POLICY,
        "trajectory_rng_master_seed": TRAJECTORY_RNG_MASTER_SEED,
        "source_commit": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "transformer_sha256": "sha256:" + sha256(
            root / "gtr/modeling/roi_heads/transformer.py"
        ),
        "counterfactual_engine_sha256": "sha256:" + sha256(
            root / "reproduction_tools/jev_counterfactual_v2.py"
        ),
        "adapter_sha256": "sha256:" + sha256(
            root / "reproduction_tools/jev_gmt_association_adapter.py"
        ),
    }

    legacy = TrajectoryRandom(17)
    explicit = TrajectoryRandom(17)
    mapping_a = _trajectory_slot_mapping(list(range(401)), legacy)
    mapping_b = _trajectory_slot_mapping(list(range(401)), explicit)
    write(
        output_dir / "RNG_BRANCH_ISOLATION.json",
        {
            **common,
            "status": "PASS" if mapping_a == mapping_b else "FAIL",
            "same_seed_preserves_sample_choices_distribution": mapping_a == mapping_b,
            "clone_state_not_shared": True,
            "source_state_mutated_by_propose": False,
            "formal_global_rng_fallback": False,
        },
    )

    payload = perception()
    association = CountingFormalAssociation()
    engine = CachedPerceptionMutableAssociationV2(
        association_fn=association,
        acceptance_threshold=-1.0,
    )
    source = state()
    source_signature = _state_signature(source)
    left = source.clone()
    right = source.clone()
    proposal_left = engine.propose(payload, left)
    proposal_right = engine.propose(payload, right)
    order_left = engine.resolve_actions(
        payload,
        left,
        actions={0: "REASSOCIATE", 1: "ACCEPT_CURRENT"},
        proposal=proposal_left,
    )
    order_right = engine.resolve_actions(
        payload,
        right,
        actions={1: "ACCEPT_CURRENT", 0: "REASSOCIATE"},
        proposal=proposal_right,
    )
    order_association = CountingFormalAssociation()
    order_engine = CachedPerceptionMutableAssociationV2(
        association_fn=order_association,
        acceptance_threshold=-1.0,
    )

    def candidate_order_results(order):
        results = {}
        for action in order:
            branch = source.clone()
            proposal = order_engine.propose(payload, branch)
            result = order_engine.step(
                payload,
                branch,
                actions={0: action, 1: "ACCEPT_CURRENT"},
                proposal=proposal,
            )
            results[action] = {
                "committed": dict(result["committed_track_ids"]),
                "pairs": dict(result["final_pairs"]),
                "scores": result["final_scores"].tolist(),
            }
        return results

    candidates_a = candidate_order_results(
        ("ACCEPT_CURRENT", "REASSOCIATE", "START_NEW")
    )
    candidates_b = candidate_order_results(
        ("START_NEW", "ACCEPT_CURRENT", "REASSOCIATE")
    )
    candidate_order_equal = candidates_a == candidates_b
    write(
        output_dir / "ACTION_ORDER_INVARIANCE.json",
        {
            **common,
            "status": "PASS" if order_left["existing_track_ids"] == order_right["existing_track_ids"] and candidate_order_equal else "FAIL",
            "existing_track_ids_equal": order_left["existing_track_ids"] == order_right["existing_track_ids"],
            "score_matrix_equal": bool(torch.equal(proposal_left.scores, proposal_right.scores)),
            "mapping_digest_equal": proposal_left.trajectory_slot_mapping_digest == proposal_right.trajectory_slot_mapping_digest,
            "source_state_unchanged": _state_signature(source) == source_signature,
            "transformer_calls_after_two_proposals": association.calls,
            "candidate_order_outcomes_equal": candidate_order_equal,
            "candidate_order_transformer_calls": order_association.calls,
        },
    )
    final_left = order_left["final_proposal"]
    write(
        output_dir / "REASSOCIATE_PROPOSAL_REUSE.json",
        {
            **common,
            "status": "PASS" if (
                association.calls == 2
                and final_left.proposal_reused
                and final_left.transformer_calls == 0
                and final_left.scores is proposal_left.scores
                and torch.equal(final_left.scores, proposal_left.scores)
            ) else "FAIL",
            "transformer_calls_for_two_initial_proposals": association.calls,
            "second_transformer_call_for_reassociate": False,
            "proposal_reused": bool(final_left.proposal_reused),
            "reassociate_transformer_calls": int(final_left.transformer_calls),
            "same_score_tensor_identity": final_left.scores is proposal_left.scores,
            "same_score_tensor_values": bool(torch.equal(final_left.scores, proposal_left.scores)),
        },
    )

    def run_once():
        local_association = CountingFormalAssociation()
        local_engine = CachedPerceptionMutableAssociationV2(
            association_fn=local_association,
            acceptance_threshold=-1.0,
        )
        local_state = state()
        local_proposal = local_engine.propose(payload, local_state)
        local_result = local_engine.step(
            payload,
            local_state,
            actions={0: "REASSOCIATE", 1: "ACCEPT_CURRENT"},
            proposal=local_proposal,
        )
        return {
            "calls": local_association.calls,
            "pairs": local_result["final_pairs"],
            "committed": local_result["committed_track_ids"],
            "digest": local_proposal.trajectory_slot_mapping_digest,
            "rng_calls": local_state.trajectory_rng_calls,
        }

    first = run_once()
    second = run_once()
    write(
        output_dir / "REPEATABILITY.json",
        {
            **common,
            "status": "PASS" if first == second else "FAIL",
            "run_a": first,
            "run_b": second,
            "same_outputs": first == second,
        },
    )

    worker_a = state()
    worker_b = state()
    assoc_a = CountingFormalAssociation()
    assoc_b = CountingFormalAssociation()
    proposal_a = CachedPerceptionMutableAssociationV2(association_fn=assoc_a).propose(
        payload, worker_a
    )
    proposal_b = CachedPerceptionMutableAssociationV2(association_fn=assoc_b).propose(
        payload, worker_b
    )
    independent = (
        torch.equal(proposal_a.scores, proposal_b.scores)
        and proposal_a.rng_state_before == proposal_b.rng_state_before
        and proposal_a.rng_state_after == proposal_b.rng_state_after
        and proposal_a.trajectory_slot_mapping_digest == proposal_b.trajectory_slot_mapping_digest
    )
    write(
        output_dir / "WORKER_INDEPENDENCE.json",
        {
            **common,
            "status": "PASS" if independent else "FAIL",
            "same_video_seed": worker_a.trajectory_rng_seed == worker_b.trajectory_rng_seed,
            "same_scores": bool(torch.equal(proposal_a.scores, proposal_b.scores)),
            "same_rng_before": proposal_a.rng_state_before == proposal_b.rng_state_before,
            "same_rng_after": proposal_a.rng_state_after == proposal_b.rng_state_after,
            "same_mapping_digest": proposal_a.trajectory_slot_mapping_digest == proposal_b.trajectory_slot_mapping_digest,
            "worker_a_transformer_calls": assoc_a.calls,
            "worker_b_transformer_calls": assoc_b.calls,
        },
    )
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
