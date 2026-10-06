"""Regression checks for native stale-bank row filtering and promotion."""

import torch

from jev_counterfactual_v2 import (
    MutableGMTState,
    build_reactivation_proposal,
    reactivation_candidates,
    subset_perception_payload,
)


class RecordingEngine:
    def __init__(self):
        self.payload_rows = None

    def propose(self, payload, state):
        self.payload_rows = list(payload["source_detection_indices"])
        return "proposal"


def main():
    payload = {
        "video_id": 1,
        "frame": 12,
        "view": 1,
        "pred_boxes": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "detection_scores": torch.tensor([0.1, 0.2, 0.3]),
        "reid_features": torch.arange(6, dtype=torch.float32).reshape(3, 2),
        "proposal_metadata": {"scores": [0.1, 0.2, 0.3], "pred_classes": [1, 1, 1]},
    }
    subset = subset_perception_payload(payload, [2, 0])
    assert subset["source_detection_indices"] == [2, 0]
    assert subset["pred_boxes"].shape == (2, 4)
    assert subset["proposal_metadata"]["scores"] == [0.3, 0.1]
    assert subset["proposal_metadata"]["pred_classes"] == [1, 1]

    state = MutableGMTState(memory_bank_size=2)
    state.memory = {3: [torch.tensor([1.0, 0.0]), torch.tensor([3.0, 0.0])]}
    candidate_ids, recent_ids = reactivation_candidates(state, bank_size=2)
    assert recent_ids == set()
    assert candidate_ids == [3]
    assert torch.equal(state.reactivation_bank[3], torch.tensor([2.0, 0.0]))

    engine = RecordingEngine()
    proposal = build_reactivation_proposal(engine, subset, state, candidate_ids, None)
    assert proposal == "proposal"
    assert engine.payload_rows == [2, 0]
    print("JEV reactivation row-filter semantics: PASS")


if __name__ == "__main__":
    main()
