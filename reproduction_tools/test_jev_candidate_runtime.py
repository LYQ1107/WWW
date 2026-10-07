"""CPU contract tests for ID-free candidate-conditioned runtime binding."""

from pathlib import Path
import tempfile
import math

import torch

from jev_candidate_model import CandidateConditionedScorer
from jev_candidate_runtime import (
    CandidateRuntimeScorer,
    candidate_feature_rows,
    start_new_feature_row,
)


def main():
    ids = [12, 7]
    scores = [0.8, 0.2]
    rows = candidate_feature_rows(
        ids,
        scores,
        detection_score=0.9,
        proposal_track_id=12,
        alternate_track_id=7,
    )
    assert all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=1e-7)
        for left, right in zip(rows[0], [0.8, 0.0, 0.6, 0.5, 2.0, 0.9, 1.0, 0.0])
    )
    assert all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=1e-7)
        for left, right in zip(rows[1], [0.2, 0.6, -0.6, 1.0, 2.0, 0.9, 0.0, 1.0])
    )
    assert all(
        math.isclose(left, right, rel_tol=0.0, abs_tol=1e-7)
        for left, right in zip(
            start_new_feature_row(scores, 0.9),
            [0.0, 0.8, -0.8, 1.0, 2.0, 0.9, 0.0, 0.0],
        )
    )

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "candidate.pth"
        model = CandidateConditionedScorer(
            state_dim=4, candidate_dim=8, hidden_dim=16, model_name="candidate_mlp"
        )
        torch.save(
            {
                "model_name": "candidate_mlp",
                "state_dim": 4,
                "candidate_dim": 8,
                "hidden_dim": 16,
                "model": model.state_dict(),
            },
            path,
        )
        scorer = CandidateRuntimeScorer(path)
        result = scorer.predict(
            torch.zeros(4),
            ids,
            scores,
            detection_score=0.9,
            proposal_track_id=12,
            alternate_track_id=7,
        )
        assert result["selected_key"] in {"track:12", "track:7", "START_NEW"}
        assert set(result["probabilities"]) == {"track:12", "track:7", "START_NEW"}
    print("JEV candidate runtime invariants: PASS")


if __name__ == "__main__":
    main()
