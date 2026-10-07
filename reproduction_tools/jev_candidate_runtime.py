"""Online runtime adapter for ID-free candidate-conditioned MATCH models.

The neural scorer sees only the 64-D mutable GMT state and candidate-local
evidence.  Track IDs are retained outside the network as metadata so the
selected column can be bound back to the native association proposal.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

from jev_candidate_model import CandidateConditionedScorer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(value: Any, name: str) -> float:
    result = float(value)
    if not torch.isfinite(torch.tensor(result, dtype=torch.float32)):
        raise ValueError(f"{name} is non-finite")
    return result


def candidate_feature_rows(
    candidate_track_ids: Sequence[int],
    candidate_scores: Sequence[float],
    *,
    detection_score: float,
    proposal_track_id: int | None,
    alternate_track_id: int | None,
) -> list[list[float]]:
    """Reproduce the ID-free candidate feature contract used in training."""

    track_ids = [int(value) for value in candidate_track_ids]
    scores = [_finite(value, "candidate score") for value in candidate_scores]
    if len(track_ids) != len(scores):
        raise ValueError("candidate ID/score length mismatch")
    if len(track_ids) != len(set(track_ids)):
        raise ValueError("candidate track IDs are not unique")
    if not track_ids:
        return []
    ordered = sorted(scores, reverse=True)
    top = ordered[0]
    second = ordered[1] if len(ordered) > 1 else top
    count = float(len(scores))
    detection_score = _finite(detection_score, "detection score")
    rows = []
    for track_id, score in zip(track_ids, scores):
        rank = 1 + sum(other > score for other in scores)
        rows.append(
            [
                score,
                top - score,
                score - second if rank == 1 else score - top,
                float(rank) / count,
                count,
                detection_score,
                float(proposal_track_id is not None and track_id == int(proposal_track_id)),
                float(alternate_track_id is not None and track_id == int(alternate_track_id)),
            ]
        )
    return rows


def start_new_feature_row(
    candidate_scores: Sequence[float], detection_score: float
) -> list[float]:
    scores = [_finite(value, "candidate score") for value in candidate_scores]
    top = max(scores) if scores else 0.0
    return [
        0.0,
        top,
        -top,
        1.0,
        float(len(scores)),
        _finite(detection_score, "detection score"),
        0.0,
        0.0,
    ]


class CandidateRuntimeScorer:
    """Load and apply one candidate-conditioned checkpoint at runtime."""

    def __init__(self, checkpoint: Path, *, device: str = "cpu"):
        self.checkpoint = Path(checkpoint).resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(self.checkpoint)
        self.device = torch.device(device)
        payload = torch.load(str(self.checkpoint), map_location="cpu")
        if not isinstance(payload, Mapping):
            raise ValueError("candidate checkpoint must contain a mapping")
        model_name = str(payload.get("model_name", ""))
        if model_name not in {"candidate_mlp", "candidate_jev"}:
            raise ValueError(f"unsupported candidate model: {model_name}")
        self.model_name = model_name
        self.metadata = {
            key: value for key, value in payload.items() if key != "model"
        }
        self.metadata["checkpoint"] = str(self.checkpoint)
        self.metadata["checkpoint_sha256"] = sha256(self.checkpoint)
        self.model = CandidateConditionedScorer(
            state_dim=int(payload.get("state_dim", 64)),
            candidate_dim=int(payload.get("candidate_dim", 8)),
            hidden_dim=int(payload.get("hidden_dim", 128)),
            model_name=model_name,
        )
        self.model.load_state_dict(payload["model"])
        self.model.to(self.device).eval()

    @torch.no_grad()
    def predict(
        self,
        state_features: torch.Tensor,
        candidate_track_ids: Sequence[int],
        candidate_scores: Sequence[float],
        *,
        detection_score: float,
        proposal_track_id: int | None,
        alternate_track_id: int | None,
    ) -> Dict[str, Any]:
        state = torch.as_tensor(state_features, dtype=torch.float32).reshape(-1)
        if state.numel() != int(self.metadata.get("state_dim", 64)):
            raise ValueError(
                f"candidate state dimension {state.numel()} does not match checkpoint"
            )
        track_ids = [int(value) for value in candidate_track_ids]
        scores = [_finite(value, "candidate score") for value in candidate_scores]
        rows = candidate_feature_rows(
            track_ids,
            scores,
            detection_score=detection_score,
            proposal_track_id=proposal_track_id,
            alternate_track_id=alternate_track_id,
        )
        keys = [f"track:{track_id}" for track_id in track_ids] + ["START_NEW"]
        features = rows + [start_new_feature_row(scores, detection_score)]
        state_batch = state.to(self.device).unsqueeze(0)
        candidate_batch = torch.tensor(
            [features], dtype=torch.float32, device=self.device
        )
        mask = torch.ones(
            (1, len(features)), dtype=torch.bool, device=self.device
        )
        output = self.model(state_batch, candidate_batch, mask)
        probabilities = output["probs"][0].detach().float().cpu().tolist()
        logits = output["logits"][0].detach().float().cpu().tolist()
        selected = int(torch.argmax(output["probs"][0]).item())
        selected_key = keys[selected]
        selected_track_id = (
            int(track_ids[selected]) if selected < len(track_ids) else None
        )
        return {
            "selected_key": selected_key,
            "selected_track_id": selected_track_id,
            "selected_action": (
                "ACCEPT_CURRENT" if selected_track_id is not None else "START_NEW"
            ),
            "selected_probability": float(probabilities[selected]),
            "candidate_keys": keys,
            "candidate_track_ids": track_ids,
            "candidate_scores": scores,
            "probabilities": {
                key: float(value) for key, value in zip(keys, probabilities)
            },
            "logits": {key: float(value) for key, value in zip(keys, logits)},
        }

