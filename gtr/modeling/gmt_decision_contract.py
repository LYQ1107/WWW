"""Pure helpers documenting the current GMT threshold decision semantics."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from .jev_decision import legal_actions_for


def gmt_threshold(
    overlap_thresh: float,
    track_length: int,
    *,
    not_mult_thresh: bool,
) -> float:
    """Return the threshold used by ``run_global_tracker_plus``."""

    return (
        float(overlap_thresh)
        if not_mult_thresh
        else float(overlap_thresh) * int(track_length)
    )


def gmt_off_match_action(score: float, threshold: float) -> str:
    """Mirror the source's strict ``traj_score > thresh`` branch."""

    return "ACCEPT_CURRENT" if float(score) > float(threshold) else "START_NEW"


def reassociation_candidate(
    scores: Sequence[float],
    selected_index: int,
    committed_indices: Iterable[int] = (),
) -> Optional[int]:
    """Return one free second-best candidate, enforcing max one reassociation."""

    committed = set(int(index) for index in committed_indices)
    ranking = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    for index in ranking:
        if index != int(selected_index) and index not in committed:
            return index
    return None


def match_legal_actions(has_reassociation_candidate: bool) -> List[str]:
    return legal_actions_for("MATCH_DECISION", can_reassociate=has_reassociation_candidate)
