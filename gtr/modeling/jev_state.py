"""Online-only feature schema for the JEV decision layer.

The tracker adapter supplies normalized scalar evidence and summaries by name;
this module packs them into the fixed state dimension consumed by JEV and by
the threshold baselines.  It intentionally accepts no GT/evaluator object.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence, Tuple

import torch
from torch import Tensor


# State positions through ``state_validity_flag`` preserve the v1 schema.
# The former reserved tail is now used for raw accumulated association
# evidence.  Those values are deliberately not clipped to [0, 1].
STATE_SCHEMA_VERSION = 2

STATE_FEATURE_NAMES: Tuple[str, ...] = (
    "accept_score",
    "reassociate_score",
    "write_memory_score",
    "reactivate_score",
    "accept_threshold",
    "unmatched_mass",
    "top1_top2_margin",
    "hungarian_conflict_count",
    "proposal_count_norm",
    "track_count_norm",
    "track_age_norm",
    "track_hits_norm",
    "track_view_count_norm",
    "track_score_mean",
    "track_score_std",
    "track_last_seen_norm",
    "memory_age_norm",
    "memory_count_norm",
    "memory_score_mean",
    "memory_score_std",
    "view_index_norm",
    "frame_index_norm",
    "window_length_norm",
    "cross_view_disagreement",
    "threshold_action_disagreement",
    "bank_action_disagreement",
    "detector_score",
    "reid_norm",
    "reid_variance",
    "box_area_norm",
    "box_center_x_norm",
    "box_center_y_norm",
    "box_aspect_ratio",
    "same_track_recent_count",
    "same_track_recent_mean",
    "same_track_recent_std",
    "candidate_count_norm",
    "candidate_entropy",
    "candidate_unmatched_mass",
    "candidate_reactivation_mass",
    "current_is_unmatched",
    "has_old_track",
    "can_reassociate",
    "memory_enabled",
    "with_iou",
    "not_mult_thresh",
    "camera_count_norm",
    "time_since_reactivation_norm",
    "reassociation_calls_norm",
    "memory_reads_norm",
    "memory_writes_norm",
    "track_fragment_count_norm",
    "track_id_switch_count_norm",
    "contamination_risk_proxy",
    "recovery_risk_proxy",
    "state_validity_flag",
    "raw_traj_score",
    "mean_traj_score",
    "log1p_traj_score",
    "score_minus_threshold",
    "score_over_threshold",
    "track_length_norm",
    "raw_score_variance",
)

STATE_FEATURE_INDEX = {name: index for index, name in enumerate(STATE_FEATURE_NAMES)}


def feature_names(state_dim: int = 64) -> Tuple[str, ...]:
    if state_dim < 4:
        raise ValueError("JEV state_dim must leave room for threshold-prefix features")
    if state_dim <= len(STATE_FEATURE_NAMES):
        return STATE_FEATURE_NAMES[:state_dim]
    return STATE_FEATURE_NAMES + tuple(
        f"reserved_{index}" for index in range(state_dim - len(STATE_FEATURE_NAMES))
    )


def safe_unit(value: float) -> float:
    """Clip only values whose schema semantics are unit-bounded."""

    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def candidate_entropy(scores: Sequence[float] | Tensor) -> float:
    """Return the bounded GMT candidate entropy used by the state schema."""

    values = torch.as_tensor(scores, dtype=torch.float32).reshape(-1)
    if values.numel() == 0:
        return 0.0
    probabilities = torch.softmax(values, dim=0)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum()
    return float(entropy.item())


def legacy_acceptance_threshold(
    base_threshold: float,
    track_length: float,
    not_mult_thresh: bool,
) -> float:
    """Match GMT's legacy OFF threshold without changing JEV features.

    ``NOT_MULT_THRESH=False`` is the historical scaled-threshold mode:
    GMT compares accumulated trajectory evidence with
    ``base_threshold * track_length``.  The feature named
    ``accept_threshold`` deliberately remains the unscaled base threshold.
    """

    threshold = float(base_threshold)
    if bool(not_mult_thresh):
        return threshold
    return threshold * max(1.0, float(track_length))


def count_track_history(
    history: Iterable[Mapping[str, object]], track_id: int | None
) -> int:
    """Count a track's occurrences in the bounded association history."""

    if track_id is None:
        return 1
    count = 0
    target = int(track_id)
    for item in history:
        assignments = item.get("assignments", {})
        if isinstance(assignments, Mapping):
            count += sum(int(value) == target for value in assignments.values())
    return max(1, int(count))


def count_memory_observations(
    memory: Mapping[int, Sequence[object]], track_id: int | None
) -> int:
    """Return GMT's memory-count feature at a decision boundary.

    ``id_reid_dict`` contains only observations written after the initial
    track seed.  The native runtime nevertheless reports the initial
    observation as memory count one, so the canonical feature contract uses
    ``1 + len(memory[track_id])`` for an existing track.
    """

    if track_id is None:
        return 1
    values = memory.get(int(track_id), ())
    return max(1, 1 + len(values))


def association_window_length(
    *,
    history_instances: int,
    view_num: int,
    view_index: int,
    first_frame_secondary_view: bool = False,
) -> int:
    """Reproduce GTRRCNN's association-window length at a decision.

    The production first-frame path passes ``T`` directly to the state
    builder, while later multi-view paths pass ``T // view_num``.  The
    caller supplies the number of historical instance slices already in the
    mutable state; this helper keeps that batching convention out of every
    replay implementation.
    """

    total = max(0, int(history_instances)) + 1
    if first_frame_secondary_view:
        return max(1, total)
    return max(1, total // max(1, int(view_num)))


def build_state_values(
    *,
    accept_score: float,
    reassociate_score: float,
    threshold: float,
    candidate_count: int,
    candidate_entropy: float,
    track_count: int,
    track_age: float,
    frame_index: float,
    window_length: float,
    view_index: float,
    can_reassociate: bool,
    memory_enabled: bool,
    with_iou: bool,
    not_mult_thresh: bool,
    has_old_track: bool = False,
    current_is_unmatched: bool = False,
    memory_count: float = 0.0,
    track_score: float = 0.0,
    track_length: float = 1.0,
    score_variance: float = 0.0,
    track_hits: float | None = None,
) -> Mapping[str, float]:
    """Build the canonical online JEV state-value mapping.

    This function is the single source of truth for the value semantics that
    precede :func:`encode_state`.  Runtime inference, traced OFF collection,
    and any closed-loop replay must call this helper rather than maintaining a
    second hand-written copy of the feature transformations.
    """

    raw_accept_score = float(accept_score)
    raw_reassociate_score = float(reassociate_score)
    raw_threshold = float(threshold)
    accept = safe_unit(raw_accept_score)
    reassociate = safe_unit(raw_reassociate_score)
    top1, top2 = sorted((accept, reassociate), reverse=True)
    safe_length = max(1.0, float(track_length))
    safe_threshold = raw_threshold if abs(raw_threshold) > 1e-8 else 1e-8
    hits = float(track_age if track_hits is None else track_hits)
    return {
        "accept_score": accept,
        "reassociate_score": reassociate,
        "write_memory_score": safe_unit(track_score),
        "reactivate_score": reassociate,
        "accept_threshold": safe_unit(raw_threshold),
        "unmatched_mass": safe_unit(1.0 - max(accept, reassociate)),
        "top1_top2_margin": safe_unit(top1 - top2),
        "candidate_count_norm": safe_unit(float(candidate_count) / 16.0),
        "candidate_entropy": safe_unit(candidate_entropy),
        "track_count_norm": safe_unit(float(track_count) / 128.0),
        "track_age_norm": safe_unit(float(track_age) / 128.0),
        "track_hits_norm": safe_unit(hits / 128.0),
        "memory_count_norm": safe_unit(float(memory_count) / 64.0),
        "track_score_mean": safe_unit(track_score),
        "track_score_std": 0.0,
        "frame_index_norm": safe_unit(float(frame_index) / max(1.0, float(window_length))),
        "window_length_norm": safe_unit(float(window_length) / 32.0),
        "view_index_norm": safe_unit(float(view_index) / 8.0),
        "current_is_unmatched": float(bool(current_is_unmatched)),
        "has_old_track": float(bool(has_old_track)),
        "can_reassociate": float(bool(can_reassociate)),
        "memory_enabled": float(bool(memory_enabled)),
        "with_iou": float(bool(with_iou)),
        "not_mult_thresh": float(bool(not_mult_thresh)),
        "state_validity_flag": 1.0,
        # Accumulated association evidence is not a probability.  Preserve the
        # raw evidence and its transforms exactly as the production GMT path.
        "raw_traj_score": raw_accept_score,
        "mean_traj_score": raw_accept_score / safe_length,
        "log1p_traj_score": math.log1p(max(0.0, raw_accept_score)),
        "score_minus_threshold": raw_accept_score - raw_threshold,
        "score_over_threshold": raw_accept_score / safe_threshold,
        "track_length_norm": safe_unit(safe_length / 128.0),
        "raw_score_variance": max(0.0, float(score_variance)),
    }


def build_state_features(*, state_dim: int = 64, **kwargs) -> Tensor:
    """Build and encode one canonical online JEV state vector."""

    return encode_state(build_state_values(**kwargs), state_dim)


def encode_state(values: Mapping[str, float], state_dim: int = 64) -> Tensor:
    """Pack finite normalized online values into a stable float tensor."""

    names = feature_names(state_dim)
    output = torch.zeros(state_dim, dtype=torch.float32)
    for name, value in values.items():
        if name not in names:
            continue
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"non-finite JEV state feature: {name}")
        output[names.index(name)] = value
    return output
