"""Online-only feature schema for the JEV decision layer.

The tracker adapter supplies normalized scalar evidence and summaries by name;
this module packs them into the fixed state dimension consumed by JEV and by
the threshold baselines.  It intentionally accepts no GT/evaluator object.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence, Tuple

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
