"""CPU invariants for the online JEV state feature schema."""

import torch

from gtr.modeling.jev_state import (
    association_window_length,
    build_state_features,
    build_state_values,
    candidate_entropy,
    count_memory_observations,
    count_track_history,
    encode_state,
    feature_names,
    legacy_acceptance_threshold,
)


def main():
    names = feature_names(64)
    assert names[0:4] == (
        "accept_score", "reassociate_score", "write_memory_score", "reactivate_score"
    )
    vector = encode_state(
        {
            "accept_score": 0.4,
            "reassociate_score": 0.2,
            "track_age_norm": 0.8,
            "unknown_future_field": 99.0,
        }
    )
    assert vector.shape == (64,)
    assert torch.isfinite(vector).all()
    assert abs(vector[0].item() - 0.4) < 1e-6
    assert abs(vector[10].item() - 0.8) < 1e-6

    values = build_state_values(
        accept_score=3.7,
        reassociate_score=1.2,
        threshold=0.1,
        candidate_count=7,
        candidate_entropy=0.6,
        track_count=9,
        track_age=5,
        frame_index=12,
        window_length=20,
        view_index=1,
        can_reassociate=True,
        memory_enabled=True,
        with_iou=False,
        not_mult_thresh=False,
        current_is_unmatched=False,
        memory_count=4,
        track_score=3.7,
        track_length=10,
        score_variance=2.5,
    )
    # Unit-bounded views are clipped, raw trajectory evidence is not.
    assert values["accept_score"] == 1.0
    assert values["reassociate_score"] == 1.0
    assert abs(values["raw_traj_score"] - 3.7) < 1e-8
    assert abs(values["mean_traj_score"] - 0.37) < 1e-8
    assert abs(values["score_minus_threshold"] - 3.6) < 1e-8
    assert abs(values["score_over_threshold"] - 37.0) < 1e-8
    assert values["can_reassociate"] == 1.0
    assert values["memory_enabled"] == 1.0

    assert abs(legacy_acceptance_threshold(0.1, 4, False) - 0.4) < 1e-8
    assert abs(legacy_acceptance_threshold(0.1, 4, True) - 0.1) < 1e-8
    assert candidate_entropy([10.0, 0.0]) < candidate_entropy([1.0, 1.0])
    history = [
        {"assignments": {0: 3, 1: 4}},
        {"assignments": {0: 3, 1: 3}},
    ]
    assert count_track_history(history, 3) == 3
    assert count_track_history(history, None) == 1
    assert count_memory_observations({3: [object(), object()]}, 3) == 3
    assert count_memory_observations({}, 3) == 1
    assert association_window_length(
        history_instances=1, view_num=2, view_index=1,
        first_frame_secondary_view=True,
    ) == 2
    assert association_window_length(
        history_instances=2, view_num=2, view_index=0,
    ) == 1
    assert association_window_length(
        history_instances=3, view_num=2, view_index=1,
    ) == 2

    shared = build_state_features(
        state_dim=64,
        accept_score=3.7,
        reassociate_score=1.2,
        threshold=0.1,
        candidate_count=7,
        candidate_entropy=0.6,
        track_count=9,
        track_age=5,
        frame_index=12,
        window_length=20,
        view_index=1,
        can_reassociate=True,
        memory_enabled=True,
        with_iou=False,
        not_mult_thresh=False,
        current_is_unmatched=False,
        memory_count=4,
        track_score=3.7,
        track_length=10,
        score_variance=2.5,
    )
    assert torch.equal(shared, encode_state(values, 64))

    try:
        encode_state({"accept_score": float("nan")})
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite state feature was accepted")
    print("JEV state invariants: PASS")


if __name__ == "__main__":
    main()
