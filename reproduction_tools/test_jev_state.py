"""CPU invariants for the online JEV state feature schema."""

import torch

from gtr.modeling.jev_state import encode_state, feature_names


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
    try:
        encode_state({"accept_score": float("nan")})
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite state feature was accepted")
    print("JEV state invariants: PASS")


if __name__ == "__main__":
    main()
