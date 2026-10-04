"""CPU invariants for offline JEV target construction and sequence splits."""

from jev_dataset_contract import validate_record
from jev_dataset_tools import make_record, split_sequences, utility_target


def main():
    actions = ["ACCEPT_CURRENT", "REASSOCIATE", "START_NEW"]
    best, probs = utility_target(
        "MATCH_DECISION", actions, {name: 1.0 for name in actions}
    )
    assert best == ("ACCEPT_CURRENT", "REASSOCIATE", "START_NEW")
    assert abs(sum(probs) - 1.0) < 1e-8

    record = make_record(
        dataset="VisionTrack",
        sequence="00001garden",
        frame=4,
        view=1,
        question_type="MATCH_DECISION",
        state={"observation": {"score": 0.7}, "gmt_evidence": {"margin": 0.1}},
        legal_actions=actions,
        action_outcomes={
            "ACCEPT_CURRENT": {"utility": 0.1},
            "REASSOCIATE": {"utility": 0.8, "metrics": {"IDSW": 0}},
            "START_NEW": {"utility": 0.2},
        },
        gmt_checkpoint_sha256="sha256:test",
        horizon=8,
    )
    assert record["best_actions"] == ["REASSOCIATE"]
    validate_record(record, allow_future_gt=True)
    try:
        validate_record(record, allow_future_gt=False)
    except ValueError as exc:
        assert "future_gt" in str(exc)
    else:
        raise AssertionError("online validator accepted an offline record")

    splits = split_sequences([f"seq-{i:02d}" for i in range(10)])
    assert set().union(*[set(value) for value in splits.values()]) == {
        f"seq-{i:02d}" for i in range(10)
    }
    assert not (set(splits["train"]) & set(splits["val"]))
    assert splits == split_sequences([f"seq-{i:02d}" for i in range(10)])
    print("JEV dataset-tool invariants: PASS")


if __name__ == "__main__":
    main()

