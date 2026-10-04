"""CPU tests for JEV data leakage and counterfactual isolation invariants."""

import copy
import json
import tempfile

from jev_counterfactual import CounterfactualBranchRunner
from jev_dataset_contract import digest_state, validate_jsonl, validate_record


def make_record(uses_future_gt=False):
    state = {"observation": {"score": 0.7}, "track_summary": {"age": 4}}
    return {
        "schema_version": 1,
        "dataset": "VisionTrack",
        "sequence": "seq0",
        "frame": 3,
        "view": 1,
        "question_type": "MATCH_DECISION",
        "state_digest": digest_state(state),
        "gmt_checkpoint_sha256": "sha256:checkpoint",
        "legal_actions": ["ACCEPT_CURRENT", "REASSOCIATE", "START_NEW"],
        "state": state,
        "action_outcomes": {
            "ACCEPT_CURRENT": {"utility": 1.0},
            "REASSOCIATE": {"utility": 0.5},
            "START_NEW": {"utility": 0.0},
        },
        "best_actions": ["ACCEPT_CURRENT"],
        "target_probs": [1.0, 0.0, 0.0],
        "horizon": 4,
        "uses_future_gt": uses_future_gt,
    }


def main():
    record = make_record()
    validate_record(record)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl") as handle:
        handle.write(json.dumps(record) + "\n")
        handle.flush()
        assert validate_jsonl(handle.name) == 1

    leaked = make_record(uses_future_gt=True)
    try:
        validate_record(leaked)
    except ValueError:
        pass
    else:
        raise AssertionError("online validator accepted future-GT record")

    source = {"track_ids": [1], "memory": {"writes": 0}}
    original = copy.deepcopy(source)

    def apply_action(branch, action):
        branch["memory"]["writes"] += int(action == "WRITE_MEMORY")
        branch["track_ids"].append(action)
        return branch

    def rollout(branch, horizon):
        branch["horizon"] = horizon
        return branch

    runner = CounterfactualBranchRunner(apply_action, rollout, lambda state: {"score": float(len(state["track_ids"]))})
    results = runner.run(source, ["WRITE_MEMORY", "SKIP_MEMORY"], horizon=4)
    assert source == original
    assert results["WRITE_MEMORY"]["utility"]["score"] == 2.0
    assert results["SKIP_MEMORY"]["utility"]["score"] == 2.0
    print("JEV contract invariants: PASS")


if __name__ == "__main__":
    main()
