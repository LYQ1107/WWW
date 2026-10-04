"""CPU tests for the policy split and official-test selection lock."""

import argparse
import json
from pathlib import Path
import tempfile

from create_final_selection_lock import build_lock
from create_jev_policy_split import build_manifest
from jev_dataset_tools import make_record


def record(sequence: str, frame: int):
    return make_record(
        dataset="fixture",
        sequence=sequence,
        frame=frame,
        view=0,
        question_type="MATCH_DECISION",
        state={"feature_vector": [0.1, 0.2, 0.3, 0.4]},
        legal_actions=["ACCEPT_CURRENT", "REASSOCIATE", "START_NEW"],
        action_outcomes={
            "ACCEPT_CURRENT": {"utility": 1.0},
            "REASSOCIATE": {"utility": 0.5},
            "START_NEW": {"utility": 0.0},
        },
        gmt_checkpoint_sha256="sha256:fixture",
        horizon=1,
    )


def write_jsonl(path: Path, sequences):
    with path.open("w", encoding="utf-8") as handle:
        for frame, sequence in enumerate(sequences):
            handle.write(json.dumps(record(sequence, frame)) + "\n")


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "policy.jsonl"
        write_jsonl(source, ["s1", "s2", "s3", "s4"])
        manifest = build_manifest([source], seed=7, val_fraction=0.25)
        assert manifest["official_test_used_for_search"] is False
        assert set(manifest["train_sequences"]).isdisjoint(manifest["val_sequences"])
        split = root / "split.json"
        split.write_text(json.dumps(manifest), encoding="utf-8")
        checkpoint = root / "controller.pth"
        checkpoint.write_bytes(b"fixture checkpoint")
        args = argparse.Namespace(
            checkpoint=checkpoint,
            policy_split=split,
            code_commit="fixture-commit",
            state_schema_version=2,
            counterfactual_engine_version="cached_perception_mutable_association_v2",
            utility_definition="identity_then_future_hota_v2",
            horizon=8,
            jev_architecture="jev",
            hidden_size=128,
            calibration_method="temperature_val_only",
            threshold_baseline="question_threshold",
            mlp_baseline="question_conditioned_mlp",
            hyperparameters='{"seed": 7}',
        )
        lock = build_lock(args)
        assert lock["policy_split_sequence_hash"] == manifest["sequence_hash"]
        assert lock["gmt_checkpoint_sha256"].startswith("sha256:")
    print("JEV selection gate invariants: PASS")


if __name__ == "__main__":
    main()

