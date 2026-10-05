"""CPU test for fail-closed shard provenance merging."""

import json
from pathlib import Path
import tempfile

from jev_dataset_tools import make_record
from merge_jev_jsonl import PROVENANCE_FIELDS, merge


def _manifest(source: Path, *, commit: str) -> dict:
    return {
        "status": "PASS",
        "gmt_checkpoint_sha256": "sha256:checkpoint",
        "config_sha256": "sha256:config",
        "source_commit": commit,
        "counterfactual_engine_version": "cached_perception_mutable_association_v2",
        "state_schema_version": 2,
        "utility_definition": "utility",
        "association_backend": "formal_gmt_transformer",
        "annotations_sha256": "sha256:annotations",
        "cache_sha256": "sha256:cache",
        "trace_sha256": "sha256:trace",
        "horizon": 32,
        "derived_horizons": [1, 8, 16, 32],
        "prelock": False,
        "selection_authority": True,
        "official_result_authority": False,
        "official_test_generation_authorized": False,
        "official_test_lock_sha256": None,
        "official_test_selection_protocol_sha256": None,
        "output": str(source),
    }


def _write_shard(path: Path, *, commit: str) -> None:
    record = make_record(
        dataset="fixture",
        sequence="s1",
        frame=0,
        view=0,
        question_type="MATCH_DECISION",
        state={"feature_vector": [0.1]},
        legal_actions=["ACCEPT_CURRENT", "REASSOCIATE", "START_NEW"],
        action_outcomes={
            "ACCEPT_CURRENT": {"utility": 1.0},
            "REASSOCIATE": {"utility": 0.5},
            "START_NEW": {"utility": 0.0},
        },
        gmt_checkpoint_sha256="sha256:checkpoint",
        horizon=32,
    )
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    path.with_suffix(path.suffix + ".manifest.json").write_text(
        json.dumps(_manifest(path, commit=commit)), encoding="utf-8"
    )


def main() -> None:
    assert len(PROVENANCE_FIELDS) >= 10
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        left = root / "left.jsonl"
        right = root / "right.jsonl"
        _write_shard(left, commit="same")
        _write_shard(right, commit="same")
        output = root / "merged.jsonl"
        result = merge([left, right], output)
        assert result["status"] == "PASS"
        _write_shard(right, commit="different")
        try:
            merge([left, right], root / "refused.jsonl")
        except ValueError as exc:
            assert "REFUSE_TO_MERGE" in str(exc)
        else:
            raise AssertionError("provenance mismatch was accepted")
    print("Shard provenance merge invariants: PASS")


if __name__ == "__main__":
    main()
