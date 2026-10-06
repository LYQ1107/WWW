"""Regression test for the formal H=8 shard finalizer boundary."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from finalize_jev_full_h8 import finalize
from jev_dataset_tools import make_record


def _fixture_record(video_id: int = 8, event_order: int = 1) -> dict:
    state = {
        "feature_vector": [0.0] * 64,
        "online_context": {
            "video_id": video_id,
            "frame": 0,
            "view": 0,
            "event_order": event_order,
            "detection_index": 0,
        },
        "counterfactual_engine": "cached_perception_mutable_association_v2",
        "association_backend": "formal_gmt_transformer",
        "perception_cache_version": "frozen_perception_cache_v2",
    }
    return make_record(
        dataset="VisionTrack",
        sequence="fixture",
        frame=0,
        view=0,
        question_type="MATCH_DECISION",
        state=state,
        legal_actions=["ACCEPT_CURRENT", "REASSOCIATE", "START_NEW"],
        action_outcomes={
            "ACCEPT_CURRENT": {"utility": 1.0},
            "REASSOCIATE": {"utility": 0.5},
            "START_NEW": {"utility": 0.0},
        },
        gmt_checkpoint_sha256="fixture-checkpoint",
        horizon=8,
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        shard_root = root / "shards"
        video_root = shard_root / "video_08"
        partition_root = root / "partition"
        video_root.mkdir(parents=True)
        partition_root.mkdir()

        record = _fixture_record()
        (video_root / "records.jsonl").write_text(
            json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest = {
            "status": "COMPLETE",
            "video_id": 8,
            "source_main_decisions": 1,
            "records": 1,
            "horizon": 8,
            "sampling": False,
            "truncation": False,
            "gmt_checkpoint_sha256": "sha256:fixture-checkpoint",
            "source_commit": "fixture-source",
            "cache_index_sha256": "sha256:cache",
            "source_trace_sha256": "sha256:trace",
            "annotations_sha256": "sha256:annotations",
            "association_backend": "formal_gmt_transformer",
            "formal_gmt_association_adapter": True,
            "counterfactual_engine": "cached_perception_mutable_association_v2",
            "state_schema_version": 2,
            "utility_definition": "fixture-utility",
        }
        (video_root / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
        )
        partition = {
            "status": "COMPLETE",
            "total_main_decisions": 1,
            "stats": {"video_count": 1},
            "videos": {"8": {"main_decisions": 1}},
        }
        (partition_root / "partition_manifest.json").write_text(
            json.dumps(partition, sort_keys=True) + "\n", encoding="utf-8"
        )

        report = finalize(
            shard_root=shard_root,
            partition_root=partition_root,
            output=root / "merged.jsonl",
            output_manifest=root / "merged.manifest.json",
        )
        assert report["status"] == "PASS"
        assert report["record_count"] == 1
    print("Formal H=8 finalizer invariants: PASS")

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        shard_root = root / "shards"
        partition_root = root / "partition"
        partition_root.mkdir()
        for video_id, source_commit in ((8, "fixture-source"), (9, "fixture-patch")):
            video_root = shard_root / f"video_{video_id:02d}"
            video_root.mkdir(parents=True)
            record = _fixture_record(video_id=video_id, event_order=video_id)
            (video_root / "records.jsonl").write_text(
                json.dumps(record, sort_keys=True) + "\n", encoding="utf-8"
            )
            manifest = {
                "status": "COMPLETE",
                "video_id": video_id,
                "source_main_decisions": 1,
                "records": 1,
                "horizon": 8,
                "sampling": False,
                "truncation": False,
                "gmt_checkpoint_sha256": "sha256:fixture-checkpoint",
                "source_commit": source_commit,
                "cache_index_sha256": "sha256:cache",
                "source_trace_sha256": "sha256:trace",
                "annotations_sha256": "sha256:annotations",
                "association_backend": "formal_gmt_transformer",
                "formal_gmt_association_adapter": True,
                "counterfactual_engine": "cached_perception_mutable_association_v2",
                "state_schema_version": 2,
                "utility_definition": "fixture-utility",
            }
            (video_root / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
            )
        partition = {
            "status": "COMPLETE",
            "total_main_decisions": 2,
            "stats": {"video_count": 2},
            "videos": {"8": {"main_decisions": 1}, "9": {"main_decisions": 1}},
        }
        (partition_root / "partition_manifest.json").write_text(
            json.dumps(partition, sort_keys=True) + "\n", encoding="utf-8"
        )
        strict_output = root / "strict.jsonl"
        try:
            finalize(
                shard_root=shard_root,
                partition_root=partition_root,
                output=strict_output,
            )
        except ValueError as exc:
            assert "mixed-source merge" in str(exc), exc
        else:
            raise AssertionError("mixed source commits must fail closed by default")

        compatibility_path = root / "compatibility.json"
        compatibility_path.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "equivalence_gate": "PASS",
                    "base_source_commit": "fixture-source",
                    "allowed_source_commits": ["fixture-source", "fixture-patch"],
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        report = finalize(
            shard_root=shard_root,
            partition_root=partition_root,
            output=root / "compatible.jsonl",
            source_compatibility_manifest=compatibility_path,
        )
        assert report["status"] == "PASS"
        assert report["source_commit"] is None
        assert report["source_commits"] == ["fixture-patch", "fixture-source"]
        assert report["source_compatibility"] == "EXPLICIT_ALLOWLIST_PASS"
    print("Formal H=8 explicit source compatibility invariants: PASS")


if __name__ == "__main__":
    main()
