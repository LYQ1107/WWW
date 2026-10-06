"""Fast source-level tests for deterministic intra-video chunking."""

from __future__ import annotations

from pathlib import Path

from jev_counterfactual_v2 import MutableGMTState
from jev_intra_video_chunking import (
    CHUNKING_SCHEMA_VERSION,
    compare_records_exact,
    load_state_snapshot,
    plan_chunks,
    save_state_snapshot,
)


def event(video: int, frame: int, view: int, question: str, row: int):
    return {
        "question": question,
        "legal_actions": ["START_NEW"],
        "context": {
            "video_id": video,
            "frame": frame,
            "view": view,
            "detection_index": row,
        },
    }


def record(frame: int, row: int):
    return {
        "question_type": "MATCH_DECISION",
        "state": {
            "online_context": {
                "video_id": 7,
                "frame": frame,
                "view": 0,
                "detection_index": row,
            }
        },
        "value": frame + row,
    }


def test_plan_boundaries_and_coverage():
    keys = [(7, index, 0) for index in range(5)]
    events = {
        key: [event(7, key[1], 0, "MATCH_DECISION", key[1])]
        for key in keys
    }
    chunks = plan_chunks(
        video_id=7,
        ordered_keys=keys,
        events_by_key=events,
        target_records=2,
    )
    assert [item["decision_count"] for item in chunks] == [2, 2, 1]
    assert [(item["key_start"], item["key_end"]) for item in chunks] == [(0, 2), (2, 4), (4, 5)]


def test_snapshot_preserves_rng_provenance(tmp_path: Path):
    state = MutableGMTState().initialize_trajectory_rng(7)
    path = tmp_path / "chunk.pt"
    digest = save_state_snapshot(
        path,
        state,
        {"video_id": 7, "key_start": 0, "schema_version": CHUNKING_SCHEMA_VERSION},
    )
    loaded, metadata = load_state_snapshot(path)
    assert digest.startswith("sha256:")
    assert metadata["key_start"] == 0
    assert loaded.trajectory_rng_seed == state.trajectory_rng_seed
    assert loaded.trajectory_rng_state == state.trajectory_rng_state
    assert loaded.trajectory_rng_calls == state.trajectory_rng_calls


def test_exact_record_comparison():
    left = [record(0, 0), record(1, 1)]
    right = [record(1, 1), record(0, 0)]
    report = compare_records_exact(left, right)
    assert report["status"] == "PASS"
    assert report["exact_record_matches"] == 2


if __name__ == "__main__":
    test_plan_boundaries_and_coverage()
    test_snapshot_preserves_rng_provenance(Path("/tmp/jev_chunking_test"))
    test_exact_record_comparison()
    print("PASS")
