"""CPU-only tests for the durable JEV v2 progress/checkpoint boundary."""

from __future__ import annotations

import json

import pytest

from jev_counterfactual_v2 import MutableGMTState
from jev_v2_progress_recovery import (
    Progress,
    append_record,
    load_checkpoint,
    save_checkpoint,
    state_signature_sha256,
    write_progress,
)


def test_progress_records_and_checkpoint_round_trip(tmp_path):
    state = MutableGMTState(next_id=7, active_ids={2, 7}).initialize_trajectory_rng(1)
    progress = Progress(
        frame=42,
        view=1,
        event_order=18,
        completed_events=12,
        total_events=50,
        branch_count=31,
        elapsed_seconds=3.5,
        last_update_utc="2026-10-07T00:00:00+00:00",
        next_key_index=19,
        total_keys=100,
    )
    provenance = {
        "canonical_commit": "frozen",
        "trace_sha256": "trace",
        "chunk_index": 0,
    }
    progress_path = tmp_path / "progress.json"
    records_path = tmp_path / "records.jsonl.partial"
    checkpoint_path = tmp_path / "state.checkpoint.pt"

    append_record(records_path, {"record": 1})
    append_record(records_path, {"record": 2})
    write_progress(progress_path, progress)
    save_checkpoint(
        checkpoint_path,
        state=state,
        provenance=provenance,
        progress=progress,
    )

    assert len(records_path.read_text(encoding="utf-8").splitlines()) == 2
    assert json.loads(progress_path.read_text(encoding="utf-8"))["next_key_index"] == 19
    loaded_state, loaded_progress, loaded_provenance = load_checkpoint(
        checkpoint_path, expected_provenance=provenance
    )
    assert state_signature_sha256(loaded_state) == state_signature_sha256(state)
    assert loaded_progress["completed_events"] == 12
    assert loaded_provenance == provenance


def test_checkpoint_fails_closed_on_provenance_drift(tmp_path):
    state = MutableGMTState().initialize_trajectory_rng(1)
    progress = Progress(
        frame=0,
        view=0,
        event_order=0,
        completed_events=0,
        total_events=1,
        branch_count=0,
        elapsed_seconds=0.0,
        last_update_utc="2026-10-07T00:00:00+00:00",
    )
    checkpoint = tmp_path / "state.checkpoint.pt"
    save_checkpoint(
        checkpoint,
        state=state,
        provenance={"canonical_commit": "frozen"},
        progress=progress,
    )
    with pytest.raises(RuntimeError, match="provenance mismatch"):
        load_checkpoint(
            checkpoint,
            expected_provenance={"canonical_commit": "different"},
        )
