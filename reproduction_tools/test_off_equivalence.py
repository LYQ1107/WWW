"""CPU tests for the revised OFF evidence gates."""

import json
from pathlib import Path
import tempfile

from off_equivalence import compare_predictions, compare_trace_events


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        native_predictions = root / "native.json"
        traced_predictions = root / "traced.json"
        native = [{"image_id": 1, "category_id": 1, "track_id": 4, "bbox": [1.0, 2.0, 3.0, 4.0], "score": 0.5}]
        traced = [{"image_id": 1, "category_id": 1, "track_id": 4, "bbox": [1.0 + 5e-7, 2.0, 3.0, 4.0], "score": 0.5}]
        native_predictions.write_text(json.dumps(native), encoding="utf-8")
        traced_predictions.write_text(json.dumps(traced), encoding="utf-8")
        report = compare_predictions(native_predictions, traced_predictions)
        assert report["discrete_trajectory_equal"] is True
        assert report["float_evidence_equal"] is True
        assert report["raw_json_equal"] is False

        native[0]["track_id"] = 5
        native_predictions.write_text(json.dumps(native), encoding="utf-8")
        report = compare_predictions(native_predictions, traced_predictions)
        assert report["discrete_trajectory_equal"] is False

        event = {
            "mode": "off",
            "future_gt_access": False,
            "question": "MATCH_DECISION",
            "legal_actions": ["ACCEPT_CURRENT", "START_NEW"],
            "off_action": "ACCEPT_CURRENT",
            "proposed_action": "ACCEPT_CURRENT",
            "committed_action": "ACCEPT_CURRENT",
            "probabilities": {"ACCEPT_CURRENT": 1.0, "START_NEW": 0.0},
            "state_feature_vector": [0.1, 0.2],
            "context": {
                "video_id": 1,
                "frame": 0,
                "view": 0,
                "event_order": 0,
                "detection_index": 0,
                "proposal_track_id": 4,
                "alternate_track_id": None,
                "track_id": None,
                "tracker_state_before": {"active_track_ids": [4]},
            },
        }
        native_trace = root / "native.jsonl"
        traced_trace = root / "traced.jsonl"
        native_trace.write_text(json.dumps(event) + "\n", encoding="utf-8")
        event["state_feature_vector"][0] += 5e-7
        traced_trace.write_text(json.dumps(event) + "\n", encoding="utf-8")
        trace_report = compare_trace_events(native_trace, traced_trace)
        assert trace_report["action_sequence_equal"] is True
        assert trace_report["trace_float_evidence_equal"] is True
    print("OFF evidence gate invariants: PASS")


if __name__ == "__main__":
    main()
