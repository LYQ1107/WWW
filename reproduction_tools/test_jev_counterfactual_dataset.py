"""CPU smoke test for offline future-GT JEV record construction."""

import json
from pathlib import Path
import tempfile

from build_jev_counterfactual_dataset import make_records
from jev_dataset_contract import validate_record


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        annotations = root / "test.json"
        trace = root / "trace.jsonl"
        checkpoint = root / "checkpoint.pth"
        checkpoint.write_bytes(b"frozen-gmt")
        annotations.write_text(
            json.dumps(
                {
                    "videos": [{"id": 1, "file_name": "seq", "view_num": 1}],
                    "images": [
                        {"id": 1, "video_id": 1, "view_id": 1, "frame_id": 1},
                        {"id": 2, "video_id": 1, "view_id": 1, "frame_id": 2},
                    ],
                    "annotations": [
                        {"image_id": 1, "instance_id": 7, "bbox": [0, 0, 10, 10]},
                        {"image_id": 2, "instance_id": 7, "bbox": [1, 0, 10, 10]},
                    ],
                }
            ),
            encoding="utf-8",
        )
        events = [
            {
                "question": "MATCH_DECISION",
                "legal_actions": ["ACCEPT_CURRENT", "REASSOCIATE", "START_NEW"],
                "off_action": "ACCEPT_CURRENT",
                "state_feature_vector": [0.0] * 64,
                "context": {
                    "video_id": 1,
                    "frame": 1,
                    "view": 0,
                    "decision_scope": "match",
                    "proposal_track_id": 10,
                    "alternate_track_id": 20,
                    "bbox_xyxy": [0, 0, 10, 10],
                },
            },
            {
                "question": "MEMORY_DECISION",
                "legal_actions": ["WRITE_MEMORY", "SKIP_MEMORY"],
                "off_action": "WRITE_MEMORY",
                "state_feature_vector": [0.0] * 64,
                "context": {
                    "video_id": 1,
                    "frame": 1,
                    "view": 0,
                    "decision_scope": "memory",
                    "track_id": 10,
                    "bbox_xyxy": [0, 0, 10, 10],
                },
            },
            {
                "question": "MATCH_DECISION",
                "legal_actions": ["ACCEPT_CURRENT", "START_NEW"],
                "off_action": "ACCEPT_CURRENT",
                "state_feature_vector": [0.0] * 64,
                "context": {
                    "video_id": 1,
                    "frame": 2,
                    "view": 0,
                    "decision_scope": "match",
                    "proposal_track_id": 10,
                    "alternate_track_id": None,
                    "bbox_xyxy": [1, 0, 11, 10],
                },
            },
        ]
        trace.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")
        records, stats = make_records(trace, annotations, "sha256:test", 2)
        assert len(records) == 3, (len(records), stats)
        assert stats["MATCH_DECISION"] == 2
        assert stats["MEMORY_DECISION"] == 1
        for record in records:
            validate_record(record, allow_future_gt=True)
        assert all(record["uses_future_gt"] for record in records)
    print("JEV counterfactual dataset invariants: PASS")


if __name__ == "__main__":
    main()
