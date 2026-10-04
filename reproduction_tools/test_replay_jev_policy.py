"""CPU smoke test for frozen GMT proposal replay."""

import json
from pathlib import Path
import tempfile

import torch

from replay_jev_policy import main


def main_test():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        predictions = root / "predictions.json"
        predictions.write_text(
            json.dumps([{"image_id": 1, "track_id": 10, "bbox": [0, 0, 10, 10], "score": 1.0}]),
            encoding="utf-8",
        )
        annotations = root / "annotations.json"
        annotations.write_text(
            json.dumps(
                {
                    "videos": [{"id": 1, "file_name": "seq", "view_num": 1}],
                    "images": [{"id": 1, "video_id": 1, "view_id": 1, "frame_id": 1, "width": 10, "height": 10}],
                    "annotations": [],
                }
            ),
            encoding="utf-8",
        )
        trace = root / "trace.jsonl"
        trace.write_text(
            json.dumps(
                {
                    "question": "MATCH_DECISION",
                    "legal_actions": ["ACCEPT_CURRENT", "REASSOCIATE", "START_NEW"],
                    "off_action": "START_NEW",
                    "state_feature_vector": [0.8, 0.1, 0.0, 0.0] + [0.0] * 60,
                    "context": {
                        "video_id": 1,
                        "frame": 1,
                        "view": 0,
                        "event_order": 0,
                        "proposal_track_id": 10,
                        "alternate_track_id": 20,
                        "bbox_xyxy": [0, 0, 10, 10],
                        "model_image_size": [10, 10],
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        controller = root / "fixed.pth"
        torch.save(
            {
                "model": {},
                "model_name": "fixed_threshold",
                "state_dim": 64,
                "hidden_dim": 128,
                "threshold": 0.2,
            },
            controller,
        )
        output = root / "replayed.json"
        import sys

        old_argv = sys.argv
        sys.argv = [
            "replay_jev_policy.py",
            "--predictions",
            str(predictions),
            "--trace",
            str(trace),
            "--annotations",
            str(annotations),
            "--controller",
            str(controller),
            "--output",
            str(output),
        ]
        try:
            main()
        finally:
            sys.argv = old_argv
        result = json.loads(output.read_text(encoding="utf-8"))
        assert result[0]["track_id"] == 10
        report = json.loads(output.with_suffix(output.suffix + ".manifest.json").read_text())
        assert report["future_gt_access"] is False
        assert report["counts"]["events"] == 1

        oracle_dataset = root / "oracle.jsonl"
        oracle_dataset.write_text(
            json.dumps(
                {
                    "question_type": "MATCH_DECISION",
                    "best_actions": ["REASSOCIATE"],
                    "state": {
                        "online_context": {
                            "video_id": 1,
                            "event_order": 0,
                        }
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
        oracle_output = root / "oracle_replayed.json"
        sys.argv = [
            "replay_jev_policy.py",
            "--predictions",
            str(predictions),
            "--trace",
            str(trace),
            "--annotations",
            str(annotations),
            "--oracle-dataset",
            str(oracle_dataset),
            "--output",
            str(oracle_output),
        ]
        try:
            main()
        finally:
            sys.argv = old_argv
        oracle_result = json.loads(oracle_output.read_text(encoding="utf-8"))
        assert oracle_result[0]["track_id"] == 20
        oracle_report = json.loads(
            oracle_output.with_suffix(oracle_output.suffix + ".manifest.json").read_text()
        )
        assert oracle_report["future_gt_access"] is True
        assert oracle_report["policy_mode"] == "offline_oracle"
    print("JEV policy replay invariants: PASS")


if __name__ == "__main__":
    main_test()
