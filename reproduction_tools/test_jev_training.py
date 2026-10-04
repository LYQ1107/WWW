"""Small end-to-end CPU smoke test for the offline JEV trainer."""

import json
from pathlib import Path
import tempfile

from jev_dataset_tools import make_record
from train_jev import main


def main_test():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        dataset = root / "records.jsonl"
        records = []
        for index in range(9):
            records.append(
                make_record(
                    dataset="VisionTrack",
                    sequence=f"seq-{index}",
                    frame=1,
                    view=1,
                    question_type="MATCH_DECISION",
                    state={"feature_vector": [0.8, 0.2, 0.1, 0.1, float(index)]},
                    legal_actions=["ACCEPT_CURRENT", "REASSOCIATE", "START_NEW"],
                    action_outcomes={
                        "ACCEPT_CURRENT": {"utility": 1.0},
                        "REASSOCIATE": {"utility": 0.0},
                        "START_NEW": {"utility": -1.0},
                    },
                    gmt_checkpoint_sha256="sha256:test",
                    horizon=4,
                )
            )
        with dataset.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        output = root / "out"
        import sys

        old_argv = sys.argv
        sys.argv = [
            "train_jev.py",
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--epochs",
            "1",
            "--batch-size",
            "3",
            "--model",
            "jev",
        ]
        try:
            main()
        finally:
            sys.argv = old_argv
        report = json.loads((output / "metrics.json").read_text())
        assert report["future_gt_used_by_model"] is False
        assert (output / "model.pth").exists()
    print("JEV training smoke: PASS")


if __name__ == "__main__":
    main_test()

