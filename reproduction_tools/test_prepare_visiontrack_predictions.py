"""Preparation smoke test using an empty prediction list (no GT mutation)."""

import json
from pathlib import Path
import sys
import tempfile

from prepare_visiontrack_predictions import main


def main_test():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        predictions = root / "predictions.json"
        predictions.write_text("[]", encoding="utf-8")
        output = root / "prepared"
        old_argv = sys.argv
        sys.argv = [
            "prepare_visiontrack_predictions.py",
            "--predictions",
            str(predictions),
            "--output",
            str(output),
        ]
        try:
            main()
        finally:
            sys.argv = old_argv
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "PASS"
        assert len(manifest["sequences"]) == 44
        assert all(value == 0 for value in manifest["prediction_rows"].values())
        assert (output / "trackeval" / "gt" / manifest["sequences"][0] / "gt").is_dir()
    print("VisionTrack preparation invariants: PASS")


if __name__ == "__main__":
    main_test()

