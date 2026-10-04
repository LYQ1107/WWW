"""TrackEval smoke test on an empty prepared prediction set."""

import json
from pathlib import Path
import sys
import tempfile

from evaluate_visiontrack import main
from prepare_visiontrack_predictions import main as prepare


def main_test():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        predictions = root / "predictions.json"
        predictions.write_text("[]", encoding="utf-8")
        prepared = root / "prepared"
        old_argv = sys.argv
        sys.argv = ["prepare", "--predictions", str(predictions), "--output", str(prepared)]
        try:
            prepare()
        finally:
            sys.argv = old_argv
        output = root / "evaluation"
        sys.argv = [
            "evaluate",
            "--prepared",
            str(prepared),
            "--output",
            str(output),
            "--allow-duplicate-gt",
        ]
        try:
            main()
        finally:
            sys.argv = old_argv
        report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        assert report["status"] == "PASS"
        assert report["strict_online"] is True
    print("VisionTrack evaluation invariants: PASS")


if __name__ == "__main__":
    main_test()
