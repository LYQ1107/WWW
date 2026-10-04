"""Cross-view TrackEval smoke test over empty predictions."""

import json
from pathlib import Path
import sys
import tempfile

from evaluate_crossview_visiontrack import main as evaluate_main
from prepare_visiontrack_predictions import main as prepare_main


def main_test():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        predictions = root / "predictions.json"
        predictions.write_text("[]", encoding="utf-8")
        prepared = root / "prepared"
        old_argv = sys.argv
        sys.argv = [
            "prepare_visiontrack_predictions.py",
            "--predictions",
            str(predictions),
            "--output",
            str(prepared),
        ]
        try:
            prepare_main()
        finally:
            sys.argv = old_argv
        output = root / "crossview"
        sys.argv = [
            "evaluate_crossview_visiontrack.py",
            "--manifest",
            str(prepared / "manifest.json"),
            "--output",
            str(output),
            "--allow-duplicate-gt",
        ]
        try:
            evaluate_main()
        finally:
            sys.argv = old_argv
        report = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        assert report["status"] == "PASS"
        assert set(report["reports"]) == {"cvidf1", "cvma"}
        assert report["reports"]["cvidf1"]["CVIDF1"] == 0.0
        assert report["reports"]["cvma"]["CVMA"] == 0.0
    print("VisionTrack cross-view evaluation invariants: PASS")


if __name__ == "__main__":
    main_test()
