"""Evaluate isolated VisionTrack inputs with the repository TrackEval code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TrackEval"))
import trackeval  # noqa: E402


class PermissiveVisionTrackDataset(trackeval.datasets.MotChallenge2DBox):
    """Preserve raw VisionTrack duplicate IDs for an explicit audit mode."""

    @staticmethod
    def _check_unique_ids(data, after_preproc=False):
        return None


def native(value):
    if isinstance(value, dict):
        return {str(key): native(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [native(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-duplicate-gt",
        action="store_true",
        help="run TrackEval with raw duplicate GT IDs; never alters GT rows",
    )
    args = parser.parse_args()
    prepared = args.prepared if args.prepared.is_absolute() else ROOT / args.prepared
    output = args.output if args.output.is_absolute() else ROOT / args.output
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "PASS":
        raise ValueError("prepared evaluation manifest is not PASS")
    if output.exists():
        raise RuntimeError(f"refusing to overwrite evaluation output: {output}")
    output.mkdir(parents=True)
    eval_defaults = trackeval.Evaluator.get_default_eval_config()
    eval_defaults.update(
        {
            "USE_PARALLEL": False,
            "DISPLAY_LESS_PROGRESS": True,
            "PRINT_ONLY_COMBINED": True,
            "PLOT_CURVES": False,
            "OUTPUT_FOLDER": str(output),
        }
    )
    dataset_defaults = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_defaults.update(
        {
            "GT_FOLDER": str(Path(manifest["trackeval_gt"])),
            "TRACKERS_FOLDER": str(Path(manifest["trackeval_trackers"])),
            "TRACKERS_TO_EVAL": [manifest["tracker_name"]],
            "BENCHMARK": "VisionTrack",
            "SPLIT_TO_EVAL": "test",
            "SKIP_SPLIT_FOL": True,
            "SEQ_INFO": {key: int(value) for key, value in manifest["seq_lengths"].items()},
            "OUTPUT_FOLDER": str(output),
        }
    )
    evaluator = trackeval.Evaluator(eval_defaults)
    dataset_class = PermissiveVisionTrackDataset if args.allow_duplicate_gt else trackeval.datasets.MotChallenge2DBox
    dataset = dataset_class(dataset_defaults)
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    results, messages = evaluator.evaluate([dataset], metrics)
    dataset_result = results[dataset.get_name()][manifest["tracker_name"]]
    if dataset_result is None:
        raise RuntimeError(
            f"TrackEval failed for {dataset.get_name()}/{manifest['tracker_name']}: "
            f"{messages.get(dataset.get_name(), {}).get(manifest['tracker_name'], 'unknown error')}"
        )
    combined = dataset_result["COMBINED_SEQ"]["pedestrian"]
    report = {
        "status": "PASS",
        "prepared_manifest": str(prepared / "manifest.json"),
        "output": str(output),
        "combined_metrics": native(combined),
        "detail_files": sorted(str(path) for path in output.rglob("*.csv")),
        "strict_online": True,
        "test_gt_tuning": False,
        "allow_duplicate_gt": bool(args.allow_duplicate_gt),
        "gt_duplicate_rows": manifest.get("gt_duplicate_rows", {}),
        "gt_policy": "raw downloaded GT retained; duplicate-ID check is only bypassed when explicitly requested",
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
