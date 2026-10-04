"""Evaluate VisionTrack cross-view identity metrics with repository TrackEval.

The original CV kit flattens each scene twice: sequential camera blocks for
CVIDF1 and interleaved camera/frame IDs for CVMA.  This script reproduces those
two input conventions without MATLAB, while preserving the raw per-view GT.
The report labels the TrackEval Identity IDF1 as CVIDF1 and CLEAR MOTA as CVMA
and also stores the unscaled values.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "TrackEval"))
import trackeval  # noqa: E402


class PermissiveVisionTrackDataset(trackeval.datasets.MotChallenge2DBox):
    @staticmethod
    def _check_unique_ids(data, after_preproc=False):
        return None


def read_rows(path: Path) -> List[List[str]]:
    rows = []
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append([field.strip() for field in line.split(",")])
    return rows


def frame_of(row: Sequence[str]) -> int:
    return int(float(row[0]))


def write_flattened(
    manifest: Mapping[str, object],
    root: Path,
    convention: str,
) -> Tuple[Path, Path, Dict[str, int]]:
    gt_root = root / "gt"
    tracker_root = root / "trackers" / "GMT" / "data"
    gt_root.mkdir(parents=True)
    tracker_root.mkdir(parents=True)
    sequences = [str(value) for value in manifest["sequences"]]
    scenes = [str(value) for value in manifest["scenes"]]
    by_scene = {
        scene: sorted(sequence for sequence in sequences if sequence.startswith(scene + "_"))
        for scene in scenes
    }
    lengths: Dict[str, int] = {}
    source_gt = Path(str(manifest["trackeval_gt"]))
    source_track = Path(str(manifest["trackeval_trackers"]))
    for scene, scene_sequences in by_scene.items():
        gt_rows: List[Tuple[int, List[str]]] = []
        track_rows: List[Tuple[int, List[str]]] = []
        offset = 0
        view_count = len(scene_sequences)
        for view_index, sequence in enumerate(scene_sequences):
            gt = read_rows(source_gt / sequence / "gt" / "gt.txt")
            track = read_rows(source_track / "GMT" / "data" / f"{sequence}.txt")
            if convention == "sequential":
                max_frame = max([frame_of(row) for row in gt] + [0])
                convert = lambda frame: frame + offset
                offset += max_frame
            elif convention == "interleaved":
                convert = lambda frame, idx=view_index: frame * view_count + idx
            else:
                raise ValueError(convention)
            gt_rows.extend((convert(frame_of(row)), row) for row in gt)
            track_rows.extend((convert(frame_of(row)), row) for row in track)
        scene_gt = gt_root / scene / "gt" / "gt.txt"
        scene_track = tracker_root / f"{scene}.txt"
        scene_gt.parent.mkdir(parents=True, exist_ok=True)
        scene_track.parent.mkdir(parents=True, exist_ok=True)
        with scene_gt.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            for frame, row in sorted(gt_rows, key=lambda item: (item[0], int(float(item[1][1])))):
                writer.writerow([frame] + row[1:])
        with scene_track.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            for frame, row in sorted(track_rows, key=lambda item: (item[0], int(float(item[1][1])))):
                writer.writerow([frame] + row[1:])
        lengths[scene] = max([frame for frame, _ in gt_rows] + [0])
    return gt_root, root / "trackers", lengths


def run_trackeval(
    gt_root: Path,
    trackers_root: Path,
    output: Path,
    lengths: Mapping[str, int],
    allow_duplicate_gt: bool,
):
    output.mkdir(parents=True)
    evaluator_config = trackeval.Evaluator.get_default_eval_config()
    evaluator_config.update(
        {
            "USE_PARALLEL": False,
            "DISPLAY_LESS_PROGRESS": True,
            "PRINT_ONLY_COMBINED": True,
            "PLOT_CURVES": False,
            "OUTPUT_FOLDER": str(output),
        }
    )
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": str(gt_root),
            "TRACKERS_FOLDER": str(trackers_root),
            "TRACKERS_TO_EVAL": ["GMT"],
            "BENCHMARK": "VisionTrackCrossView",
            "SPLIT_TO_EVAL": "test",
            "SKIP_SPLIT_FOL": True,
            "SEQ_INFO": dict(lengths),
            "OUTPUT_FOLDER": str(output),
        }
    )
    evaluator = trackeval.Evaluator(evaluator_config)
    dataset_class = PermissiveVisionTrackDataset if allow_duplicate_gt else trackeval.datasets.MotChallenge2DBox
    dataset = dataset_class(dataset_config)
    results, messages = evaluator.evaluate(
        [dataset],
        [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()],
    )
    result = results[dataset.get_name()]["GMT"]
    if result is None:
        raise RuntimeError(messages)
    combined = result["COMBINED_SEQ"]["pedestrian"]
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-duplicate-gt", action="store_true")
    args = parser.parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True)
    reports = {}
    for convention, name in (("sequential", "cvidf1"), ("interleaved", "cvma")):
        flat_root = output / name / "inputs"
        gt_root, trackers_root, lengths = write_flattened(manifest, flat_root, convention)
        combined = run_trackeval(
            gt_root,
            trackers_root,
            output / name / "evaluation",
            lengths,
            args.allow_duplicate_gt,
        )
        identity = combined["Identity"]
        clear = combined["CLEAR"]
        reports[name] = {
            "time_axis": convention,
            "CVIDF1": float(identity["IDF1"] * 100.0),
            "CVMA": float(clear["MOTA"] * 100.0),
            "identity_idf1_raw": float(identity["IDF1"]),
            "clear_mota_raw": float(clear["MOTA"]),
            "identity": identity,
            "clear": clear,
            "sequence_lengths": lengths,
        }
    report = {
        "status": "PASS",
        "source_manifest": str(manifest_path.resolve()),
        "allow_duplicate_gt": bool(args.allow_duplicate_gt),
        "gt_policy": "raw per-view GT is read through manifest symlinks; no GT labels changed",
        "metric_policy": "CVIDF1=TrackEval Identity IDF1*100; CVMA=TrackEval CLEAR MOTA*100",
        "reports": reports,
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
