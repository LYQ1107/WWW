"""Convert current GMT COCO-style outputs into isolated VisionTrack eval inputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Dict, Iterable, List, Mapping, Tuple


ROOT = Path(__file__).resolve().parents[1]
DATASET = Path("/data/DATASETS/TRACKING/JDE/VisionTrack")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=source.is_dir())


def duplicate_gt_rows(path: Path) -> int:
    counts = defaultdict(int)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            counts[(int(float(row[0])), int(float(row[1])))] += 1
    return sum(count - 1 for count in counts.values() if count > 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    args = parser.parse_args()
    predictions_path = args.predictions if args.predictions.is_absolute() else ROOT / args.predictions
    dataset = args.dataset if args.dataset.is_absolute() else ROOT / args.dataset
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)
    annotation_path = dataset / "annotations" / f"{args.split}.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite prepared evaluation input: {output}")

    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {int(image["id"]): image for image in annotations["images"]}
    sequence_images: Dict[str, List[Mapping[str, object]]] = {}
    sequence_views: Dict[str, set] = {}
    for image in images.values():
        sequence = str(image["file_name"]).split("/", 1)[0]
        sequence_images.setdefault(sequence, []).append(image)
        sequence_views.setdefault(sequence, set()).add(int(image["view_id"]))
    expected_sequences = sorted(sequence_images)
    if not expected_sequences:
        raise ValueError("annotation file contains no images")
    raw_predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    if not isinstance(raw_predictions, list):
        raise ValueError("predictions JSON must be a list")

    by_sequence: Dict[str, List[Tuple[int, int, float, float, float, float, float]]] = {
        sequence: [] for sequence in expected_sequences
    }
    for item in raw_predictions:
        if "image_id" not in item or "track_id" not in item or "bbox" not in item:
            raise ValueError("every prediction must contain image_id, track_id and bbox")
        image_id = int(item["image_id"])
        if image_id not in images:
            raise ValueError(f"prediction references unknown image_id={image_id}")
        image = images[image_id]
        sequence = str(image["file_name"]).split("/", 1)[0]
        bbox = list(item["bbox"])
        if len(bbox) != 4:
            raise ValueError("prediction bbox must be xywh with four values")
        x, y, width, height = (float(value) for value in bbox)
        score = float(item.get("score", 1.0))
        by_sequence[sequence].append(
            (int(image["frame_id"]), int(item["track_id"]), x, y, width, height, score)
        )

    output.mkdir(parents=True)
    trackeval_root = output / "trackeval"
    gt_root = trackeval_root / "gt"
    tracker_root = trackeval_root / "trackers" / "GMT" / "data"
    seq_lengths = {}
    gt_duplicate_rows = {}
    for sequence in expected_sequences:
        source_sequence = dataset / args.split / sequence
        source_gt = source_sequence / "gt"
        if not source_gt.is_dir():
            raise FileNotFoundError(source_gt)
        frames = sequence_images[sequence]
        length = max(int(image["frame_id"]) for image in frames)
        seq_lengths[sequence] = length
        gt_duplicate_rows[sequence] = duplicate_gt_rows(source_gt / "gt.txt")
        link(source_gt, gt_root / sequence / "gt")
        destination = tracker_root / f"{sequence}.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = sorted(by_sequence[sequence], key=lambda row: (row[0], row[1]))
        with destination.open("w", encoding="utf-8") as handle:
            for frame, track_id, x, y, width, height, score in rows:
                handle.write(
                    f"{frame},{track_id},{x:.6f},{y:.6f},{width:.6f},{height:.6f},"
                    f"{score:.6f},-1,-1,-1\n"
                )

    # Prepare the original cross-view converter's exact input layout.  Its
    # input track boxes are XYXY while COCO predictions are XYWH.
    cross_root = output / "crossview_input"
    cross_gt = cross_root / "gt"
    cross_track = cross_root / "track"
    scenes = sorted({sequence.rsplit("_", 1)[0] for sequence in expected_sequences})
    for scene in scenes:
        scene_sequences = sorted(
            sequence for sequence in expected_sequences if sequence.startswith(scene + "_")
        )
        for sequence in scene_sequences:
            view = sequence.rsplit("_", 1)[1]
            source_gt = dataset / args.split / sequence / "gt" / f"{view}.txt"
            if not source_gt.is_file():
                raise FileNotFoundError(source_gt)
            link(source_gt, cross_gt / scene / "gt" / f"{view}.txt")
            destination = cross_track / scene / f"{view}.txt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            rows = sorted(by_sequence[sequence], key=lambda row: (row[0], row[1]))
            with destination.open("w", encoding="utf-8") as handle:
                for frame, track_id, x, y, width, height, _score in rows:
                    handle.write(
                        f"{frame},{track_id},{x:.6f},{y:.6f},{x + width:.6f},{y + height:.6f}\n"
                    )
    (cross_root / "seqs.txt").write_text("MOT16\n" + "\n".join(scenes) + "\n", encoding="utf-8")

    manifest = {
        "status": "PASS",
        "split": args.split,
        "dataset": str(dataset),
        "annotation": str(annotation_path),
        "annotation_sha256": sha256(annotation_path),
        "predictions": str(predictions_path),
        "predictions_sha256": sha256(predictions_path),
        "trackeval_gt": str(gt_root),
        "trackeval_trackers": str(trackeval_root / "trackers"),
        "tracker_name": "GMT",
        "seq_lengths": seq_lengths,
        "sequences": expected_sequences,
        "scenes": scenes,
        "prediction_rows": {sequence: len(by_sequence[sequence]) for sequence in expected_sequences},
        "gt_duplicate_rows": gt_duplicate_rows,
        "gt_duplicate_policy": "raw GT retained; strict TrackEval rejects these IDs unless explicitly overridden",
        "crossview_input": str(cross_root),
        "gt_policy": "symlink downloaded split GT; no GT rows or labels changed",
        "box_policy": "TrackEval=XYWH; cross-view input=XYXY as required by original converter",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
