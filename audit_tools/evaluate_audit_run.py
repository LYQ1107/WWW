#!/usr/bin/env python3
"""Evaluate one audit COCO prediction file with the bundled TrackEval.

The released MOTChallenge evaluator is reused unchanged.  This adapter only
stages the read-only VisionTrack GT files and converts the audit JSON's XYWH
boxes to the MOT text container expected by TrackEval.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.symlink_to(src, target_is_directory=src.is_dir())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-json", type=Path, required=True)
    ap.add_argument("--annotations", type=Path, required=True)
    ap.add_argument("--gt-root", type=Path, required=True,
                    help="TrackEval/data/gt/mot_challenge/vision-train")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    gt = json.loads(args.annotations.read_text())
    videos = {v["id"]: v for v in gt["videos"]}
    images = {x["id"]: x for x in gt["images"]}
    image_max_by_seq: dict[str, int] = {}
    for im in gt["images"]:
        video = videos[int(im["video_id"])]
        seq = f'{video["file_name"]}_View{int(im["view_id"])}'
        image_max_by_seq[seq] = max(image_max_by_seq.get(seq, 0), int(im["frame_id"]))
    preds = json.loads(args.pred_json.read_text())
    by_seq: dict[str, list[dict]] = {}
    for p in preds:
        im = images.get(int(p["image_id"]))
        if im is None:
            continue
        video = videos[int(im["video_id"])]
        seq = f'{video["file_name"]}_View{int(im["view_id"])}'
        by_seq.setdefault(seq, []).append((im, p))

    # Stage a self-contained GT root using symlinks.  Some repository archives
    # have both canonical scene_ViewN directories and legacy scene/gt/ViewN.txt;
    # support either layout without copying the dataset.
    staged_gt = args.output / "gt" / "vision-train"
    for seq in sorted(by_seq):
        scene, view = seq.rsplit("_View", 1)
        canonical = args.gt_root / seq
        legacy = args.gt_root / scene
        seqdir = staged_gt / seq
        (seqdir / "gt").mkdir(parents=True, exist_ok=True)
        gt_src = canonical / "gt" / "gt.txt"
        info_src = canonical / "seqinfo.ini"
        if not gt_src.exists():
            gt_src = legacy / "gt" / f"View{view}.txt"
        if not gt_src.exists():
            raise FileNotFoundError(f"GT missing for {seq}: {gt_src}")
        _link(gt_src.resolve(), seqdir / "gt" / "gt.txt")
        # Stage sequence metadata as a small audit-owned file.  A few released
        # GT archives have stale seqLength values even though their GT rows
        # contain later frames (for example 00005garden_View2).  Use the
        # annotation frame extent for this diagnostic subset while leaving the
        # source GT and its provenance untouched.
        n = image_max_by_seq.get(seq, max(int(im["frame_id"]) for im, _ in by_seq[seq]))
        info_text = info_src.read_text() if info_src.exists() else "[Sequence]\n"
        lines = []
        saw_len = False
        for line in info_text.splitlines():
            if line.lower().startswith("seqlength") and "=" in line:
                lines.append(f"seqLength={n}")
                saw_len = True
            else:
                lines.append(line)
        if not saw_len:
            lines.append(f"seqLength={n}")
        (seqdir / "seqinfo.ini").write_text("\n".join(lines).rstrip() + "\n")

    seqmap = staged_gt.parent / "seqmaps" / "vision-train.txt"
    seqmap.parent.mkdir(parents=True, exist_ok=True)
    seqmap.write_text("name\n" + "\n".join(sorted(by_seq)) + "\n")

    tracker_data = args.output / "moteval" / "trainval" / "pred" / "data"
    tracker_data.mkdir(parents=True, exist_ok=True)
    count = 0
    for seq, rows in sorted(by_seq.items()):
        rows.sort(key=lambda z: (int(z[0]["frame_id"]), int(z[1].get("track_id", -1))))
        # The released writer can emit one trailing record when a diagnostic
        # TEST_LEN window ends at a sequence boundary.  Keep the raw audit
        # output unchanged, but never feed a frame outside TrackEval's
        # declared seqLength to the official evaluator.
        max_frame = None
        seqinfo = staged_gt / seq / "seqinfo.ini"
        if seqinfo.exists():
            for line in seqinfo.read_text().splitlines():
                if line.lower().startswith("seqlength") and "=" in line:
                    try:
                        max_frame = int(line.split("=", 1)[1].strip())
                    except ValueError:
                        pass
                    break
        out = tracker_data / f"{seq}.txt"
        with out.open("w") as f:
            for im, p in rows:
                if "track_id" not in p:
                    continue
                if max_frame is not None and int(im["frame_id"]) > max_frame:
                    continue
                bbox = p.get("bbox", [0, 0, 0, 0])
                if len(bbox) != 4:
                    continue
                x, y, w, h = [float(v) for v in bbox]
                score = float(p.get("score", 1.0))
                f.write(f'{int(im["frame_id"])},{int(p["track_id"])},'
                        f"{x:.6f},{y:.6f},{w:.6f},{h:.6f},{score:.6f},-1,-1,-1\n")
                count += 1

    # Import the repository's bundled TrackEval, never the optional pip copy.
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "TrackEval"))
    import trackeval  # noqa: E402

    eval_cfg = trackeval.Evaluator.get_default_eval_config()
    eval_cfg.update({"USE_PARALLEL": False, "NUM_PARALLEL_CORES": 1,
                     "DISPLAY_LESS_PROGRESS": True, "BREAK_ON_ERROR": True})
    ds_cfg = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    ds_cfg.update({
        "GT_FOLDER": str(staged_gt),
        "TRACKERS_FOLDER": str(args.output / "moteval" / "trainval"),
        "OUTPUT_FOLDER": str(args.output / "trackeval"),
        "TRACKERS_TO_EVAL": ["pred"],
        "BENCHMARK": "vision",
        "SPLIT_TO_EVAL": "train",
        "SEQMAP_FILE": str(seqmap),
        "SKIP_SPLIT_FOL": True,
        "TRACKER_SUB_FOLDER": "data",
        "OUTPUT_SUB_FOLDER": "",
        "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
    })
    metrics_cfg = {"METRICS": ["HOTA", "CLEAR", "Identity"]}
    evaluator = trackeval.Evaluator(eval_cfg)
    dataset = trackeval.datasets.MotChallenge2DBox(ds_cfg)
    metrics = [trackeval.metrics.HOTA(metrics_cfg),
               trackeval.metrics.CLEAR(metrics_cfg),
               trackeval.metrics.Identity(metrics_cfg)]
    results = evaluator.evaluate([dataset], metrics)

    def serial(x):
        if hasattr(x, "tolist"):
            return x.tolist()
        raise TypeError(type(x).__name__)

    ret_path = args.output / "trackeval_return.json"
    ret_path.write_text(json.dumps(results, default=serial, indent=2) + "\n")
    overall = results[0]["MotChallenge2DBox"]["pred"]["COMBINED_SEQ"]["pedestrian"]
    metrics = {
        "HOTA": float(overall["HOTA"]["HOTA(0)"].item() if hasattr(overall["HOTA"]["HOTA(0)"], "item") else overall["HOTA"]["HOTA(0)"]) * 100,
        "AssA": float(overall["HOTA"]["AssA"][0]) * 100,
        "MOTA": float(overall["CLEAR"]["MOTA"]) * 100,
        "IDF1": float(overall["Identity"]["IDF1"]) * 100,
        "number_predictions": count,
        "sequence_count": len(by_seq),
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
