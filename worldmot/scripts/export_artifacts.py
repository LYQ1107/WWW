#!/usr/bin/env python
"""Materialize reviewable 2-D/3-D logs and prediction-only videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from worldmot.data.visiontrack import VisionTrack
from worldmot.data.synchronized_loader import load_pair


def color(gid: int):
    # Stable palette independent of GT and of process hash randomization.
    return (int((37 * gid + 53) % 255), int((97 * gid + 101) % 255), int((17 * gid + 193) % 255))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="/data/DATASETS/TRACKING/JDE/VisionTrack")
    ap.add_argument("--jsonl", required=True); ap.add_argument("--output-dir", required=True)
    ap.add_argument("--fps", type=float, default=5.0); ap.add_argument("--max-width", type=int, default=960)
    args = ap.parse_args()
    dataset = VisionTrack(args.dataset_root)
    rows = [json.loads(x) for x in Path(args.jsonl).read_text(encoding="utf-8").splitlines() if x.strip()]
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    tracks3d = [{"timestamp": r["timestamp"], "scene_id": r.get("scene_id"), "tracks": r.get("tracks_3d", []), "alignment": r.get("alignment")} for r in rows]
    (outdir / "tracks_3d.json").write_text(json.dumps(tracks3d, indent=2, ensure_ascii=False), encoding="utf-8")
    tracks2d = [{"timestamp": r["timestamp"], "scene_id": r.get("scene_id"), "tracks_2d_by_camera": r.get("tracks_2d_by_camera", {})} for r in rows]
    (outdir / "tracks_2d.json").write_text(json.dumps(tracks2d, indent=2, ensure_ascii=False), encoding="utf-8")
    writers = {}
    written = 0
    try:
        for row in rows:
            images = load_pair(dataset, next(dataset.synchronized_pairs(int(row["scene_id"]), start_frame=int(row["timestamp"]), limit=1)))
            panels = []
            for view in sorted(images):
                im = cv2.cvtColor(images[view], cv2.COLOR_RGB2BGR)
                h, w = im.shape[:2]
                scale = min(1.0, float(args.max_width) / w)
                if scale < 1: im = cv2.resize(im, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
                for track in row.get("tracks_2d_by_camera", {}).get(str(view), row.get("tracks_2d_by_camera", {}).get(view, [])):
                    bbox = track.get("bbox_xywh")
                    if not track.get("visible") or bbox is None: continue
                    x, y, bw, bh = [float(v) * scale for v in bbox]
                    gid = int(track["global_id"]); c = color(gid)
                    cv2.rectangle(im, (int(x), int(y)), (int(x + bw), int(y + bh)), c, 2)
                    cv2.putText(im, f"W{gid} {track.get('status','')}", (int(x), max(16, int(y - 4))), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2, cv2.LINE_AA)
                cv2.putText(im, f"view {view} t={row['timestamp']}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                panels.append(im)
            if len(panels) == 2:
                height = max(x.shape[0] for x in panels)
                panels = [cv2.copyMakeBorder(x, 0, height - x.shape[0], 0, 0, cv2.BORDER_CONSTANT) for x in panels]
                frame = np.concatenate(panels, axis=1)
            else: frame = panels[0]
            if "writer" not in writers:
                writers["writer"] = cv2.VideoWriter(str(outdir / "prediction_overlay.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (frame.shape[1], frame.shape[0]))
            writers["writer"].write(frame); written += 1
    finally:
        for w in writers.values(): w.release()
    summary = {"frames": len(rows), "video_frames": written, "tracks_3d": str(outdir / "tracks_3d.json"), "tracks_2d": str(outdir / "tracks_2d.json"), "video": str(outdir / "prediction_overlay.mp4"), "prediction_only": True}
    (outdir / "artifact_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
