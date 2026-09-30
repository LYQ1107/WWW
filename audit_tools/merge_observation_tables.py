#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()
    paths = sorted(args.parts.glob("*.parquet"))
    if not paths:
        raise SystemExit("no parquet shards")
    frames = [pd.read_parquet(p) for p in paths]
    df = pd.concat(frames, ignore_index=True).sort_values(
        ["scene", "view_id", "frame_id", "gt_instance_id", "pred_track_id"]
    ).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    manifest = {
        "parts": [str(p) for p in paths],
        "output": str(args.output),
        "rows": len(df),
        "scenes": sorted(df.scene.unique().tolist()),
        "matching": "Hungarian cost=1-IoU, accept IoU >= 0.5",
        "diagnostic_not_official_matching": True,
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
