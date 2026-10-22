#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow this script to be run directly from the repository checkout without
# requiring an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worldmot.data.visiontrack import VisionTrack
from worldmot.evaluation.evaluate_2d import evaluate_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default="/data/DATASETS/TRACKING/JDE/VisionTrack")
    ap.add_argument("--jsonl", required=True); ap.add_argument("--output", required=True); ap.add_argument("--iou", type=float, default=0.3)
    args = ap.parse_args()
    metrics = evaluate_jsonl(VisionTrack(args.dataset_root), args.jsonl, args.iou)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
