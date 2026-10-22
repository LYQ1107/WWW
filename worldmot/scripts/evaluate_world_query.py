#!/usr/bin/env python3
"""Run the official-evaluation prerequisite gate; never emit fake metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from worldmot.evaluation.official_tracking_eval import evaluation_gate


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tracker-output", required=True)
    p.add_argument("--sequence-root", required=True)
    p.add_argument("--trackeval-root")
    p.add_argument("--output", required=True)
    args = p.parse_args(argv)
    try:
        result = evaluation_gate(tracker_output=args.tracker_output,
                                 complete_sequence_root=args.sequence_root,
                                 trackeval_root=args.trackeval_root)
    except RuntimeError as exc:
        result = {"status": "BLOCKED", "reason": str(exc)}
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

