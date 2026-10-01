#!/usr/bin/env python
"""Compare immutable baseline and association-replay tracker traces."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


def load(path):
    payload = torch.load(path, map_location="cpu")
    return payload["scene"], payload["records"]


def main():
    base = Path(sys.argv[1]); replay = Path(sys.argv[2]); out = Path(sys.argv[3])
    rows = []
    passed = True
    for path in sorted(base.glob("*.pt")):
        other = replay / path.name
        if not other.exists():
            passed = False
            rows.append({"scene": path.stem, "status": "MISSING_REPLAY"})
            continue
        scene_a, a = load(path); scene_b, b = load(other)
        scene_ok = scene_a == scene_b and len(a) == len(b)
        first_diff = None
        if scene_ok:
            for i, (ra, rb) in enumerate(zip(a, b)):
                ia = ra["track_ids"].tolist(); ib = rb["track_ids"].tolist()
                if ia != ib or int(ra["count"]) != int(rb["count"]):
                    first_diff = {"sequence_index": i, "baseline": ia, "replay": ib}
                    scene_ok = False
                    break
        if not scene_ok:
            passed = False
        rows.append({"scene": scene_a, "status": "PASS" if scene_ok else "FAIL",
                     "baseline_records": len(a), "replay_records": len(b),
                     "first_diff": first_diff})
    payload = {"status": "PASS" if passed else "FAIL", "scenes": rows,
               "comparison": "post-association track_ids and counts, exact list equality"}
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
