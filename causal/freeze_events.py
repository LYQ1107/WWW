#!/usr/bin/env python3
"""Freeze correction and injection schedules from the baseline decision log.

Selection is deterministic and uses only current-frame information recorded by
the log pass.  No future outcome, HOTA, or test metric is read here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    with args.decisions.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda r: (r["scene"], int(r["frame"]), int(r["view"]), int(r["detection_index"])))

    corrections, injections = [], []
    last_correction = defaultdict(lambda: -10**9)
    last_injection = defaultdict(lambda: -10**9)
    for r in rows:
        g = r.get("gt_id")
        if g is None:
            continue
        scene, frame = str(r["scene"]), int(r["frame"])
        key = (scene, int(g))
        base = r.get("baseline_pred_id")
        canonical = r.get("canonical_gt_baseline")
        corr = r.get("correct_existing_id")
        wrong = r.get("wrong_existing_id")

        # Natural error correction: the current baseline ID must have an
        # established, wrong canonical identity and a free correct candidate.
        if (base is not None and int(base) >= 0 and canonical is not None
                and int(canonical) != int(g) and corr is not None
                and int(corr) != int(base) and frame - last_correction[key] > 20):
            event = {
                "event_id": f"correction_{len(corrections):06d}",
                "event_key": r["event_key"], "scene": scene,
                "frame": frame, "view": int(r["view"]),
                "image_id": int(r["image_id"]), "detection_index": int(r["detection_index"]),
                "gt_id": int(g), "p_baseline": int(base), "p_correct": int(corr),
                "baseline_canonical_gt": int(canonical),
                "rule": "matched_gt; established_wrong_baseline; free_correct_candidate; >20-frame spacing",
            }
            corrections.append(event)
            last_correction[key] = frame

        # Realistic error injection: current baseline is correct and the
        # wrong candidate is an existing, established identity selected by GMT.
        if (base is not None and int(base) >= 0 and canonical is not None
                and int(canonical) == int(g) and wrong is not None
                and frame - last_injection[key] > 40):
            event = {
                "event_id": f"injection_{len(injections):06d}",
                "event_key": r["event_key"], "scene": scene,
                "frame": frame, "view": int(r["view"]),
                "image_id": int(r["image_id"]), "detection_index": int(r["detection_index"]),
                "gt_id": int(g), "p_baseline": int(base), "p_wrong": int(wrong),
                "baseline_canonical_gt": int(canonical),
                "rule": "matched_gt; correct baseline; free established wrong candidate; >40-frame spacing",
            }
            injections.append(event)
            last_injection[key] = frame

    args.output_root.mkdir(parents=True, exist_ok=True)
    event_dir = args.output_root.parent / "events"
    event_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(event_dir / "baseline_decisions.parquet", index=False)
    common = {
        "source_decisions": str(args.decisions),
        "source_sha256": sha256(args.decisions),
        "selection_uses_future_outcomes": False,
        "purity_threshold": None,
        "row_count": len(rows),
    }
    (args.output_root / "correction_events.json").write_text(json.dumps({**common, "events": corrections}, indent=2) + "\n")
    (args.output_root / "injection_events.json").write_text(json.dumps({**common, "events": injections}, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "correction_events": len(corrections), "injection_events": len(injections)}, indent=2))


if __name__ == "__main__":
    main()
