#!/usr/bin/env python3
"""Aggregate P50/P95 stage timings without confusing cache replay with live E2E."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def add(store, route, scope, stage, value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return
    if np.isfinite(value) and value >= 0:
        store.setdefault((route, scope, stage), []).append(value)


def read_cache(root, store):
    root = Path(root)
    for path in sorted(root.glob("scene_*/shard_*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        for record in data.get("records", []):
            rt = record.get("runtime", {})
            for key, value in rt.items():
                add(store, "OMEGA_CACHE_BUILD", "TRAIN" if "train" in root.name else "TEST", key, value)


def read_jsonl(root, route, scope, key, store):
    root = Path(root)
    for path in sorted(root.glob("scene_*/timings.jsonl")):
        for line in path.open():
            if not line.strip():
                continue
            row = json.loads(line)
            add(store, route, scope, key, row.get(key))
            for stage, value in (row.get("stages", {}) or {}).items():
                add(store, route, scope, stage, value)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", action="append", default=[])
    p.add_argument("--replay", action="append", default=[])
    p.add_argument("--live", action="append", default=[])
    p.add_argument("--output", required=True)
    a = p.parse_args(); store = {}
    for path in a.cache: read_cache(path, store)
    for path in a.replay: read_jsonl(path, "W3_W4_CACHE_REPLAY", "TEST", "cache_replay_s", store)
    for path in a.live: read_jsonl(path, "LIVE_RGB_TO_ID", "TEST", "live_e2e_s", store)
    rows = []
    for (route, scope, stage), values in sorted(store.items()):
        values = np.asarray(values, dtype=np.float64)
        p50 = float(np.percentile(values, 50)); p95 = float(np.percentile(values, 95))
        rows.append({"route": route, "scope": scope, "stage": stage,
                     "samples": int(len(values)), "p50_s": p50, "p95_s": p95,
                     "p50_fps_pairs": 1.0 / max(p50, 1e-12),
                     "p95_fps_pairs": 1.0 / max(p95, 1e-12)})
    out = Path(a.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        fields = list(rows[0]) if rows else ["route", "scope", "stage", "samples", "p50_s", "p95_s", "p50_fps_pairs", "p95_fps_pairs"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    print(json.dumps({"rows": len(rows), "output": str(out), "routes": sorted({x["route"] for x in rows})}, indent=2))


if __name__ == "__main__":
    main()
