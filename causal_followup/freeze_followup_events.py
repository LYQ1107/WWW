#!/usr/bin/env python
"""Freeze baseline-valid, event-isolated C1b/C2b schedules.

The original first-round manifests are read-only inputs.  Selection is based
on the immutable C0 baseline decision log and current-frame occupancy, then
written to new follow-up manifests with provenance hashes.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAUSAL = ROOT / "causal"
OUT = ROOT / "causal_followup" / "manifests"
OUT.mkdir(parents=True, exist_ok=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path):
    payload = json.loads(path.read_text())
    return payload.get("events", payload if isinstance(payload, list) else [])


def load_decisions(path):
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def event_key(row):
    return str(row["event_key"])


def valid_rows(manifest_rows, decisions):
    by_key = {event_key(x): x for x in decisions}
    by_frame = defaultdict(list)
    for row in decisions:
        by_frame[(row["scene"], int(row["frame"]), int(row["view"]))].append(row)
    selected = []
    reasons = defaultdict(int)
    for event in manifest_rows:
        key = event_key(event)
        row = by_key.get(key)
        if row is None:
            reasons["missing_baseline_decision"] += 1; continue
        if row.get("gt_id") != event.get("gt_id"):
            reasons["gt_mismatch"] += 1; continue
        candidates = {int(x) for x in row.get("candidate_ids", [])}
        target_key = "p_correct" if "p_correct" in event else "p_wrong"
        target = int(event[target_key])
        if target not in candidates:
            reasons[f"{target_key}_inactive"] += 1; continue
        if target == int(row.get("baseline_pred_id", -1)):
            reasons["target_equals_baseline"] += 1; continue
        occupied = False
        for other in by_frame[(row["scene"], int(row["frame"]), int(row["view"]))]:
            if int(other.get("detection_index", -1)) != int(row["detection_index"]):
                if int(other.get("baseline_pred_id", -1)) == target:
                    occupied = True
        if occupied:
            reasons["target_occupied"] += 1; continue
        out = dict(event)
        out.update({
            "baseline_log_key": key,
            "baseline_pred_id_verified": int(row["baseline_pred_id"]),
            "baseline_candidate_ids_verified": sorted(candidates),
            "selection": "C0 baseline-valid; active target; unoccupied current frame",
        })
        selected.append(out)
    return selected, dict(reasons)


def stratified_max(rows, maximum=256):
    if len(rows) <= maximum:
        return rows
    groups = defaultdict(list)
    for row in rows:
        groups[row["scene"]].append(row)
    # Allocate at least one to each scene, then proportional remaining slots.
    total = len(rows)
    quotas = {s: max(1, int(maximum * len(v) / total)) for s, v in groups.items()}
    while sum(quotas.values()) > maximum:
        s = max(quotas, key=lambda k: (quotas[k], k))
        if quotas[s] > 1: quotas[s] -= 1
        else: break
    while sum(quotas.values()) < maximum:
        s = max(groups, key=lambda k: (len(groups[k]) - quotas[k], k))
        if quotas[s] < len(groups[s]): quotas[s] += 1
        else: break
    chosen = []
    for scene, group in sorted(groups.items()):
        group = sorted(group, key=lambda x: hashlib.sha256(event_key(x).encode()).hexdigest())
        chosen.extend(group[:quotas[scene]])
    return sorted(chosen, key=lambda x: event_key(x))


def write(name, rows, source, reasons, extra=None):
    payload = {
        "format": "gmt-event-isolated-followup-v1",
        "name": name,
        "source_manifest": str(source),
        "source_sha256": sha(source),
        "baseline_decisions": str(CAUSAL / "runs/C0_baseline_log/decisions.jsonl"),
        "baseline_decisions_sha256": sha(CAUSAL / "runs/C0_baseline_log/decisions.jsonl"),
        "selection_reasons": reasons,
        "event_count": len(rows),
        "events": rows,
    }
    if extra: payload.update(extra)
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def main():
    correction = CAUSAL / "manifests/correction_events.json"
    injection = CAUSAL / "manifests/injection_events.json"
    decisions = CAUSAL / "runs/C0_baseline_log/decisions.jsonl"
    drows = load_decisions(decisions)
    c1, cr = valid_rows(load_rows(correction), drows)
    c2, ir = valid_rows(load_rows(injection), drows)
    c2 = stratified_max(c2, 256)
    p1 = write("C1b_events.json", c1, correction, cr, {
        "event_semantics": "state repair/quarantine branches; one event per isolated replay",
        "expected_conditions": ["C1b-SHAM", "C1b-A", "C1b-B", "C1b-C"],
    })
    p2 = write("C2b_events.json", c2, injection, ir, {
        "event_semantics": "single wrong-state commit; one event per isolated replay",
        "selection_cap": 256,
        "selection_rule": "scene-stratified deterministic SHA256(event_key) order",
        "expected_conditions": ["C2b-SHAM", "C2b-INJECTION"],
    })
    print(json.dumps({"C1b": {"path": str(p1), "count": len(c1), "reasons": cr},
                      "C2b": {"path": str(p2), "count": len(c2), "reasons": ir}}, indent=2))


if __name__ == "__main__":
    main()
