#!/usr/bin/env python3
"""Analyze target-centric Causal Audit v2 replay CSVs.

The event is the resampling unit.  Target observations are paired by frozen
event key and horizon; frames without a matched treated GT are excluded as
``target_not_observable`` rather than assigned zero error.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


SEED = 20260930
BOOTSTRAPS = 10000
HORIZONS = (1, 2, 5, 10, 20)
GATE_HORIZONS = (2, 5, 10, 20)


def _float(row, key):
    value = row.get(key, "")
    if value in (None, "", "None", "null"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(row, key):
    value = row.get(key, "")
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def load_rows(path: Path):
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("condition") == "_META":
                continue
            try:
                horizon = int(row.get("horizon", ""))
            except (TypeError, ValueError):
                continue
            row = dict(row)
            row["horizon"] = horizon
            row["target_error_rate"] = _float(row, "target_error_rate")
            row["target_wrong_id_rate"] = _float(row, "target_wrong_id_rate")
            row["target_correct_rate"] = _float(row, "target_correct_rate")
            row["target_fragment_rate"] = _float(row, "target_fragment_rate")
            row["error_rate"] = _float(row, "error_rate")
            cv = row.get("target_cross_view_consistency", "")
            row["target_cross_view_consistency"] = _float({"x": cv}, "x")
            row["target_n_visible_matched"] = _int(row, "target_n_visible_matched")
            row["target_correct_count"] = _int(row, "target_correct_count")
            row["target_wrong_id_count"] = _int(row, "target_wrong_id_count")
            row["target_other_id_count"] = _int(row, "target_other_id_count")
            row["wrong"] = _int(row, "wrong")
            row["total"] = _int(row, "total")
            rows.append(row)
    return rows


def _pairs(rows, treatment, field):
    index = {(r["event_key"], r["horizon"], r["condition"]): r for r in rows}
    events = sorted({r["event_key"] for r in rows})
    out = {}
    for h in HORIZONS:
        pairs = []
        for event in events:
            sham = index.get((event, h, "SHAM"))
            treat = index.get((event, h, treatment))
            if sham is None or treat is None:
                continue
            sv = sham.get(field)
            tv = treat.get(field)
            if sv is None or tv is None:
                continue
            pairs.append({
                "event_key": event,
                "scene": sham.get("scene", event.split("|", 1)[0]),
                "sham": float(sv),
                "treatment": float(tv),
                "delta": float(tv - sv),
                "sham_row": sham,
                "treatment_row": treat,
            })
        out[h] = pairs
    return out


def _bootstrap(values, rng):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"estimate": None, "ci95_low": None, "ci95_high": None, "n_events": 0}
    draws = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    means = values[draws].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "n_events": int(len(values)),
    }


def _pooled(pairs, metric):
    if not pairs:
        return {"sham": None, "treatment": None, "pooled_delta": None}
    if metric == "target_error_rate":
        sw = sum(x["sham_row"]["target_n_visible_matched"] - x["sham_row"]["target_correct_count"] for x in pairs)
        st = sum(x["sham_row"]["target_n_visible_matched"] for x in pairs)
        tw = sum(x["treatment_row"]["target_n_visible_matched"] - x["treatment_row"]["target_correct_count"] for x in pairs)
        tt = sum(x["treatment_row"]["target_n_visible_matched"] for x in pairs)
    elif metric == "target_wrong_id_rate":
        sw = sum(x["sham_row"]["target_wrong_id_count"] for x in pairs)
        st = sum(x["sham_row"]["target_n_visible_matched"] for x in pairs)
        tw = sum(x["treatment_row"]["target_wrong_id_count"] for x in pairs)
        tt = sum(x["treatment_row"]["target_n_visible_matched"] for x in pairs)
    elif metric == "target_correct_rate":
        sw = sum(x["sham_row"]["target_correct_count"] for x in pairs)
        st = sum(x["sham_row"]["target_n_visible_matched"] for x in pairs)
        tw = sum(x["treatment_row"]["target_correct_count"] for x in pairs)
        tt = sum(x["treatment_row"]["target_n_visible_matched"] for x in pairs)
    elif metric == "target_fragment_rate":
        sw = sum(x["sham_row"]["target_other_id_count"] for x in pairs)
        st = sum(x["sham_row"]["target_n_visible_matched"] for x in pairs)
        tw = sum(x["treatment_row"]["target_other_id_count"] for x in pairs)
        tt = sum(x["treatment_row"]["target_n_visible_matched"] for x in pairs)
    elif metric == "error_rate":
        sw = sum(x["sham_row"]["wrong"] for x in pairs)
        st = sum(x["sham_row"]["total"] for x in pairs)
        tw = sum(x["treatment_row"]["wrong"] for x in pairs)
        tt = sum(x["treatment_row"]["total"] for x in pairs)
    else:
        sw = float(sum(x["sham"] for x in pairs) / len(pairs))
        tw = float(sum(x["treatment"] for x in pairs) / len(pairs))
        return {"sham": sw, "treatment": tw, "pooled_delta": tw - sw}
    return {
        "sham": float(sw / st) if st else None,
        "treatment": float(tw / tt) if tt else None,
        "pooled_delta": float(tw / tt - sw / st) if st and tt else None,
    }


def _scene_effects(pairs):
    out = {}
    for scene in sorted({x["scene"] for x in pairs}):
        subset = [x for x in pairs if x["scene"] == scene]
        out[scene] = {
            "n_events": len(subset),
            "estimate": float(np.mean([x["delta"] for x in subset])),
            "pooled": _pooled(subset, "target_error_rate"),
        }
    return out


def _skip_summary(rows, treatment):
    treatment_rows = [r for r in rows if r.get("condition") == treatment]
    row_counts = Counter()
    event_counts = defaultdict(set)
    for row in treatment_rows:
        reason = row.get("repair_skip_reason") or row.get("skip_reason") or ""
        if reason:
            row_counts[reason] += 1
            event_counts[reason].add(row["event_key"])
        if row.get("target_skip_reason") == "target_not_observable":
            row_counts["target_not_observable"] += 1
            event_counts["target_not_observable"].add(row["event_key"])
    return {
        "row_counts": dict(sorted(row_counts.items())),
        "event_counts": {k: len(v) for k, v in sorted(event_counts.items())},
    }


def summarize(rows, treatment):
    metrics = [
        "target_error_rate",
        "target_wrong_id_rate",
        "target_correct_rate",
        "target_fragment_rate",
        "error_rate",
        "target_cross_view_consistency",
    ]
    rng = np.random.default_rng(SEED)
    out = {
        "treatment": treatment,
        "event_count": len({r["event_key"] for r in rows if r.get("condition") == treatment}),
        "skip_summary": _skip_summary(rows, treatment),
        "metrics": {},
    }
    for metric in metrics:
        metric_out = {}
        paired = _pairs(rows, treatment, metric)
        for h in HORIZONS:
            pairs = paired[h]
            boot = _bootstrap([x["delta"] for x in pairs], rng)
            pooled = _pooled(pairs, metric)
            # Scene effects use the same metric; the pooled field is only a
            # readable count-based summary for target metrics.
            scene = {}
            for s in sorted({x["scene"] for x in pairs}):
                subset = [x for x in pairs if x["scene"] == s]
                scene[s] = {
                    "n_events": len(subset),
                    "estimate": float(np.mean([x["delta"] for x in subset])),
                }
            executable = len({x["event_key"] for x in pairs})
            metric_out[str(h)] = {
                **boot,
                **pooled,
                "executable_events": executable,
                "scene": scene,
            }
        out["metrics"][metric] = metric_out
    return out


def _fmt(x):
    return "NA" if x is None else f"{x:.12g}"


def _table(summary, metric, title):
    lines = [f"### {title}", "", "| Horizon | Executable events | SHAM | Treatment | Paired delta | 95% CI |", "|---:|---:|---:|---:|---:|---|"]
    for h in HORIZONS:
        x = summary["metrics"][metric][str(h)]
        lines.append(
            f"| +{h} | {x['executable_events']} | {_fmt(x['sham'])} | {_fmt(x['treatment'])} | "
            f"{_fmt(x['estimate'])} | [{_fmt(x['ci95_low'])}, {_fmt(x['ci95_high'])}] |"
        )
    return lines


def write_report(path, title, summary):
    lines = [
        f"# {title}",
        "",
        "This is target-centric event-isolated replay over the frozen C1b/C2b event manifest.",
        "The primary endpoint is the treated GT identity's future error rate; the old global frame error is secondary.",
        "",
        f"Frozen event count: **{summary['event_count']}**.",
        f"Bootstrap: seed `{SEED}`, `{BOOTSTRAPS}` event resamples.",
        "",
        "Skip and observability accounting:",
        "",
        "```json",
        json.dumps(summary["skip_summary"], indent=2, sort_keys=True),
        "```",
        "",
    ]
    lines += _table(summary, "target_error_rate", "Primary: treated-identity error rate")
    lines += ["", *_table(summary, "target_correct_rate", "Target correct rate"), ""]
    lines += _table(summary, "target_wrong_id_rate", "Target wrong-injected-ID rate")
    lines += ["", *_table(summary, "target_fragment_rate", "Target fragment/other-ID rate"), ""]
    lines += _table(summary, "error_rate", "Secondary: global frame error rate")
    lines += ["", *_table(summary, "target_cross_view_consistency", "Target cross-view same-ID consistency"), ""]
    lines += ["## Scene-level primary effects", "", "| Horizon | Scene | Events | Paired target-error delta |", "|---:|---|---:|---:|"]
    for h in HORIZONS:
        for scene, value in summary["metrics"]["target_error_rate"][str(h)]["scene"].items():
            lines.append(f"| +{h} | {scene} | {value['n_events']} | {_fmt(value['estimate'])} |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def gate(summary, direction):
    metric = summary["metrics"]["target_error_rate"]
    support = []
    for h in GATE_HORIZONS:
        x = metric[str(h)]
        if direction == "positive":
            ci_ok = x["ci95_low"] is not None and x["ci95_low"] > 0
            scene_ok = sum(v["estimate"] > 0 for v in x["scene"].values()) >= 2
        else:
            ci_ok = x["ci95_high"] is not None and x["ci95_high"] < 0
            scene_ok = sum(v["estimate"] < 0 for v in x["scene"].values()) >= 2
        if ci_ok and scene_ok:
            support.append(h)
    return support


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c2c", type=Path, required=True)
    ap.add_argument("--c1c", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    c2_rows = load_rows(args.c2c)
    c1_rows = load_rows(args.c1c)
    c2 = summarize(c2_rows, "C2c-INJECTION")
    c1 = summarize(c1_rows, "C1c-FULL-STATE-REPAIR")
    out = args.root / "causal_v2"
    (out / "C2c_STATISTICS.json").write_text(json.dumps(c2, indent=2, sort_keys=True) + "\n")
    (out / "C1c_STATISTICS.json").write_text(json.dumps(c1, indent=2, sort_keys=True) + "\n")
    write_report(out / "reports/C2c_TARGET_INJECTION.md", "C2c target-centric isolated injection", c2)
    write_report(out / "reports/C1c_TARGET_STATE_REPAIR.md", "C1c target-centric complete state repair", c1)
    c2_support = gate(c2, "positive")
    c1_support = gate(c1, "negative")
    mechanism = bool(c2_support and c1_support)
    decision = {
        "mechanism_validated": "YES" if mechanism else "NO",
        "C2c_propagation_support_horizons": c2_support,
        "C1c_recovery_support_horizons": c1_support,
        "C2c_event_count": c2["event_count"],
        "C1c_event_count": c1["event_count"],
        "old_STATE_CAUSAL_GO_preserved": "NO-GO",
        "rule": "C2c target-error lower CI >0 and >=2/3 scenes positive at one of +2/+5/+10/+20; C1c target-error upper CI <0 and >=2/3 scenes negative at one of +2/+5/+10/+20",
        "seed": SEED,
        "bootstrap_resamples": BOOTSTRAPS,
    }
    (out / "MECHANISM_GATE.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    matrix = [
        "# Causal Audit v2 mechanism evidence matrix", "",
        "The old `STATE_CAUSAL_GO = NO-GO` is immutable. This is a post-hoc mechanism gate.", "",
        "| Test | Primary endpoint | Result | Gate implication |", "|---|---|---|---|",
        f"| C2c isolated injection | target GT error | support horizons `{c2_support}` | {'PASS' if c2_support else 'FAIL'} |",
        f"| C1c full internal repair | target GT error | support horizons `{c1_support}` | {'PASS' if c1_support else 'FAIL'} |",
        "", "```json", json.dumps(decision, indent=2), "```", "",
    ]
    (out / "MECHANISM_EVIDENCE_MATRIX.md").write_text("\n".join(matrix) + "\n")
    final = [
        "# Final Causal Audit v2 mechanism decision", "",
        "The previous pre-registered result remains `STATE_CAUSAL_GO = NO-GO`.",
        "This post-hoc audit cannot overwrite it.", "",
        f"## Causal Audit v2 result: **MECHANISM_VALIDATED = {'YES' if mechanism else 'NO'}**", "",
        f"C2c propagation support horizons: `{c2_support}`.",
        f"C1c recovery support horizons: `{c1_support}`.", "",
        "The gate requires both conditions to pass at their own eligible horizon.",
    ]
    if mechanism:
        final += ["", "The target-centric mechanism is supported; JEV-GMT design is allowed by this audit, but training remains prohibited until separately authorized."]
    else:
        final += ["", "At least one target-centric mechanism condition failed. Stop the state-recovery/contamination motivation for JEV-GMT. Do not design another state-repair audit to chase the result.", "", "A future decision-formulation audit may be planned without training, using the separate candidate-probability/uncertainty/abstention motivation."]
    (out / "FINAL_MECHANISM_DECISION.md").write_text("\n".join(final) + "\n")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
