#!/usr/bin/env python
"""Summarize event-isolated C1b/C2b replay results and apply the causal gate.

The unit of resampling is a frozen event.  Each event has paired SHAM and
treatment outcomes, so bootstrap intervals preserve that pairing.  Scene
summaries are reported separately to make a result from one long scene unable
to masquerade as a cross-scene effect.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "causal_followup" / "results"
REPORTS = ROOT / "causal_followup" / "reports"
TARGET_HORIZONS = (5, 10, 20)
SEED = 20260930
BOOTSTRAPS = 4000


def load_rows(path: Path):
    rows = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("condition") == "_META":
                continue
            try:
                h = int(row.get("horizon", ""))
                wrong = int(float(row.get("wrong", "0")))
                total = int(float(row.get("total", "0")))
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "event_key": row["event_key"],
                    "scene": row["scene"],
                    "condition": row["condition"],
                    "horizon": h,
                    "wrong": wrong,
                    "total": total,
                    "rate": (wrong / total) if total else np.nan,
                    "skip_reason": row.get("skip_reason") or None,
                }
            )
    return rows


def paired(rows, treatment, sham="SHAM"):
    index = {(r["event_key"], r["horizon"], r["condition"]): r for r in rows}
    events = sorted({r["event_key"] for r in rows})
    by_h = {}
    for h in TARGET_HORIZONS:
        pairs = []
        for event in events:
            a = index.get((event, h, sham))
            b = index.get((event, h, treatment))
            if not a or not b or not np.isfinite(a["rate"]) or not np.isfinite(b["rate"]):
                continue
            pairs.append(
                {
                    "event_key": event,
                    "scene": a["scene"],
                    "sham_rate": a["rate"],
                    "treatment_rate": b["rate"],
                    "delta": b["rate"] - a["rate"],
                    "sham_wrong": a["wrong"],
                    "sham_total": a["total"],
                    "treatment_wrong": b["wrong"],
                    "treatment_total": b["total"],
                }
            )
        by_h[h] = pairs
    return by_h


def bootstrap_ci(values, rng):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return {"estimate": None, "ci95_low": None, "ci95_high": None, "n_events": 0}
    estimate = float(values.mean())
    draws = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    means = values[draws].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "estimate": estimate,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "n_events": int(len(values)),
    }


def pooled_stats(pairs):
    sw = sum(x["sham_wrong"] for x in pairs)
    st = sum(x["sham_total"] for x in pairs)
    tw = sum(x["treatment_wrong"] for x in pairs)
    tt = sum(x["treatment_total"] for x in pairs)
    return {
        "sham_wrong": int(sw),
        "sham_total": int(st),
        "sham_error_rate": float(sw / st) if st else None,
        "treatment_wrong": int(tw),
        "treatment_total": int(tt),
        "treatment_error_rate": float(tw / tt) if tt else None,
        "pooled_delta": float(tw / tt - sw / st) if st and tt else None,
    }


def scene_stats(pairs):
    out = {}
    for scene in sorted({x["scene"] for x in pairs}):
        subset = [x for x in pairs if x["scene"] == scene]
        p = pooled_stats(subset)
        p["n_events"] = len(subset)
        out[scene] = p
    return out


def summarize(rows, treatments):
    rng = np.random.default_rng(SEED)
    out = {"target_horizons": list(TARGET_HORIZONS), "treatments": {}}
    for treatment in treatments:
        h_out = {}
        all_pairs = paired(rows, treatment)
        treatment_rows = [r for r in rows if r["condition"] == treatment]
        skip_counts = {}
        for r in treatment_rows:
            if r["total"] == 0:
                reason = r.get("skip_reason") or "no_estimable_total"
                skip_counts[reason] = skip_counts.get(reason, 0) + 1
        event_count = len({r["event_key"] for r in treatment_rows})
        for h, pairs in all_pairs.items():
            boot = bootstrap_ci([x["delta"] for x in pairs], rng)
            h_out[str(h)] = {
                **boot,
                **pooled_stats(pairs),
                "scene": scene_stats(pairs),
            }
        out["treatments"][treatment] = {
            "event_count": event_count,
            "skip_reason_rows": skip_counts,
            "horizons": h_out,
        }
    return out


def fmt(x):
    return "NA" if x is None else f"{100.0 * x:.3f}%"


def report(name, summary, treatment):
    lines = [
        f"# {name}",
        "",
        "This report uses event-isolated counterfactual replay over the frozen "
        "association cache. Each treatment is paired with a SHAM branch from "
        "the same pre-event tracker snapshot. Error rate is wrong/total at the "
        "future horizon; delta is treatment minus SHAM, in percentage points.",
        "",
        f"Treatment: `{treatment}`; bootstrap seed: `{SEED}`; resamples: `{BOOTSTRAPS}`.",
        "",
        "| Horizon | Events | SHAM error | Treatment error | Δ | 95% CI for paired Δ |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for h in TARGET_HORIZONS:
        x = summary["treatments"][treatment]["horizons"][str(h)]
        ci = f"[{fmt(x['ci95_low'])}, {fmt(x['ci95_high'])}]"
        lines.append(
            f"| +{h} | {x['n_events']} | {fmt(x['sham_error_rate'])} | "
            f"{fmt(x['treatment_error_rate'])} | {fmt(x['pooled_delta'])} | {ci} |"
        )
    lines += ["", "## Scene-stratified pooled rates", ""]
    for h in TARGET_HORIZONS:
        x = summary["treatments"][treatment]["horizons"][str(h)]
        lines += [f"### +{h}", "", "| Scene | Events | SHAM | Treatment | Δ |", "|---|---:|---:|---:|---:|"]
        for scene, s in x["scene"].items():
            lines.append(
                f"| {scene} | {s['n_events']} | {fmt(s['sham_error_rate'])} | "
                f"{fmt(s['treatment_error_rate'])} | {fmt(s['pooled_delta'])} |"
            )
        lines.append("")
    metadata = summary["treatments"][treatment]
    lines += [
        "## Executability",
        "",
        f"Frozen event count: **{metadata['event_count']}**.",
        f"Rows with no estimable total by reason: `{json.dumps(metadata['skip_reason_rows'], sort_keys=True)}`.",
        "The paired estimates above include only events with both SHAM and treatment totals at that horizon.",
        "",
    ]
    return "\n".join(lines) + "\n"


def gate(summary):
    # Strong positive injection: every target horizon has a positive paired
    # effect with a 95% lower bound above zero.
    inj = summary["treatments"]["C2b-INJECTION"]["horizons"]
    c2_positive_horizons = [
        h for h in TARGET_HORIZONS
        if inj[str(h)]["estimate"] is not None and inj[str(h)]["ci95_low"] > 0
    ]
    c2_scene_horizons = [
        h for h in TARGET_HORIZONS
        if sum(
            inj[str(h)]["scene"][s]["pooled_delta"] > 0
            for s in inj[str(h)]["scene"]
        ) >= 2
    ]
    c2_positive = bool(c2_positive_horizons)
    c2_scene = bool(set(c2_positive_horizons) & set(c2_scene_horizons))
    c2_strong = bool(c2_positive and c2_scene)

    repair_candidates = {}
    for treatment in ("C1b-B", "C1b-C"):
        x = summary["treatments"][treatment]["horizons"]
        negative_ci = all(x[str(h)]["ci95_high"] < 0 for h in TARGET_HORIZONS)
        negative_scenes = all(
            sum(x[str(h)]["scene"][s]["pooled_delta"] < 0 for s in x[str(h)]["scene"]) >= 2
            for h in TARGET_HORIZONS
        )
        repair_candidates[treatment] = {
            "negative_ci_all_target_horizons": bool(negative_ci),
            "negative_in_at_least_two_scenes_all_target_horizons": bool(negative_scenes),
            "strong": bool(negative_ci and negative_scenes),
        }
    c1_strong = any(v["strong"] for v in repair_candidates.values())
    return {
        "C2b_strong_positive": c2_strong,
        "C2b_positive_horizons_ci_lower_gt_zero": c2_positive_horizons,
        "C2b_at_least_two_scenes_positive_horizons": c2_scene_horizons,
        "C1b_repair_candidates": repair_candidates,
        "C1b_strong_repair": c1_strong,
        "overall": "STRONG_GO" if c1_strong and c2_strong else "STOP_NO_STRONG_CAUSAL_GATE",
        "rule": "C2b lower 95% CI > 0 at at least one of +5/+10/+20 and >=2/3 scenes positive at that horizon; C1b-B or C1b-C upper 95% CI < 0 and >=2/3 scenes negative at +5/+10/+20",
    }


def main():
    c1 = load_rows(OUT / "C1b_state_repair.csv")
    c2 = load_rows(OUT / "C2b_isolated_injection.csv")
    c1_summary = summarize(c1, ["C1b-A", "C1b-B", "C1b-C"])
    c2_summary = summarize(c2, ["C2b-INJECTION"])
    summary = {
        "format": "gmt-causal-followup-statistics-v1",
        "seed": SEED,
        "bootstrap_resamples": BOOTSTRAPS,
        "C1b": c1_summary,
        "C2b": c2_summary,
    }
    combined = {"treatments": {**c1_summary["treatments"], **c2_summary["treatments"]}}
    summary["gate"] = gate(combined)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (OUT / "FOLLOWUP_STATISTICS.json").write_text(json.dumps(summary, indent=2) + "\n")
    (REPORTS / "C1b_STATE_REPAIR.md").write_text(report("C1b isolated state-repair follow-up", c1_summary, "C1b-C") + "\nThe companion assignment-only and quarantine branches are included in `FOLLOWUP_STATISTICS.json`.\n")
    (REPORTS / "C2b_ISOLATED_INJECTION.md").write_text(report("C2b isolated single-error injection", c2_summary, "C2b-INJECTION"))
    print(json.dumps(summary["gate"], indent=2))


if __name__ == "__main__":
    main()
