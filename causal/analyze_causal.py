#!/usr/bin/env python3
"""Paired, horizon-specific analysis for correction and injection audits."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


HORIZONS = (1, 2, 5, 10, 20)


def read_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return pd.DataFrame(rows)


def bootstrap_ci(values, seed=20260930, n=10000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    means = np.empty(n)
    for i in range(n):
        means[i] = rng.choice(values, len(values), replace=True).mean()
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def outcomes(events, run: pd.DataFrame, target_field: str, target_source: str,
             expected_intervention: str):
    rows = []
    if run.empty:
        return pd.DataFrame()
    for event in events:
        scene, gt, frame = str(event["scene"]), int(event["gt_id"]), int(event["frame"])
        target = int(event[target_field])
        current = run[run.event_key.astype(str) == str(event["event_key"])]
        applied = bool(len(current) and str(current.iloc[0].get("intervention")) == expected_intervention)
        apply_reason = "applied" if applied else (str(current.iloc[0].get("intervention")) if len(current) else "missing")
        sub = run[(run.scene == scene) & (run.gt_id == gt)]
        for k in HORIZONS:
            f = sub[sub.frame == frame + k]
            if f.empty:
                rows.append({"event_id": event["event_id"], "scene": scene, "gt_id": gt,
                             "event_frame": frame, "horizon": k, "estimable": False,
                             "applied": applied, "apply_reason": apply_reason,
                             "error": np.nan, "n_future_observations": 0,
                             "cross_view_error": np.nan, "target_id": target})
                continue
            err = (f.selected_pred_id.astype(int) != target).astype(float)
            other = f[f.view.astype(int) != int(event["view"])]
            rows.append({"event_id": event["event_id"], "scene": scene, "gt_id": gt,
                         "event_frame": frame, "horizon": k, "estimable": True,
                         "applied": applied, "apply_reason": apply_reason,
                         "error": float(err.mean()), "n_future_observations": int(len(f)),
                         "cross_view_error": float((other.selected_pred_id.astype(int) != target).mean()) if len(other) else np.nan,
                         "target_id": target, "target_source": target_source})
    return pd.DataFrame(rows)


def paired(effect: pd.DataFrame, sham: pd.DataFrame, label: str):
    key = ["event_id", "horizon"]
    a = effect[effect.estimable & effect.applied].merge(
        sham[sham.estimable & sham.applied], on=key, suffixes=("_effect", "_sham"))
    if a.empty:
        return pd.DataFrame()
    a["difference"] = a.error_effect - a.error_sham
    a["cross_view_difference"] = a.cross_view_error_effect - a.cross_view_error_sham
    a["intervention"] = label
    return a


def summary(paired_df: pd.DataFrame):
    out = []
    for k in HORIZONS:
        x = paired_df[paired_df.horizon == k]
        diffs = x.difference.dropna().to_numpy(float)
        out.append({
            "horizon": k, "events": int(len(diffs)),
            "sham_error": float(x.error_sham.mean()) if len(x) else None,
            "intervention_error": float(x.error_effect.mean()) if len(x) else None,
            "difference_intervention_minus_sham": float(diffs.mean()) if len(diffs) else None,
            "bootstrap_ci95": bootstrap_ci(diffs),
            "same_direction_fraction": float((diffs > 0).mean()) if len(diffs) else None,
            "cross_view_difference": float(x.cross_view_difference.mean()) if len(x) else None,
        })
    return pd.DataFrame(out)


def streaks(paired_df):
    values = []
    for eid, g in paired_df.groupby("event_id"):
        for suffix in ("effect", "sham"):
            x = g.sort_values("horizon")[f"error_{suffix}"].fillna(0).to_numpy()
            cur = best = 0
            for v in x:
                cur = cur + 1 if v > 0 else 0
                best = max(best, cur)
            values.append({"event_id": eid, "condition": suffix, "future_error_streak": best})
    return pd.DataFrame(values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--effect", type=Path, required=True)
    ap.add_argument("--sham", type=Path, required=True)
    ap.add_argument("--target-field", required=True)
    ap.add_argument("--output-csv", type=Path, required=True)
    ap.add_argument("--output-report", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--effect-metrics", type=Path)
    ap.add_argument("--sham-metrics", type=Path)
    args = ap.parse_args()
    events = json.loads(args.events.read_text())["events"]
    effect_kind = "correction" if args.target_field == "p_correct" else "injection"
    effect = outcomes(events, read_jsonl(args.effect), args.target_field, args.label, effect_kind)
    sham = outcomes(events, read_jsonl(args.sham), args.target_field, "sham", "sham")
    p = paired(effect, sham, args.label)
    s = summary(p)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    # The requested CSV contains one row per event/horizon plus the paired error.
    p.to_csv(args.output_csv, index=False)
    scene = (p.groupby(["scene", "horizon"], as_index=False)["difference"].mean()
             if not p.empty else pd.DataFrame())
    applied_effect = int(effect[effect.applied].event_id.nunique()) if not effect.empty else 0
    applied_sham = int(sham[sham.applied].event_id.nunique()) if not sham.empty else 0
    report = [f"# {args.label}", "", f"Events in frozen manifest: {len(events)}",
              f"Events applied in intervention run: {applied_effect}",
              f"Events observed in sham run: {applied_sham}", "",
              "The intervention is compared with its same-event sham. The current event frame is excluded; future identity error is the fraction of matched observations for the same scene/GT at exactly t+k whose selected ID differs from the frozen target ID.", "",
              "## Paired horizon summary", "", s.to_markdown(index=False), "",
              "Bootstrap intervals resample events with seed 20260930. No result is used to select events.", "",
              "## Scene-level paired effects", "", scene.to_markdown(index=False) if not scene.empty else "No estimable scene effects.", ""]
    if args.effect_metrics or args.sham_metrics:
        em = json.loads(args.effect_metrics.read_text()) if args.effect_metrics and args.effect_metrics.exists() else {}
        sm = json.loads(args.sham_metrics.read_text()) if args.sham_metrics and args.sham_metrics.exists() else {}
        report += ["", "## Tracking metrics", "", pd.DataFrame([
            {"condition": "intervention", **em}, {"condition": "sham", **sm}
        ]).to_markdown(index=False)]
    st = streaks(p)
    if not st.empty:
        report += ["", "## Future error streak", "", st.groupby("condition")["future_error_streak"].agg(["count", "mean", "median", "max"]).to_markdown()]
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text("\n".join(report) + "\n")
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
