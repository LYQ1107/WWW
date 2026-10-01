#!/usr/bin/env python3
"""Build the predeclared causal gate and final stop/continue status."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def ci(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return (None, None)
    rng = np.random.default_rng(20260930)
    means = np.asarray([rng.choice(x, len(x), replace=True).mean() for _ in range(10000)])
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def table(path: Path, expected: str):
    df = pd.read_csv(path)
    out = []
    for h in (1, 2, 5, 10, 20):
        x = df[df.horizon == h]
        vals = x.difference.dropna().to_numpy(float)
        lo, hi = ci(vals)
        scene = x.groupby("scene").difference.mean() if len(x) else pd.Series(dtype=float)
        same = ((scene < 0) if expected == "negative" else (scene > 0)).mean() if len(scene) else np.nan
        strong = bool((hi is not None and hi < 0) if expected == "negative" else (lo is not None and lo > 0))
        out.append({"horizon": h, "events": len(vals), "mean_difference": float(vals.mean()) if len(vals) else None,
                    "ci95_low": lo, "ci95_high": hi, "scene_same_direction_fraction": float(same) if np.isfinite(same) else None,
                    "direction_expected": expected, "paired_ci_excludes_zero_expected_direction": strong})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--correction", type=Path, required=True)
    ap.add_argument("--injection", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--status", type=Path, required=True)
    args = ap.parse_args()
    c = table(args.correction, "negative")
    i = table(args.injection, "positive")
    common = []
    for h in (5, 10, 20):
        cr = c[c.horizon == h].iloc[0]
        ir = i[i.horizon == h].iloc[0]
        if (bool(cr.paired_ci_excludes_zero_expected_direction)
                and bool(ir.paired_ci_excludes_zero_expected_direction)
                and float(cr.scene_same_direction_fraction or 0) >= .5
                and float(ir.scene_same_direction_fraction or 0) >= .5):
            common.append(h)
    correction_any = bool(c.paired_ci_excludes_zero_expected_direction.any())
    injection_any = bool(i.paired_ci_excludes_zero_expected_direction.any())
    if common:
        gate = "STRONG GO"
        next_step = "Both interventions meet the predeclared lag-5/10/20 paired gate. Phase D may proceed; no test metrics have been opened for model selection."
    elif correction_any or injection_any:
        gate = "CONDITIONAL GO"
        next_step = "Only one causal intervention meets the predeclared evidence direction. Stop before implementing JEV-GMT and report the ambiguity."
    else:
        gate = "NO-GO"
        next_step = "Neither intervention meets the predeclared causal gate. Stop; do not implement JEV-GMT."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    body = ["# Causal evidence matrix", "", "The primary endpoint is future identity error. Current event frames are excluded. Bootstrap intervals resample frozen events with seed 20260930.", "", "## C1 Oracle correction (expected negative)", "", c.to_markdown(index=False), "", "## C2 Error injection (expected positive)", "", i.to_markdown(index=False), "", "## Gate", "", f"**{gate}**", "", next_step, "", "The event schedules were frozen from baseline decisions before either intervention outcome was read.", ""]
    args.output.write_text("\n".join(body))
    status = {"causal_gate": gate, "common_strong_horizons": common, "correction_any_strong": correction_any, "injection_any_strong": injection_any, "next_step": next_step}
    args.status.parent.mkdir(parents=True, exist_ok=True)
    args.status.write_text("# Final project status\n\n" + json.dumps(status, indent=2) + "\n\n" + next_step + "\n")
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
