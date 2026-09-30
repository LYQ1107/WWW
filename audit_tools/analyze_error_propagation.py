#!/usr/bin/env python3
"""A4 descriptive identity-error persistence and cross-view cascade audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--figures", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.table).sort_values(["scene", "gt_instance_id", "frame_id", "view_id"])

    # GT-centric dominant predicted ID, then contiguous wrong-observation
    # streaks in frame/view order.
    gt_counts = df.groupby(["scene", "gt_instance_id", "pred_track_id"]).size().rename("n").reset_index()
    gt_dom = gt_counts.sort_values(
        ["scene", "gt_instance_id", "n", "pred_track_id"], ascending=[True, True, False, True]
    ).drop_duplicates(["scene", "gt_instance_id"])
    gt_dom = gt_dom.rename(columns={"pred_track_id": "gt_dominant_pred"})[["scene", "gt_instance_id", "gt_dominant_pred"]]
    df = df.merge(gt_dom, on=["scene", "gt_instance_id"], how="left")
    df["gt_wrong_identity"] = df.pred_track_id != df.gt_dominant_pred

    streaks = []
    for (scene, gid), part in df.groupby(["scene", "gt_instance_id"]):
        part = part.sort_values(["frame_id", "view_id"])
        current = 0
        for wrong in part.gt_wrong_identity.tolist():
            if wrong:
                current += 1
            elif current:
                streaks.append(current); current = 0
        if current:
            streaks.append(current)
    lengths = np.asarray(streaks, dtype=float)
    survival_rows = []
    for k in [1, 2, 5, 10, 20]:
        survival_rows.append({
            "threshold_frames_or_observations": k,
            "error_streak_count": int(np.sum(lengths >= k)),
            "P_error_streak_ge_k": float(np.mean(lengths >= k)) if len(lengths) else float("nan"),
            "definition": "consecutive matched observations in frame/view order",
        })
    survival_rows.append({
        "threshold_frames_or_observations": "summary",
        "error_streak_count": len(lengths),
        "P_error_streak_ge_k": float(np.mean(lengths >= 1)) if len(lengths) else float("nan"),
        "median_error_streak": float(np.median(lengths)) if len(lengths) else float("nan"),
        "mean_error_streak": float(np.mean(lengths)) if len(lengths) else float("nan"),
        "p95_error_streak": float(np.quantile(lengths, 0.95)) if len(lengths) else float("nan"),
    })
    pd.DataFrame(survival_rows).to_csv(args.results / "A4_error_survival.csv", index=False)

    # Prediction-centric contamination runs.
    cont_rows = []
    for (scene, pid), part in df.groupby(["scene", "pred_track_id"]):
        part = part.sort_values(["frame_id", "view_id"])
        counts = part.gt_instance_id.value_counts()
        canonical = int(sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0])
        bad = part.gt_instance_id != canonical
        run = []
        for row, is_bad in zip(part.itertuples(), bad.tolist()):
            if is_bad:
                run.append(row)
            elif run:
                cont_rows.append({"scene": scene, "pred_track_id": pid, "canonical_gt": canonical, "start_frame": run[0].frame_id, "end_frame": run[-1].frame_id, "duration_observations": len(run), "views": len({x.view_id for x in run})}); run=[]
        if run:
            cont_rows.append({"scene": scene, "pred_track_id": pid, "canonical_gt": canonical, "start_frame": run[0].frame_id, "end_frame": run[-1].frame_id, "duration_observations": len(run), "views": len({x.view_id for x in run})})
    cont = pd.DataFrame(cont_rows)
    cont.to_csv(args.results / "A4_contamination_events.csv", index=False)

    # Cross-view conditional survival.  For every GT/frame/view error, compare
    # another-view error at t+k with the unconditional other-view error rate.
    pair_rows = []
    for (scene, gid), part in df.groupby(["scene", "gt_instance_id"]):
        by = {(int(r.frame_id), int(r.view_id)): bool(r.gt_wrong_identity) for r in part.itertuples()}
        views = sorted(part.view_id.unique())
        for k in [1, 2, 5, 10, 20]:
            cond_num = cond_den = base_num = base_den = 0
            for (frame, view), err in by.items():
                for other in views:
                    if other == view:
                        continue
                    target = (frame + k, int(other))
                    if target not in by:
                        continue
                    cond_den += int(err)
                    cond_num += int(err and by[target])
                    base_den += 1
                    base_num += int(by[target])
            pair_rows.append({"scene": scene, "gt_instance_id": gid, "lag_frames": k, "conditional_num": cond_num, "conditional_den": cond_den, "conditional_error_rate": cond_num / cond_den if cond_den else np.nan, "baseline_num": base_num, "baseline_den": base_den, "baseline_error_rate": base_num / base_den if base_den else np.nan})
    cascade = pd.DataFrame(pair_rows)
    cascade.to_csv(args.results / "A4_cross_view_cascade.csv", index=False)
    # Pool numerator/denominator rather than averaging scene/entity rates.  The
    # conditional estimate is P(error at t+k | error at t), while the baseline
    # is P(error at t+k) over all eligible cross-view targets.
    sums = cascade.groupby("lag_frames")[["conditional_num", "conditional_den",
                                           "baseline_num", "baseline_den"]].sum().reset_index()
    sums["conditional_error_rate"] = sums["conditional_num"] / sums["conditional_den"].replace(0, np.nan)
    sums["baseline_error_rate"] = sums["baseline_num"] / sums["baseline_den"].replace(0, np.nan)
    aggregate = sums[["lag_frames", "conditional_error_rate", "baseline_error_rate"]]
    ax = aggregate.plot(x="lag_frames", y=["conditional_error_rate", "baseline_error_rate"], marker="o", figsize=(6, 4))
    ax.set_ylabel("error probability")
    ax.set_title("A4 cross-view error survival")
    ax.legend(["P(error at t+k | error at t)", "arbitrary target error"])
    fig = ax.get_figure(); fig.tight_layout(); fig.savefig(args.figures / "A4_error_survival_curve.png", dpi=140); plt.close(fig)

    manifest = {
        "table": str(args.table),
        "matching_is_diagnostic": True,
        "gt_dominant_definition": "majority predicted track per (scene, gt_instance_id), ties by lower track ID",
        "streak_definition": "consecutive wrong matched observations in frame/view order",
        "cross_view_definition": "other-view exact frame+k conditional versus arbitrary target baseline",
        "survival_events": len(lengths),
        "contamination_events": len(cont_rows),
        "results": [str(args.results / x) for x in ["A4_error_survival.csv", "A4_contamination_events.csv", "A4_cross_view_cascade.csv"]],
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
