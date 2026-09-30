#!/usr/bin/env python3
"""A1 quality stratification for the baseline observation table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SEED = 20260930


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n=10000):
    if len(values) == 0:
        return float("nan"), float("nan")
    samples = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--figures", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(args.table)

    # Canonical predicted identity and purity are descriptive labels only.
    counts = df.groupby(["scene", "pred_track_id", "gt_instance_id"]).size().rename("n").reset_index()
    canon = counts.sort_values(
        ["scene", "pred_track_id", "n", "gt_instance_id"],
        ascending=[True, True, False, True],
    ).drop_duplicates(["scene", "pred_track_id"])
    canon = canon.rename(columns={"gt_instance_id": "canonical_gt", "n": "canonical_n"})
    totals = counts.groupby(["scene", "pred_track_id"])["n"].sum().rename("track_matched_n")
    purity = canon.set_index(["scene", "pred_track_id"])["canonical_n"] / totals
    df = df.merge(canon[["scene", "pred_track_id", "canonical_gt"]], on=["scene", "pred_track_id"], how="left")
    df["dominant_identity_correct"] = (df["gt_instance_id"] == df["canonical_gt"]).astype(float)
    df["track_purity"] = [float(purity[(r.scene, r.pred_track_id)]) for r in df.itertuples()]
    fragments = df.groupby(["scene", "gt_instance_id"])["pred_track_id"].nunique().rename("gt_fragment_count")
    df = df.join(fragments, on=["scene", "gt_instance_id"])
    # Cross-view consistency is assigned only to GT/frame groups with >=2
    # matched views; unmatched single-view groups are excluded from its mean.
    cv = df.groupby(["scene", "frame_id", "gt_instance_id"])
    cv_values = cv["pred_track_id"].agg(lambda x: float(x.nunique() == 1) if len(x) >= 2 else np.nan)
    df = df.join(cv_values.rename("cross_view_consistent"), on=["scene", "frame_id", "gt_instance_id"])

    quality_defs = {
        "normalized_area": "area",
        "pred_score": "score",
        "crop_blur": "blur",
    }
    summaries = []
    effects = []
    rng = np.random.default_rng(SEED)
    for col, label in quality_defs.items():
        valid = df[col].notna() & np.isfinite(df[col].to_numpy(dtype=float))
        work = df.loc[valid].copy()
        # Rank first makes quartiles deterministic even with many ties.
        work["quality_bin"] = pd.qcut(work[col].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
        metrics = [
            "dominant_identity_correct", "track_purity", "gt_fragment_count",
            "cross_view_consistent", "iou", "pred_score", "normalized_area",
        ]
        for q, part in work.groupby("quality_bin", observed=False):
            row = {"quality_dimension": label, "quality_variable": col, "quality_bin": str(q), "matched_observations": len(part)}
            for metric in metrics:
                row["mean_" + metric] = float(part[metric].mean())
            summaries.append(row)
        by_scene = []
        all_scenes = sorted(df["scene"].unique())
        grouped_scene = {scene: part for scene, part in work.groupby("scene")}
        for scene in all_scenes:
            part = grouped_scene.get(scene, work.iloc[0:0])
            q1 = part.loc[part.quality_bin == "Q1", "dominant_identity_correct"].mean()
            q4 = part.loc[part.quality_bin == "Q4", "dominant_identity_correct"].mean()
            gap = q1 - q4 if np.isfinite(q1) and np.isfinite(q4) else np.nan
            by_scene.append({"quality_dimension": label, "quality_variable": col, "scene": scene,
                             "Q1_identity_consistency": q1, "Q4_identity_consistency": q4,
                             "Q1_minus_Q4": gap, "n_scene": len(part)})
        effects.extend(by_scene)
        gaps = np.array([x["Q1_minus_Q4"] for x in by_scene if np.isfinite(x["Q1_minus_Q4"])], dtype=float)
        lo, hi = bootstrap_ci(gaps, rng)
        summaries.append({
            "quality_dimension": label, "quality_variable": col, "quality_bin": "Q1_minus_Q4",
            "matched_observations": len(work), "identity_gap": float(np.mean(gaps)) if len(gaps) else np.nan,
            "scene_median_gap": float(np.median(gaps)) if len(gaps) else np.nan,
            "bootstrap95_low": lo, "bootstrap95_high": hi,
        })
        # Diagnostic figure: all bins, identity consistency and purity.
        agg = work.groupby("quality_bin", observed=False)[["dominant_identity_correct", "track_purity"]].mean().reindex(["Q1", "Q2", "Q3", "Q4"])
        ax = agg.plot(kind="bar", ylim=(0, 1), figsize=(6, 4), color=["#3568a8", "#d77b28"])
        ax.set_xlabel("quality bin (global quartiles)")
        ax.set_ylabel("rate / purity")
        ax.set_title(f"{label}: quality vs identity consistency")
        ax.legend(["dominant identity correct", "track purity"], loc="best")
        fig = ax.get_figure(); fig.tight_layout(); fig.savefig(args.figures / f"A1_{label}_vs_identity.png", dpi=140); plt.close(fig)

    strat = pd.DataFrame(summaries)
    strat.to_csv(args.results / "A1_quality_stratification.csv", index=False)
    pd.DataFrame(effects).to_csv(args.results / "A1_scene_effects.csv", index=False)
    manifest = {
        "table": str(args.table), "seed": SEED, "quality_variables": list(quality_defs),
        "quartiles": "global rank-based Q1-Q4", "bootstrap_resamples": 10000,
        "matching_is_diagnostic": True, "rows": len(df),
        "results": [str(args.results / "A1_quality_stratification.csv"), str(args.results / "A1_scene_effects.csv")],
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(strat.to_string(index=False))
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
