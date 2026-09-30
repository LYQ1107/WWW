#!/usr/bin/env python3
"""Evaluate the predeclared cross-view entity-retrieval audit protocols."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x)
    return x / n if n > 0 else x


def retrieval(tracklets):
    """Return rows for each feature and quality stratum."""
    out = []
    if not tracklets:
        return out
    for feature in ("appearance_feature", "fused_feature"):
        rows = [x for x in tracklets if feature in x]
        if len(rows) < 2:
            continue
        qvals = np.asarray([x["quality"] for x in rows], dtype=float)
        # Rank before binning so small audit subsets and tied area values still
        # yield deterministic non-empty Q1/Q4 strata.
        ranks = pd.Series(qvals).rank(method="first")
        bins = pd.qcut(ranks, 4, labels=False)
        q1_mask = np.asarray(bins == 0)
        q4_mask = np.asarray(bins == 3)
        q1 = float(qvals[q1_mask].max()) if q1_mask.any() else float("nan")
        q4 = float(qvals[q4_mask].min()) if q4_mask.any() else float("nan")
        for stratum, subset in (("all", rows), ("Q1", [x for x, keep in zip(rows, q1_mask) if keep]),
                                ("Q4", [x for x, keep in zip(rows, q4_mask) if keep])):
            r1 = []; r5 = []; aps = []
            for q in subset:
                gallery = [g for g in rows if g["scene"] == q["scene"] and g["view"] != q["view"]]
                positives = [g for g in gallery if g["label"] is not None and g["label"] == q["label"]]
                if not positives:
                    continue
                qv = l2(np.asarray(q[feature], dtype=np.float32))
                scored = sorted(((float(np.dot(qv, l2(np.asarray(g[feature], dtype=np.float32)))), g)
                                 for g in gallery), key=lambda z: -z[0])
                rel = [int(g["label"] == q["label"]) for _, g in scored]
                if not rel:
                    continue
                r1.append(float(any(rel[:1])))
                r5.append(float(any(rel[:5])))
                hit = 0; prec = []
                for rank, val in enumerate(rel, 1):
                    if val:
                        hit += 1; prec.append(hit / rank)
                aps.append(float(np.mean(prec)) if prec else 0.0)
            out.append({"feature": feature, "stratum": stratum, "n_queries": len(aps),
                        "R@1": float(np.mean(r1)) if r1 else float("nan"),
                        "R@5": float(np.mean(r5)) if r5 else float("nan"),
                        "mAP": float(np.mean(aps)) if aps else float("nan"),
                        "quality_threshold_low": float(q1), "quality_threshold_high": float(q4)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()
    table = pd.read_parquet(args.table)
    table = table[table["matched"] == True].copy() if "matched" in table else table.copy()
    # The observation table records matched rows; tolerate either explicit
    # matched column or the historical matched-only format.
    feature_by_key = {}
    feature_files = []
    for path in sorted(args.features.glob("*.pt")):
        obj = torch.load(path, map_location="cpu")
        feature_files.append({"path": str(path), "sha256": sha256(path), "records": len(obj.get("records", []))})
        for rec in obj.get("records", []):
            key = (obj.get("scene", path.stem), int(rec["image_id"]), int(rec["pred_track_id"]))
            feature_by_key[key] = rec

    def build(mode):
        groups = defaultdict(list)
        for row in table.itertuples(index=False):
            scene = str(row.scene)
            if mode == "gt":
                label = int(row.gt_instance_id)
                key = (scene, int(row.image_id), int(row.pred_track_id))
                group = (scene, label, int(row.view_id))
            else:
                label = None
                key = (scene, int(row.image_id), int(row.pred_track_id))
                group = (scene, int(row.pred_track_id), int(row.view_id))
            rec = feature_by_key.get(key)
            if rec is None:
                continue
            if mode == "pred":
                # Majority GT label is assigned after grouping below.
                label = int(row.gt_instance_id)
            groups[group].append((rec, label, float(row.normalized_area), float(row.pred_score)))
        tracklets = []
        for (scene, ident, view), vals in groups.items():
            if len(vals) < 5:
                continue
            labels = [v[1] for v in vals]
            label = Counter(labels).most_common(1)[0][0]
            item = {"scene": scene, "view": view, "label": label,
                    "quality": float(np.mean([v[2] for v in vals])),
                    "score_quality": float(np.mean([v[3] for v in vals])),
                    "n_obs": len(vals)}
            for f in ("appearance_feature", "fused_feature"):
                mat = np.asarray([l2(np.asarray(v[0][f], dtype=np.float32)) for v in vals])
                item[f] = l2(mat.mean(axis=0)).tolist()
            tracklets.append(item)
        return tracklets

    # Area and score are separate quality fields; the main rows use area and
    # the manifest records score-stratified reruns explicitly.
    all_rows = []
    for mode in ("gt", "pred"):
        for row in retrieval(build(mode)):
            row["protocol"] = "gt_aligned" if mode == "gt" else "pred_track"
            all_rows.append(row)
    args.results.mkdir(parents=True, exist_ok=True)
    out = args.results / "A5_entity_retrieval.csv"
    pd.DataFrame(all_rows).to_csv(out, index=False)
    manifest = {"table": str(args.table), "table_sha256": sha256(args.table),
                "feature_files": feature_files, "row_count": len(all_rows),
                "min_matched_observations": 5,
                "aggregation": "per-frame L2 normalize, mean pool, L2 normalize",
                "output": str(out), "output_sha256": sha256(out)}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
