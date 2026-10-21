"""Verify the fresh all-24 causal score-matrix audit before supervision."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "manifests/omega_full24_training_protocol_v1.json"
DEFAULT_ROOT = Path("/data1/liuyeqiang/WWW_omega_fulltrain_runtime_20261016/full_train_logit_audit_v1")


def _sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(PROTOCOL.read_text())
    rows = []
    total_score_rows = 0
    total_candidates = 0
    for scene in protocol["dataset"]["video_ids"]:
        folder = args.root / f"scene{scene}_full_logit_v1"
        manifest = json.loads((folder / "manifest.json").read_text()) if (folder / "manifest.json").exists() else {}
        if manifest.get("status") != "COMPLETE_FULL_CAUSAL_AUDIT_WITH_NATIVE_SCORE_MATRICES":
            raise ValueError(f"scene {scene} is incomplete: {manifest.get('status')}")
        score_rows = 0
        candidate_cells = 0
        with gzip.open(folder / "rankings.jsonl.gz", "rt") as stream:
            for line in stream:
                ranking = json.loads(line)
                if "track_ids" not in ranking or "track_scores" not in ranking:
                    raise ValueError(f"scene {scene} has a rank-only row")
                if len(ranking["track_scores"]) != len(ranking["queries"]):
                    raise ValueError(f"scene {scene} query/score row mismatch")
                for values in ranking["track_scores"]:
                    if len(values) != len(ranking["track_ids"]):
                        raise ValueError(f"scene {scene} track score width mismatch")
                    candidate_cells += len(values)
                score_rows += len(ranking["queries"])
        if score_rows != manifest.get("persistent_dense_candidate_score_matrices"):
            raise ValueError(f"scene {scene} score-row count differs from manifest")
        total_score_rows += score_rows
        total_candidates += candidate_cells
        rows.append({"scene": scene, "role": "fit" if scene in protocol["scene_split"]["fit_video_ids"] else "validation", "score_rows": score_rows, "candidate_cells": candidate_cells, "manifest_sha256": _sha256(folder / "manifest.json")})
    report = {
        "status": "PASS_COMPLETE_ALL24_NATIVE_SCORE_MATRIX_COVERAGE",
        "protocol": str(PROTOCOL),
        "protocol_sha256": _sha256(PROTOCOL),
        "audit_root": str(args.root),
        "scene_count": len(rows),
        "fit_scene_ids": protocol["scene_split"]["fit_video_ids"],
        "validation_scene_ids": protocol["scene_split"]["validation_video_ids"],
        "score_rows": total_score_rows,
        "candidate_cells": total_candidates,
        "score_semantics": "actual activated GMT association probabilities summed by live track ID before the original Hungarian/threshold solver; ranking-only proxies are not used",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "score_rows": total_score_rows, "candidate_cells": total_candidates}), flush=True)


if __name__ == "__main__":
    main()
