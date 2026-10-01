#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN="${1:?run name}"
PY="${GMT_PY:-/data3/liuyeqiang/GMT_VisionTrack_repro/tools/miniconda3/envs/GMT/bin/python}"
GT="$ROOT/../GMT_challenge_audit/audit/cache/sanity_subset_test.json"
OUT="$ROOT/causal/runs/$RUN"
"$PY" "$ROOT/audit_tools/evaluate_audit_run.py" \
  --pred-json "$OUT/predictions.json" \
  --annotations "$GT" \
  --gt-root "$ROOT/../GMT_paper_repro_clean/TrackEval/data/gt/mot_challenge/vision-train" \
  --output "$OUT/evaluation" >"$OUT/evaluation.log" 2>&1
