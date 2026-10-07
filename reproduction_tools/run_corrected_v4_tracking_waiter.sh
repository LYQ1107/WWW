#!/usr/bin/env bash

# Wait for the corrected-v4 hard gates, resolve the manifest-bound partition
# trace, then run the fail-closed small-video tracking comparison.
# This waiter never owns or restarts the video1 record builder.

set -u

RUNTIME=/home/liuyeqiang/WWW_jev_rng_v4_runtime
REPO=/data1/liuyeqiang/WWW_rng_fix_v4
V1="$RUNTIME/formal_current_head_corrected_full"
RECORDS="$V1/video01_records.jsonl"
MANIFEST="$RECORDS.manifest.json"
# The old small-gate checkpoints and records are screening-only. The
# corrected-v4 tracking run consumes the completed canonical corrected-feature
# controllers unless an explicit compatible methods root is supplied.
METHODS="${METHODS_ROOT:-$RUNTIME/small_h8_training_v4_canonical_features/methods_v1}"
TRACE="$RUNTIME/formal_current_head_off_trace/video01/trace_video_01.jsonl"
PROV="$REPO/reports/JEV_RNG_V4/CURRENT_HEAD_VIDEO01_PROVENANCE.json"
CAND="$REPO/reports/JEV_RNG_V4/REACTIVATION_CANDIDATE_PARITY_CURRENT_HEAD_VIDEO01.json"
PARITY="$REPO/reports/JEV_RNG_V4/RUNTIME_FEATURE_PARITY_VIDEO01_CURRENT_HEAD.json"
STABILITY="$REPO/reports/JEV_RNG_V4/RUNTIME_FEATURE_PARITY_STABILITY_VIDEO01_CURRENT_HEAD.json"
FORMAL="$REPO/reports/JEV_RNG_V4/FORMAL_GMT_INTRA_VIDEO_CHUNK_EQUIVALENCE_VIDEO07_CURRENT_HEAD_V10.json"
TRAIN="${TRAINING_REPORT:-$REPO/reports/JEV_RNG_V4/CORRECTED_V4_SMALL_H8_THREE_WAY_VIDEO06_VIDEO07.json}"
OUTROOT="${TRACKING_OUTPUT_ROOT:-$RUNTIME/current_head_video01_tracking}"
OUTREPORT="$REPO/reports/JEV_RNG_V4/CURRENT_HEAD_VIDEO01_THREE_WAY_TRACKING.json"

echo "watcher_started=$(date -Is)"
while ! jq -e '.status == "PASS" and (.records | tonumber) == 8995' "$MANIFEST" >/dev/null 2>&1; do
    sleep 30
done

MANIFEST_MTIME=$(stat -c %Y "$MANIFEST")
echo "manifest_ready=$(date -Is) trace=$TRACE"
if [ -z "$TRACE" ] || [ ! -f "$TRACE" ]; then
    echo "missing_manifest_bound_trace=$TRACE"
    exit 3
fi

all_hard_gates_pass() {
    for report in "$PROV" "$CAND" "$PARITY" "$STABILITY" "$FORMAL" "$TRAIN"; do
        if ! jq -e '.status == "PASS"' "$report" >/dev/null 2>&1; then
            return 1
        fi
    done
    return 0
}
while ! all_hard_gates_pass; do
    sleep 30
done
while [ ! -f "$CAND" ] || [ "$(stat -c %Y "$CAND")" -le "$MANIFEST_MTIME" ]; do
    sleep 30
done

echo "all_gates_present=$(date -Is)"
set +e
PYTHONPATH="$REPO:$REPO/reproduction_tools:$REPO/third_party/CenterNet2" \
    /home/liuyeqiang/anaconda3/envs/GMT/bin/python -u \
    "$REPO/reproduction_tools/run_corrected_v4_tracking.py" \
    --video-id 1 \
    --trace "$TRACE" \
    --records "$RECORDS" \
    --methods-root "$METHODS" \
    --output-root "$OUTROOT" \
    --output-report "$OUTREPORT" \
    --provenance-report "$PROV" \
    --candidate-report "$CAND" \
    --feature-parity-report "$PARITY" \
    --stability-report "$STABILITY" \
    --formal-report "$FORMAL" \
    --training-report "$TRAIN" \
    --device cuda:0 \
    --tolerance 2e-5
RC=$?
set -e

cd "$REPO"
while [ -e .git/index.lock ]; do
    sleep 5
done
git add "$OUTREPORT"
git commit -m "Record gated corrected v4 tracking result" || true
git push origin HEAD || {
    git pull --rebase origin jev/counterfactual-rng-isolation-v4-20261006
    git push origin HEAD
}
echo "tracking_exit=$RC"
exit "$RC"
