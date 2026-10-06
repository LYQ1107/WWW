#!/usr/bin/env bash

# Run the required three-repeat runtime feature-parity stability gate after
# the existing corrected video6 artifact completes. This waiter never owns or
# restarts the video6 builder.

set -u

REPO=/data1/liuyeqiang/WWW_rng_fix_v4
RUNTIME=/home/liuyeqiang/WWW_jev_rng_v4_runtime
V6=$RUNTIME/small_h8_rng_controlled_v4_current_head_75b0aea/video_06
MANIFEST=$V6/manifest.json
OUT=$REPO/reports/JEV_RNG_V4/RUNTIME_FEATURE_PARITY_STABILITY_VIDEO06_CORRECTED_CURRENT_HEAD.json
STABILITY_ROOT=$RUNTIME/small_h8_stability_v10_current_head
LOG=$RUNTIME/current_head_v6_stability_waiter.log

exec > >(tee -a "$LOG") 2>&1
echo "watcher_started=$(date -Is)"
while ! jq -e '.status == "COMPLETE" and (.records | tonumber) == 5164' \
    "$MANIFEST" >/dev/null 2>&1; do
    if jq -e '.status == "FAILED"' "$MANIFEST" >/dev/null 2>&1; then
        echo "video6_manifest_failed=$(date -Is)"
        exit 2
    fi
    sleep 30
done

TRACE=$(jq -r '.trace_partition // .source_trace // empty' "$MANIFEST")
echo "video6_manifest_ready=$(date -Is) trace=$TRACE"
if [ -z "$TRACE" ] || [ ! -f "$TRACE" ]; then
    echo "missing_manifest_bound_trace=$TRACE"
    exit 3
fi

export CUDA_VISIBLE_DEVICES=4
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO:$REPO/reproduction_tools:$REPO/third_party/CenterNet2"
set +e
/home/liuyeqiang/anaconda3/envs/GMT/bin/python -u \
    "$REPO/reproduction_tools/run_jev_feature_parity_stability.py" \
    --video-id 6 \
    --trace "$TRACE" \
    --records "$V6/records.jsonl" \
    --output "$OUT" \
    --runtime-root "$STABILITY_ROOT" \
    --device cuda:0 \
    --max-frame 600 \
    --repetitions 3 \
    --tolerance 2e-5
RC=$?
set -e

cd "$REPO"
while [ -e .git/index.lock ]; do sleep 5; done
git add "$OUT" reproduction_tools/run_current_head_v6_stability_waiter.sh reproduction_tools/run_corrected_v4_tracking_waiter.sh
git commit -m "Run current-head video6 stability gate" || true
git push origin HEAD || {
    git pull --rebase origin jev/counterfactual-rng-isolation-v4-20261006
    git push origin HEAD
}
echo "video6_stability_exit=$RC"
exit "$RC"
