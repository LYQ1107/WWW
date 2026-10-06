#!/usr/bin/env bash

# Validate and replay the separate current-head video01 rebuild. The original
# GPU8 video01 builder is intentionally outside this waiter and is never
# stopped, migrated, or overwritten.

set -u

REPO=/data1/liuyeqiang/WWW_rng_fix_v4
RUNTIME=/home/liuyeqiang/WWW_jev_rng_v4_runtime
PYTHON=/home/liuyeqiang/anaconda3/envs/GMT/bin/python
ROOT=$RUNTIME/formal_current_head_corrected_full
TRACE=$RUNTIME/formal_current_head_off_trace/video01/trace_video_01.jsonl
RECORDS=$ROOT/video01_records.jsonl
MANIFEST=$RECORDS.manifest.json
REPORT_ROOT=$REPO/reports/JEV_RNG_V4
PROVENANCE=$REPORT_ROOT/CURRENT_HEAD_VIDEO01_PROVENANCE.json
PARITY=$REPORT_ROOT/RUNTIME_FEATURE_PARITY_VIDEO01_CURRENT_HEAD.json
LOG=$RUNTIME/formal_current_head_video01_aftercare.log

export PYTHONPATH="$REPO:$REPO/reproduction_tools:$REPO/third_party/CenterNet2"
exec > >(tee -a "$LOG") 2>&1
echo "video01_aftercare_started=$(date -Is)"

while ! jq -e '.status == "PASS" and (.records | tonumber) == 8995' \
    "$MANIFEST" >/dev/null 2>&1; do
    sleep 30
done
echo "video01_current_head_manifest_ready=$(date -Is)"

set +e
"$PYTHON" -u "$REPO/reproduction_tools/validate_jev_video_artifact.py" \
    --manifest "$MANIFEST" --records "$RECORDS" \
    --output "$PROVENANCE" --expected-records 8995 --horizon 8
VALIDATION_RC=$?
set -e
if [ "$VALIDATION_RC" -ne 0 ]; then
    echo "video01_current_head_validation_failed=$VALIDATION_RC"
    cd "$REPO"
    while [ -e .git/index.lock ]; do sleep 5; done
    git add "$PROVENANCE"
    git commit -m "Record current-head video01 provenance failure" || true
    git push origin HEAD || {
        git pull --rebase origin jev/counterfactual-rng-isolation-v4-20261006
        git push origin HEAD
    }
    exit 2
fi

set +e
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    JEV_CACHE_PATH=/data1/liuyeqiang/WWW/outputs/research_final_v2/off/perception_cache_train \
    "$PYTHON" -u "$REPO/reproduction_tools/run_jev_feature_parity_stability.py" \
    --video-id 1 --trace "$TRACE" --records "$RECORDS" \
    --output "$PARITY" --runtime-root "$RUNTIME/formal_current_head_video01_parity" \
    --device cuda:0 --max-frame 1000000 --repetitions 3 --tolerance 2e-5
PARITY_RC=$?
set -e

cd "$REPO"
while [ -e .git/index.lock ]; do sleep 5; done
git add "$PROVENANCE" "$PARITY"
git commit -m "Record current-head video01 runtime parity" || true
git push origin HEAD || {
    git pull --rebase origin jev/counterfactual-rng-isolation-v4-20261006
    git push origin HEAD
}
echo "video01_aftercare_complete=$(date -Is) validation_rc=$VALIDATION_RC parity_rc=$PARITY_RC"
exit "$PARITY_RC"
