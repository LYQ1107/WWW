#!/usr/bin/env bash

# Validate the completed corrected video1 artifact and compare all three
# question types against the live mutated-state OFF replay. This waiter never
# owns or restarts the video1 builder.

set -u

REPO=/data1/liuyeqiang/WWW_rng_fix_v4
RUNTIME=/home/liuyeqiang/WWW_jev_rng_v4_runtime
V1=$RUNTIME/small_h8_rng_controlled_v4_current/video_01
MANIFEST=$V1/manifest.json
RECORDS=$V1/records.jsonl
CANDIDATE=$REPO/reports/JEV_RNG_V4/REACTIVATION_CANDIDATE_PARITY.json
PROVENANCE=$REPO/reports/JEV_RNG_V4/VIDEO01_CORRECTED_CURRENT_HEAD_PROVENANCE.json
PARITY=$REPO/reports/JEV_RNG_V4/RUNTIME_FEATURE_PARITY_VIDEO01_CORRECTED_CURRENT_HEAD.json
LOG=$RUNTIME/video01_full_parity_corrected_v4_waiter.log

exec > >(tee -a "$LOG") 2>&1
echo "watcher_started=$(date -Is)"

while ! jq -e '.status == "COMPLETE" and (.records | tonumber) == 8996' \
    "$MANIFEST" >/dev/null 2>&1; do
    sleep 30
done
MANIFEST_MTIME=$(stat -c %Y "$MANIFEST")
TRACE=$(jq -r '.trace_partition // empty' "$MANIFEST")
echo "manifest_ready=$(date -Is) trace=$TRACE"
if [ -z "$TRACE" ] || [ ! -f "$TRACE" ]; then
    echo "manifest_bound_trace_missing=$TRACE"
    exit 3
fi

set +e
PYTHONPATH="$REPO:$REPO/reproduction_tools:$REPO/third_party/CenterNet2" \
    /home/liuyeqiang/anaconda3/envs/GMT/bin/python -u \
    "$REPO/reproduction_tools/validate_jev_video_artifact.py" \
    --manifest "$MANIFEST" --records "$RECORDS" \
    --output "$PROVENANCE" --expected-records 8996 --horizon 8
VALIDATION_RC=$?
set -e

cd "$REPO"
while [ -e .git/index.lock ]; do sleep 5; done
git add "$PROVENANCE"
git commit -m "Validate corrected current-head video1 artifact" || true
git push origin HEAD || {
    git pull --rebase origin jev/counterfactual-rng-isolation-v4-20261006
    git push origin HEAD
}
if [ "$VALIDATION_RC" -ne 0 ]; then
    echo "validation_failed=$VALIDATION_RC"
    exit "$VALIDATION_RC"
fi

# Candidate parity and this full replay both use GPU0. Wait for the new
# candidate report, not merely the legacy file left by the old small gate.
while [ ! -f "$CANDIDATE" ] || [ "$(stat -c %Y "$CANDIDATE")" -le "$MANIFEST_MTIME" ]; do
    sleep 30
done
echo "candidate_ready=$(date -Is)"

export JEV_CACHE_PATH=/data1/liuyeqiang/WWW/outputs/research_final_v2/off/perception_cache_train
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO:$REPO/reproduction_tools:$REPO/third_party/CenterNet2"
set +e
/home/liuyeqiang/anaconda3/envs/GMT/bin/python -u \
    "$REPO/reproduction_tools/run_jev_runtime_feature_parity.py" \
    --video-id 1 --trace "$TRACE" --records "$RECORDS" \
    --output "$PARITY" --device cuda:0 --tolerance 2e-5
PARITY_RC=$?
set -e

cd "$REPO"
while [ -e .git/index.lock ]; do sleep 5; done
git add "$PARITY"
git commit -m "Record corrected current-head video1 runtime parity" || true
git push origin HEAD || {
    git pull --rebase origin jev/counterfactual-rng-isolation-v4-20261006
    git push origin HEAD
}
echo "parity_waiter_exit=$PARITY_RC"
exit "$PARITY_RC"
