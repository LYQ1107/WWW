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
NATIVE_TRACE=$RUNTIME/native_video1_corrected_v3/native_off_trace_video01.jsonl
RECORDS=$ROOT/video01_records.jsonl
MANIFEST=$RECORDS.manifest.json
REPORT_ROOT=$REPO/reports/JEV_RNG_V4
PROVENANCE=$REPORT_ROOT/CURRENT_HEAD_VIDEO01_PROVENANCE.json
CANDIDATE=$REPORT_ROOT/REACTIVATION_CANDIDATE_PARITY_CURRENT_HEAD_VIDEO01.json
PARITY=$REPORT_ROOT/RUNTIME_FEATURE_PARITY_VIDEO01_CURRENT_HEAD.json
STABILITY=$REPORT_ROOT/RUNTIME_FEATURE_PARITY_STABILITY_VIDEO01_CURRENT_HEAD.json
HARD_GATES=$REPORT_ROOT/VIDEO01_CORRECTED_HARD_GATES.json
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

# First run the complete feature replay gate. This intentionally covers all
# question types (MATCH, MEMORY, and REACTIVATION) and writes the explicit
# per-question TOTAL parity table; candidate parity is a separate
# native-vs-mutable semantic gate below.
set +e
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    JEV_CACHE_PATH=/data1/liuyeqiang/WWW/outputs/research_final_v2/off/perception_cache_train \
    "$PYTHON" -u "$REPO/reproduction_tools/run_jev_runtime_feature_parity.py" \
    --video-id 1 --trace "$TRACE" --records "$RECORDS" \
    --output "$PARITY" --device cuda:0 --tolerance 2e-5
PARITY_RC=$?
set -e

# The v3 native trace was produced by the actual current-head GMT runtime,
# contains exactly video01, and carries native_candidate_* fields. Compare it
# only after the full three-question feature gate; the legacy candidate report
# is not reused.
if [ ! -f "$NATIVE_TRACE" ]; then
    echo "missing_current_head_native_trace=$NATIVE_TRACE"
    exit 3
fi
set +e
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    JEV_CACHE_PATH=/data1/liuyeqiang/WWW/outputs/research_final_v2/off/perception_cache_train \
    "$PYTHON" -u "$REPO/reproduction_tools/compare_reactivation_candidates.py" \
    --video-id 1 --native-trace "$NATIVE_TRACE" --records "$RECORDS" \
    --output "$CANDIDATE" \
    --replay-root "$RUNTIME/reactivation_candidate_replay_current_head_video01" \
    --device cuda:0 --max-frame 1000000 --tolerance 2e-5
CANDIDATE_RC=$?
set -e

# Repeat the same corrected OFF replay three times after the explicit parity
# table has passed. This freezes the numerical tolerance independently from
# the semantic candidate gate.
set +e
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    JEV_CACHE_PATH=/data1/liuyeqiang/WWW/outputs/research_final_v2/off/perception_cache_train \
    "$PYTHON" -u "$REPO/reproduction_tools/run_jev_feature_parity_stability.py" \
    --video-id 1 --trace "$TRACE" --records "$RECORDS" \
    --output "$STABILITY" --runtime-root "$RUNTIME/formal_current_head_video01_parity_stability" \
    --device cuda:0 --max-frame 1000000 --repetitions 3 --tolerance 2e-5
STABILITY_RC=$?
set -e

set +e
"$PYTHON" -u "$REPO/reproduction_tools/update_video01_hard_gates.py" \
    --manifest "$MANIFEST" --provenance "$PROVENANCE" --parity "$PARITY" \
    --candidate "$CANDIDATE" --stability "$STABILITY" \
    --output "$HARD_GATES"
HARD_GATES_RC=$?
set -e

cd "$REPO"
while [ -e .git/index.lock ]; do sleep 5; done
git add "$PROVENANCE" "$PARITY" "$STABILITY"
git add "$CANDIDATE"
git add "$HARD_GATES"
git commit -m "Record current-head video01 runtime parity" || true
git push origin HEAD || {
    git pull --rebase origin jev/counterfactual-rng-isolation-v4-20261006
    git push origin HEAD
}
echo "video01_aftercare_complete=$(date -Is) validation_rc=$VALIDATION_RC parity_rc=$PARITY_RC candidate_rc=$CANDIDATE_RC stability_rc=$STABILITY_RC hard_gates_rc=$HARD_GATES_RC"
if [ "$CANDIDATE_RC" -ne 0 ] || [ "$PARITY_RC" -ne 0 ] || [ "$STABILITY_RC" -ne 0 ]; then
    exit 4
fi
exit 0
