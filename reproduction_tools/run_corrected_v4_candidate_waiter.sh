#!/usr/bin/env bash

# Wait for the bounded native video1 OFF run and corrected builder, then run
# the reactivation candidate parity gate. This never owns or restarts video1.

set -u

REPO=/data1/liuyeqiang/WWW_rng_fix_v4
RUNTIME=/home/liuyeqiang/WWW_jev_rng_v4_runtime
NATIVE=${NATIVE_ROOT:-$RUNTIME/native_video1_corrected_v1}
NATIVE_TRACE=$NATIVE/native_off_trace_video01.jsonl
NATIVE_PID=${NATIVE_PID:-24960}
V1=$RUNTIME/small_h8_rng_controlled_v4_current/video_01
MANIFEST=$V1/manifest.json
REPORT=$REPO/reports/JEV_RNG_V4/REACTIVATION_CANDIDATE_PARITY.json
REPLAY=${REPLAY_ROOT:-$RUNTIME/reactivation_candidate_replay_corrected_v1}
LOG=${CANDIDATE_WAITER_LOG:-$RUNTIME/corrected_video1_candidate_parity_waiter_v2.log}

mkdir -p "$REPLAY"
exec > >(tee -a "$LOG") 2>&1
echo "watcher_started=$(date -Is) native_pid=$NATIVE_PID"

while kill -0 "$NATIVE_PID" 2>/dev/null; do
    sleep 30
done
if [ ! -s "$NATIVE_TRACE" ]; then
    echo "native_trace_missing=$NATIVE_TRACE"
    exit 3
fi
echo "native_done=$(date -Is) native_lines=$(wc -l < "$NATIVE_TRACE")"

while ! jq -e '.status == "COMPLETE" and (.records | tonumber) == 8996' \
    "$MANIFEST" >/dev/null 2>&1; do
    sleep 30
done
echo "video1_manifest_ready=$(date -Is)"

export JEV_CACHE_PATH=/data1/liuyeqiang/WWW/outputs/research_final_v2/off/perception_cache_train
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO:$REPO/reproduction_tools:$REPO/third_party/CenterNet2"
set +e
/home/liuyeqiang/anaconda3/envs/GMT/bin/python -u \
    "$REPO/reproduction_tools/compare_reactivation_candidates.py" \
    --video-id 1 \
    --native-trace "$NATIVE_TRACE" \
    --records "$V1/records.jsonl" \
    --output "$REPORT" \
    --replay-root "$REPLAY" \
    --device cuda:0 \
    --max-frame 260 \
    --tolerance 2e-5
RC=$?
set -e

cd "$REPO"
while [ -e .git/index.lock ]; do sleep 5; done
git add "$REPORT"
git commit -m "Record corrected video1 native candidate parity" || true
git push origin HEAD || {
    git pull --rebase origin jev/counterfactual-rng-isolation-v4-20261006
    git push origin HEAD
}
echo "candidate_waiter_exit=$RC"
exit "$RC"
