#!/usr/bin/env bash

# Verify the already-running current-head v10 video7 single/chunk gate.
# This script never launches, stops, or migrates the builders.

set -u

REPO=/data1/liuyeqiang/WWW_rng_fix_v4
RUNTIME=/home/liuyeqiang/WWW_jev_rng_v4_runtime/formal_same_code_video7_v10_current_head
REPORT=$REPO/reports/JEV_RNG_V4/FORMAL_GMT_INTRA_VIDEO_CHUNK_EQUIVALENCE_VIDEO07_CURRENT_HEAD_V10.json
PLAN=$REPO/reports/JEV_RNG_V4/INTRA_VIDEO_CHUNK_PLAN_VIDEO07.json
WARMUP=$RUNTIME/warmup/warmup_manifest.json
QUEUE=$RUNTIME/single_queue.json
SINGLE=$RUNTIME/single/video_07
CHUNKED=$RUNTIME/chunked/video_07
LOG=$RUNTIME/verifier.log
EXPECTED_SOURCE=161331e
EXPECTED_SOURCE_FULL=161331e2ea2636396eaf30e001aff61deb49cba4

exec > >(tee -a "$LOG") 2>&1
echo "verifier_started=$(date -Is) expected_source=$EXPECTED_SOURCE expected_source_full=$EXPECTED_SOURCE_FULL"
RESOLVED_SOURCE=$(git -C "$REPO" rev-parse "${EXPECTED_SOURCE}^{commit}" 2>/dev/null || true)
if [ "$RESOLVED_SOURCE" != "$EXPECTED_SOURCE_FULL" ]; then
    echo "source_commit_mapping_failed=$RESOLVED_SOURCE"
    exit 3
fi

while true; do
    if jq -e '.videos["7"].status == "FAILED"' "$QUEUE" >/dev/null 2>&1; then
        echo "single_queue_failed=$(date -Is)"
        exit 2
    fi
    if jq -e --arg c "$EXPECTED_SOURCE" \
        '.source_commit == $c and .status == "COMPLETE" and (.records | tonumber) == 3337' \
        "$SINGLE/manifest.json" >/dev/null 2>&1 && \
       jq -e --arg c "$EXPECTED_SOURCE" \
        '.source_commit == $c and .status == "COMPLETE" and (.records | tonumber) == 1500' \
        "$CHUNKED/chunk_0000/manifest.json" >/dev/null 2>&1 && \
       jq -e --arg c "$EXPECTED_SOURCE" \
        '.source_commit == $c and .status == "COMPLETE" and (.records | tonumber) == 1503' \
        "$CHUNKED/chunk_0001/manifest.json" >/dev/null 2>&1 && \
       jq -e --arg c "$EXPECTED_SOURCE" \
        '.source_commit == $c and .status == "COMPLETE" and (.records | tonumber) == 334' \
        "$CHUNKED/chunk_0002/manifest.json" >/dev/null 2>&1 && \
       jq -e --arg c "$EXPECTED" '.source_commit == $c and .status == "PASS"' \
        "$WARMUP" >/dev/null 2>&1; then
        break
    fi
    sleep 30
done

echo "all_current_head_manifests_ready=$(date -Is)"
set +e
PYTHONPATH="$REPO:$REPO/reproduction_tools:$REPO/third_party/CenterNet2" \
    /home/liuyeqiang/anaconda3/envs/GMT/bin/python -u \
    "$REPO/reproduction_tools/verify_jev_intra_video_chunk_equivalence.py" \
    --single-records "$SINGLE/records.jsonl" \
    --chunk-records \
        "$CHUNKED/chunk_0000/records.jsonl" \
        "$CHUNKED/chunk_0001/records.jsonl" \
        "$CHUNKED/chunk_0002/records.jsonl" \
    --single-manifest "$SINGLE/manifest.json" \
    --chunk-manifests \
        "$CHUNKED/chunk_0000/manifest.json" \
        "$CHUNKED/chunk_0001/manifest.json" \
        "$CHUNKED/chunk_0002/manifest.json" \
    --chunk-plan "$PLAN" \
    --warmup-manifest "$WARMUP" \
    --expected-source-commit "$EXPECTED_SOURCE" \
    --expected-source-commit-full "$EXPECTED_SOURCE_FULL" \
    --output "$REPORT"
RC=$?
set -e

cd "$REPO"
while [ -e .git/index.lock ]; do sleep 5; done
git add reproduction_tools/run_current_head_formal_v10_verifier.sh "$REPORT"
git commit -m "Record current-head formal chunk equivalence gate" || true
git push origin HEAD || {
    git pull --rebase origin jev/counterfactual-rng-isolation-v4-20261006
    git push origin HEAD
}
echo "current_head_formal_v10_verifier_exit=$RC"
exit "$RC"
