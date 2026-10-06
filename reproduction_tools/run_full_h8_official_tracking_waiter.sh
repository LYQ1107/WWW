#!/usr/bin/env bash

# Wait for the locked Full-H=8 postprocess and corrected-v4 runtime gates,
# then launch the one-shot official closed-loop controller comparison.
# This waiter never reruns GMT OFF and never touches any active builder.

set -u

REPO=/data1/liuyeqiang/WWW_rng_fix_v4
RUNTIME=${FULL_H8_RUNTIME_ROOT:-/home/liuyeqiang/WWW_jev_rng_v4_runtime/full_h8_current_head}
PYTHON=/home/liuyeqiang/anaconda3/envs/GMT/bin/python
FORMAL=$REPO/reports/JEV_RNG_V4/FORMAL_GMT_INTRA_VIDEO_CHUNK_EQUIVALENCE_VIDEO07_CURRENT_HEAD_V10.json
BASELINE=/data1/liuyeqiang/WWW_jev_v2/reports/CHECKPOINT_BASELINE_AUDIT_20261006.json
CHECKPOINT=/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth
CONFIG=$REPO/configs/VISION_test.yaml
TEST_CACHE=/data1/liuyeqiang/WWW/outputs/research_final_v2/off/perception_cache_test
DATASET=/data/DATASETS/TRACKING/JDE/VisionTrack
OUTROOT=${OFFICIAL_TRACKING_OUTPUT_ROOT:-$RUNTIME/official_tracking}
REPORT=$REPO/reports/JEV_RNG_V4/FULL_H8_OFFICIAL_TRACKING.json
MARKDOWN=$REPO/docs/JEV_RNG_V4_FULL_H8_OFFICIAL_TRACKING.md
LOG=$RUNTIME/full_h8_official_tracking_waiter.log

mkdir -p "$RUNTIME"
exec > >(tee -a "$LOG") 2>&1
echo "official_tracking_waiter_started=$(date -Is)"

required_reports=(
    "$FORMAL"
    "$REPO/reports/JEV_RNG_V4/CURRENT_HEAD_VIDEO01_PROVENANCE.json"
    "$REPO/reports/JEV_RNG_V4/RUNTIME_FEATURE_PARITY_VIDEO01_CURRENT_HEAD.json"
    "$REPO/reports/JEV_RNG_V4/REACTIVATION_CANDIDATE_PARITY_CURRENT_HEAD_VIDEO01.json"
    "$REPO/reports/JEV_RNG_V4/RUNTIME_FEATURE_PARITY_STABILITY_VIDEO01_CURRENT_HEAD.json"
    "$REPO/reports/JEV_RNG_V4/CURRENT_HEAD_VIDEO01_THREE_WAY_TRACKING.json"
    "$REPO/reports/JEV_RNG_V4/FULL_H8_CURRENT_HEAD_AFTERCARE.json"
    "$RUNTIME/POSTPROCESS_MANIFEST.json"
    "$RUNTIME/FIRST_ROUND_REPORT.json"
)

all_pass() {
    local path
    for path in "${required_reports[@]}"; do
        if [ ! -f "$path" ] || ! jq -e '.status == "PASS"' "$path" >/dev/null 2>&1; then
            return 1
        fi
    done
    return 0
}

while true; do
    if [ -f "$FORMAL" ]; then
        formal_status=$(jq -r '.status // "INVALID"' "$FORMAL" 2>/dev/null || echo INVALID)
        if [ "$formal_status" = "FAIL" ]; then
            echo "official_tracking_not_authorized=formal_gate_failed"
            exit 6
        fi
    fi
    if all_pass; then
        break
    fi
    sleep 60
done

if [ ! -f "$CHECKPOINT" ] || [ ! -f "$BASELINE" ] || [ ! -f "$CONFIG" ] || \
   [ ! -f "$TEST_CACHE/index.jsonl" ] || [ ! -d "$DATASET" ]; then
    echo "official_tracking_inputs_missing"
    exit 7
fi

if [ -f "$REPORT" ]; then
    if jq -e '.status == "PASS"' "$REPORT" >/dev/null 2>&1; then
        echo "official_tracking_already_published=$(date -Is)"
        exit 0
    fi
    echo "refusing_to_overwrite_existing_official_report=$REPORT"
    exit 8
fi
if [ -e "$OUTROOT" ]; then
    echo "refusing_to_overwrite_existing_official_output=$OUTROOT"
    exit 9
fi

echo "all_official_tracking_gates_pass=$(date -Is)"
set +e
PYTHONPATH="$REPO:$REPO/reproduction_tools:$REPO/third_party/CenterNet2" \
    "$PYTHON" -u "$REPO/reproduction_tools/run_full_h8_official_tracking.py" \
    --runtime-root "$RUNTIME" \
    --repo-root "$REPO" \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --test-cache "$TEST_CACHE" \
    --dataset "$DATASET" \
    --formal-gate-report "$FORMAL" \
    --baseline-audit "$BASELINE" \
    --gpus 4,8,9 \
    --output-root "$OUTROOT" \
    --output-report "$REPORT" \
    --output-markdown "$MARKDOWN"
RC=$?
set -e
if [ "$RC" -ne 0 ]; then
    echo "official_tracking_failed=$RC"
    exit "$RC"
fi

cd "$REPO"
while [ -e .git/index.lock ]; do
    sleep 5
done
git add "$REPORT" "$MARKDOWN"
git commit -m "Publish official full H8 tracking comparison" || true
git push origin HEAD || {
    git pull --rebase origin jev/counterfactual-rng-isolation-v4-20261006
    git push origin HEAD
}
echo "official_tracking_publication_complete=$(date -Is)"
exit 0
