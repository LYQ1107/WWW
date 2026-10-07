#!/usr/bin/env bash

# v2-bound aftercare for the live corrected video01 builder.
# This waiter never starts, stops, migrates, or modifies the builder itself.
# It must not be replaced by the older aftercare script, which points to the
# pre-v2 artifact directory.

set -u

REPO=/data1/liuyeqiang/WWW_rng_fix_v4
RUNTIME=/home/liuyeqiang/WWW_jev_rng_v4_runtime
PYTHON=/home/liuyeqiang/anaconda3/envs/GMT/bin/python
CANONICAL_COMMIT=4108f18f5040432f68d55872e81e9f76e9acd08f
ROOT=$RUNTIME/formal_current_head_corrected_full_v2
TRACE=$RUNTIME/formal_current_head_off_trace/video01/trace_video_01.jsonl
NATIVE_TRACE=$RUNTIME/native_video1_corrected_v3/native_off_trace_video01.jsonl
RECORDS=$ROOT/video01_records.jsonl
MANIFEST=$RECORDS.manifest.json
REPORT_ROOT=$REPO/reports/JEV_RNG_V4
METHOD_ROOT=$RUNTIME/small_h8_training_v4_canonical_features/methods_v1
TRAINING=$REPORT_ROOT/CORRECTED_V4_SMALL_H8_THREE_WAY_VIDEO06_VIDEO07.json
PROVENANCE=$REPORT_ROOT/VIDEO01_V2_PROVENANCE_20261007.json
CHECKPOINT=$REPORT_ROOT/VIDEO01_V2_CHECKPOINT_WRAPPER_20261007.json
PARITY=$REPORT_ROOT/VIDEO01_V2_RUNTIME_FEATURE_PARITY_20261007.json
CANDIDATE=$REPORT_ROOT/VIDEO01_V2_REACTIVATION_CANDIDATE_PARITY_20261007.json
STABILITY=$REPORT_ROOT/VIDEO01_V2_RUNTIME_FEATURE_STABILITY_20261007.json
CLOSED_LOOP=$REPORT_ROOT/VIDEO01_V2_THREE_WAY_TRACKING_20261007.json
HARD_GATES=$REPORT_ROOT/VIDEO01_V2_HARD_GATES_20261007.json
LOG=$RUNTIME/formal_current_head_video01_v2_aftercare.log

export PYTHONPATH="$REPO:$REPO/reproduction_tools:$REPO/third_party/CenterNet2"
mkdir -p "$REPORT_ROOT"
exec > >(tee -a "$LOG") 2>&1
echo "video01_v2_aftercare_started=$(date -Is)"
echo "canonical_commit=$CANONICAL_COMMIT"

while [ ! -f "$MANIFEST" ]; do
    sleep 30
done
echo "video01_v2_manifest_seen=$(date -Is)"

if ! jq -e --arg commit "$CANONICAL_COMMIT" \
    '.status == "PASS" and (.records | tonumber) == 8995 and .source_commit == $commit' \
    "$MANIFEST" >/dev/null 2>&1; then
    echo "v2_manifest_binding_failed=$MANIFEST"
    exit 2
fi

set +e
"$PYTHON" -u "$REPO/reproduction_tools/validate_jev_video_artifact.py" \
    --manifest "$MANIFEST" --records "$RECORDS" \
    --output "$PROVENANCE" --expected-records 8995 --horizon 8
PROVENANCE_RC=$?
set -e

set +e
"$PYTHON" -u "$REPO/reproduction_tools/validate_v2_controller_checkpoints.py" \
    --methods-root "$METHOD_ROOT" --records "$RECORDS" --manifest "$MANIFEST" \
    --training-report "$TRAINING" --expected-source-commit "$CANONICAL_COMMIT" \
    --output "$CHECKPOINT"
CHECKPOINT_RC=$?
set -e

if [ "$PROVENANCE_RC" -eq 0 ]; then
    set +e
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        JEV_CACHE_PATH=/data1/liuyeqiang/WWW/outputs/research_final_v2/off/perception_cache_train \
    "$PYTHON" -u "$REPO/reproduction_tools/run_jev_runtime_feature_parity.py" \
        --video-id 1 --trace "$TRACE" --records "$RECORDS" \
        --output "$PARITY" --device cuda:0 --tolerance 2e-5
    PARITY_RC=$?

    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        JEV_CACHE_PATH=/data1/liuyeqiang/WWW/outputs/research_final_v2/off/perception_cache_train \
        "$PYTHON" -u "$REPO/reproduction_tools/compare_reactivation_candidates.py" \
        --video-id 1 --native-trace "$NATIVE_TRACE" --records "$RECORDS" \
        --output "$CANDIDATE" \
        --replay-root "$RUNTIME/reactivation_candidate_replay_video01_v2" \
        --device cuda:0 --max-frame 1000000 --tolerance 2e-5
    CANDIDATE_RC=$?
    set +e

    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        JEV_CACHE_PATH=/data1/liuyeqiang/WWW/outputs/research_final_v2/off/perception_cache_train \
        "$PYTHON" -u "$REPO/reproduction_tools/run_jev_feature_parity_stability.py" \
        --video-id 1 --trace "$TRACE" --records "$RECORDS" \
        --output "$STABILITY" \
        --runtime-root "$RUNTIME/formal_current_head_video01_v2_parity_stability" \
        --device cuda:0 --max-frame 1000000 --repetitions 3 --tolerance 2e-5
    STABILITY_RC=$?
    set -e
else
    PARITY_RC=3
    CANDIDATE_RC=3
    STABILITY_RC=3
    echo "v2_provenance_failed_skip_runtime_replays"
fi

"$PYTHON" -u "$REPO/reproduction_tools/update_video01_hard_gates.py" \
    --manifest "$MANIFEST" --provenance "$PROVENANCE" --parity "$PARITY" \
    --candidate "$CANDIDATE" --stability "$STABILITY" \
    --closed-loop "$CLOSED_LOOP" --output "$HARD_GATES" || true

# Do not run a controller on a failed v2 parity/candidate gate. A parity
# failure is BLOCKED evidence, never a learned-model failure. If all required
# v2 gates pass, use the existing corrected-v4 calibrated controllers and keep
# the tracking report separate from legacy screening reports.
if [ "$PROVENANCE_RC" -eq 0 ] && [ "$CHECKPOINT_RC" -eq 0 ] && \
   [ "$PARITY_RC" -eq 0 ] && [ "$CANDIDATE_RC" -eq 0 ] && [ "$STABILITY_RC" -eq 0 ]; then
    set +e
    "$PYTHON" -u "$REPO/reproduction_tools/run_corrected_v4_tracking.py" \
        --video-id 1 --trace "$TRACE" --records "$RECORDS" \
        --methods-root "$METHOD_ROOT" --output-root "$RUNTIME/formal_current_head_video01_v2_closed_loop" \
        --output-report "$CLOSED_LOOP" --provenance-report "$PROVENANCE" \
        --candidate-report "$CANDIDATE" --feature-parity-report "$PARITY" \
        --stability-report "$STABILITY" \
        --formal-report "$REPORT_ROOT/FORMAL_GMT_INTRA_VIDEO_CHUNK_EQUIVALENCE_VIDEO07_CURRENT_HEAD_V10.json" \
        --training-report "$TRAINING" --device cuda:0 --tolerance 2e-5
    CLOSED_LOOP_RC=$?
    set -e
else
    CLOSED_LOOP_RC=3
    echo "v2_closed_loop_blocked_by_pre_gate"
fi

"$PYTHON" -u "$REPO/reproduction_tools/update_video01_hard_gates.py" \
    --manifest "$MANIFEST" --provenance "$PROVENANCE" --parity "$PARITY" \
    --candidate "$CANDIDATE" --stability "$STABILITY" \
    --closed-loop "$CLOSED_LOOP" --output "$HARD_GATES" || true

echo "video01_v2_aftercare_complete=$(date -Is) provenance_rc=$PROVENANCE_RC checkpoint_rc=$CHECKPOINT_RC parity_rc=$PARITY_RC candidate_rc=$CANDIDATE_RC stability_rc=$STABILITY_RC closed_loop_rc=$CLOSED_LOOP_RC"
if [ "$PROVENANCE_RC" -ne 0 ] || [ "$CHECKPOINT_RC" -ne 0 ] || \
   [ "$PARITY_RC" -ne 0 ] || [ "$CANDIDATE_RC" -ne 0 ] || [ "$STABILITY_RC" -ne 0 ]; then
    exit 4
fi
exit "$CLOSED_LOOP_RC"
