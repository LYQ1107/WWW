#!/usr/bin/env bash

# Wait for the corrected video7 builder to release GPU7, then build and verify
# a same-code single-worker reference against three deterministic chunks.
# This is a gate for future full H=8 chunking only; it never touches video1.

set -u

REPO=/data1/liuyeqiang/WWW_rng_fix_v4
RUNTIME=/home/liuyeqiang/WWW_jev_rng_v4_runtime
PARTITION=$RUNTIME/small_h8_partition_fixed
PLAN=$REPO/reports/JEV_RNG_V4/INTRA_VIDEO_CHUNK_PLAN_VIDEO07.json
CACHE=/data1/liuyeqiang/WWW/outputs/research_final_v2/off/perception_cache_train
ANNOTATIONS=/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json
CHECKPOINT=/data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth
CONFIG=$REPO/configs/VISION_test.yaml
V7=$RUNTIME/small_h8_rng_controlled_v4_current_head_75b0aea/video_07
FORMAL=$RUNTIME/formal_same_code_video7_v9_head
QUEUE=$FORMAL/single_queue.json
REPORT=$REPO/reports/JEV_RNG_V4/FORMAL_GMT_INTRA_VIDEO_CHUNK_EQUIVALENCE_VIDEO07_CORRECTED_PINNED.json
LOG=$RUNTIME/formal_same_code_video7_v9_head_waiter.log

mkdir -p "$FORMAL"
exec > >(tee -a "$LOG") 2>&1
echo "watcher_started=$(date -Is)"

# The corrected v7 artifact must be complete before GPU7 is reused. Its source
# commit is recorded as input provenance; A/B below share the current checkout.
while ! jq -e '.status == "COMPLETE" and (.records | tonumber) == 3337' \
    "$V7/manifest.json" >/dev/null 2>&1; do
    sleep 30
done
INPUT_SOURCE=$(jq -r '.source_commit // empty' "$V7/manifest.json")
SOURCE_COMMIT=$(git -C "$REPO" rev-parse HEAD)
echo "v7_ready=$(date -Is) input_source=$INPUT_SOURCE formal_source=$SOURCE_COMMIT"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$REPO:$REPO/reproduction_tools:$REPO/third_party/CenterNet2"
export JEV_PROVENANCE_SOURCE_COMMIT="$SOURCE_COMMIT"

echo "warmup_started=$(date -Is)"
CUDA_VISIBLE_DEVICES=7 "$REPO/reproduction_tools/warmup_jev_intra_video_chunks.py" \
    --partition-manifest "$PARTITION/partition_manifest.json" \
    --chunk-plan "$PLAN" \
    --cache "$CACHE" \
    --annotations "$ANNOTATIONS" \
    --checkpoint "$CHECKPOINT" \
    --config-file "$CONFIG" \
    --output "$FORMAL/warmup" \
    --device cuda:0 --view-num 2 --history-limit 80
echo "warmup_done=$(date -Is)"

pids=()
CUDA_VISIBLE_DEVICES=1 "$REPO/reproduction_tools/run_jev_full_h8_fast_worker.py" \
    --partition-root "$PARTITION" --cache "$CACHE" --annotations "$ANNOTATIONS" \
    --checkpoint "$CHECKPOINT" --config-file "$CONFIG" --output-root "$FORMAL/single" \
    --queue "$QUEUE" --worker-id corrected-v4-v9-single-gpu1 --device cuda:0 \
    --view-num 2 --history-limit 80 --horizon 8 > "$FORMAL/single.log" 2>&1 &
pids+=("$!")

for spec in "0 3" "1 5" "2 7"; do
    set -- $spec
    IDX=$1
    GPU=$2
    CUDA_VISIBLE_DEVICES=$GPU "$REPO/reproduction_tools/run_jev_intra_video_chunk.py" \
        --partition-manifest "$PARTITION/partition_manifest.json" \
        --chunk-plan "$PLAN" --warmup-root "$FORMAL/warmup" \
        --chunk-index "$IDX" --cache "$CACHE" --annotations "$ANNOTATIONS" \
        --checkpoint "$CHECKPOINT" --config-file "$CONFIG" --output-root "$FORMAL/chunked" \
        --device cuda:0 --view-num 2 --history-limit 80 --horizon 8 \
        > "$FORMAL/chunk_$IDX.log" 2>&1 &
    pids+=("$!")
done

RC=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        RC=1
    fi
done
if [ "$RC" -ne 0 ]; then
    echo "formal_build_failed=$(date -Is)"
else
    echo "formal_builds_done=$(date -Is)"
fi

if [ "$RC" -eq 0 ] && [ -f "$FORMAL/single/video_07/records.jsonl" ] && \
   [ -f "$FORMAL/warmup/warmup_manifest.json" ] && \
   [ -f "$FORMAL/chunked/video_07/chunk_0000/records.jsonl" ] && \
   [ -f "$FORMAL/chunked/video_07/chunk_0001/records.jsonl" ] && \
   [ -f "$FORMAL/chunked/video_07/chunk_0002/records.jsonl" ]; then
    set +e
    PYTHONPATH="$REPO:$REPO/reproduction_tools:$REPO/third_party/CenterNet2" \
        /home/liuyeqiang/anaconda3/envs/GMT/bin/python -u \
        "$REPO/reproduction_tools/verify_jev_intra_video_chunk_equivalence.py" \
        --single-records "$FORMAL/single/video_07/records.jsonl" \
        --chunk-records \
            "$FORMAL/chunked/video_07/chunk_0000/records.jsonl" \
            "$FORMAL/chunked/video_07/chunk_0001/records.jsonl" \
            "$FORMAL/chunked/video_07/chunk_0002/records.jsonl" \
        --single-manifest "$FORMAL/single/video_07/manifest.json" \
        --chunk-manifests \
            "$FORMAL/chunked/video_07/chunk_0000/manifest.json" \
            "$FORMAL/chunked/video_07/chunk_0001/manifest.json" \
            "$FORMAL/chunked/video_07/chunk_0002/manifest.json" \
        --chunk-plan "$PLAN" \
        --warmup-manifest "$FORMAL/warmup/warmup_manifest.json" \
        --output "$REPORT"
    VERIFY_RC=$?
    set -e
    if [ "$VERIFY_RC" -ne 0 ]; then
        RC="$VERIFY_RC"
    fi
else
    echo "formal_artifacts_missing=$(date -Is)"
    RC=1
fi

if [ ! -f "$REPORT" ]; then
    /home/liuyeqiang/anaconda3/envs/GMT/bin/python - "$REPORT" "$SOURCE_COMMIT" "$INPUT_SOURCE" <<'PY'
import json
import sys
from pathlib import Path

path, source_commit, input_source = sys.argv[1:]
Path(path).write_text(json.dumps({
    "status": "ERROR",
    "canonical_authority": "NO",
    "error": "same-code formal build or verification did not complete",
    "formal_source_commit": source_commit,
    "input_video7_source_commit": input_source,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
fi

cd "$REPO"
while [ -e .git/index.lock ]; do sleep 5; done
git add "$REPORT"
git commit -m "Record same-code corrected video7 chunk gate" || true
git push origin HEAD || {
    git pull --rebase origin jev/counterfactual-rng-isolation-v4-20261006
    git push origin HEAD
}
echo "formal_waiter_exit=$RC"
exit "$RC"
