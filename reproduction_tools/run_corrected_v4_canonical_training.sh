#!/usr/bin/env bash

# Train the first single-seed corrected-v4 three-way policy comparison from
# the already completed canonical-feature v6/v7 shards.  This is deliberately
# separate from the later current-head rebuild: the latter remains available
# for provenance/audit work and never replaces this explicitly authorized
# canonical-feature training pool.

set -euo pipefail

REPO=/data1/liuyeqiang/WWW_rng_fix_v4
RUNTIME=/home/liuyeqiang/WWW_jev_rng_v4_runtime
PYTHON=/home/liuyeqiang/anaconda3/envs/GMT/bin/python
CANONICAL=$RUNTIME/small_h8_rng_controlled_v3_canonical_features
TRAINROOT=${TRAINROOT:-$RUNTIME/small_h8_training_v4_canonical_features}
DATASET=$TRAINROOT/compact_v1
SPLIT=$TRAINROOT/policy_split_seed20261003.json
METHODS=$TRAINROOT/methods_v1
REPORT=$REPO/reports/JEV_RNG_V4/CORRECTED_V4_SMALL_H8_THREE_WAY_VIDEO06_VIDEO07.json
MARKDOWN=$REPO/docs/JEV_RNG_V4_CORRECTED_V4_SMALL_H8_THREE_WAY_VIDEO06_VIDEO07.md
LOG=$TRAINROOT/canonical_training.log

mkdir -p "$TRAINROOT"
exec > >(tee -a "$LOG") 2>&1
echo "canonical_training_started=$(date -Is)"
echo "canonical_root=$CANONICAL"
echo "train_root=$TRAINROOT"
echo "seed=20261003"

for video in 06 07; do
    manifest="$CANONICAL/video_${video}/manifest.json"
    jq -e --argjson id "$((10#$video))" \
        '.status == "COMPLETE" and (.records | tonumber) > 0 and .video_id == $id and .horizon == 8' \
        "$manifest" >/dev/null
done

if [ ! -f "$DATASET/manifest.json" ]; then
    if [ -e "$DATASET" ]; then
        echo "incomplete compact output exists: $DATASET" >&2
        exit 2
    fi
    PYTHONPATH="$REPO:$REPO/reproduction_tools:$REPO/third_party/CenterNet2" \
        "$PYTHON" -u "$REPO/reproduction_tools/jev_compact_dataset.py" \
        --input "$CANONICAL/video_06/records.jsonl" "$CANONICAL/video_07/records.jsonl" \
        --output "$DATASET"
fi

if [ ! -f "$SPLIT" ]; then
    PYTHONPATH="$REPO:$REPO/reproduction_tools:$REPO/third_party/CenterNet2" \
        "$PYTHON" -u "$REPO/reproduction_tools/create_jev_policy_split.py" \
        --input "$CANONICAL/video_06/records.jsonl" "$CANONICAL/video_07/records.jsonl" \
        --output "$SPLIT" --seed 20261003 --val-fraction 0.2
fi

mkdir -p "$METHODS"
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH="$REPO:$REPO/reproduction_tools:$REPO/third_party/CenterNet2" \
    "$PYTHON" -u "$REPO/reproduction_tools/run_jev_three_way_method.py" \
    --dataset "$DATASET" --split-manifest "$SPLIT" \
    --output "$METHODS/question_threshold" --model question_threshold \
    --hidden-dim 140 --epochs 20 --batch-size 128 --lr 0.001 --device cuda:0 \
    > "$TRAINROOT/question_threshold.log" 2>&1 &
P_THRESHOLD=$!

CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH="$REPO:$REPO/reproduction_tools:$REPO/third_party/CenterNet2" \
    "$PYTHON" -u "$REPO/reproduction_tools/run_jev_three_way_method.py" \
    --dataset "$DATASET" --split-manifest "$SPLIT" \
    --output "$METHODS/question_conditioned_mlp" --model question_conditioned_mlp \
    --hidden-dim 139 --epochs 20 --batch-size 128 --lr 0.001 --device cuda:0 \
    > "$TRAINROOT/question_conditioned_mlp.log" 2>&1 &
P_MLP=$!

CUDA_VISIBLE_DEVICES=9 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    PYTHONPATH="$REPO:$REPO/reproduction_tools:$REPO/third_party/CenterNet2" \
    "$PYTHON" -u "$REPO/reproduction_tools/run_jev_three_way_method.py" \
    --dataset "$DATASET" --split-manifest "$SPLIT" \
    --output "$METHODS/jev" --model jev \
    --hidden-dim 128 --epochs 20 --batch-size 128 --lr 0.001 --device cuda:0 \
    > "$TRAINROOT/jev.log" 2>&1 &
P_JEV=$!

RC=0
wait "$P_THRESHOLD" || RC=1
wait "$P_MLP" || RC=1
wait "$P_JEV" || RC=1
if [ "$RC" -ne 0 ]; then
    echo "three_way_training_failed=$RC"
    exit "$RC"
fi

PYTHONPATH="$REPO:$REPO/reproduction_tools:$REPO/third_party/CenterNet2" \
    "$PYTHON" -u "$REPO/reproduction_tools/aggregate_jev_three_way_compact.py" \
    --dataset "$DATASET" --split-manifest "$SPLIT" --methods-root "$METHODS" \
    --output "$REPORT" --markdown "$MARKDOWN"

cd "$REPO"
while [ -e .git/index.lock ]; do sleep 5; done
git add "$REPORT" "$MARKDOWN"
git commit -m "Add corrected canonical-feature three-way results" || true
git push origin HEAD || {
    git pull --rebase origin jev/counterfactual-rng-isolation-v4-20261006
    git push origin HEAD
}
echo "canonical_training_complete=$(date -Is)"
