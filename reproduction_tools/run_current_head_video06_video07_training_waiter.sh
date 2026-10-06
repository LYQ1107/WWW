#!/usr/bin/env bash

# Start the first corrected current-head three-way policy comparison as soon
# as the full video06/video07 provenance and runtime gates pass. This is a
# single-seed screening comparison, not the 24-video final result.

set -u

REPO=/data1/liuyeqiang/WWW_rng_fix_v4
RUNTIME=/home/liuyeqiang/WWW_jev_rng_v4_runtime
PYTHON=/home/liuyeqiang/anaconda3/envs/GMT/bin/python
FULL=$RUNTIME/formal_current_head_corrected_full
TRAINROOT=$RUNTIME/small_h8_training_current_head_video06_video07
DATASET=$TRAINROOT/compact_v1
SPLIT=$TRAINROOT/policy_split_seed20261003.json
METHODS=$TRAINROOT/methods_v1
REPORT=$REPO/reports/JEV_RNG_V4/CURRENT_HEAD_VIDEO06_VIDEO07_THREE_WAY.json
MARKDOWN=$REPO/docs/JEV_RNG_V4_CURRENT_HEAD_VIDEO06_VIDEO07_THREE_WAY.md
REPORT_ROOT=$REPO/reports/JEV_RNG_V4
LOG=$TRAINROOT/current_head_training.log

export PYTHONPATH="$REPO:$REPO/reproduction_tools:$REPO/third_party/CenterNet2"
mkdir -p "$TRAINROOT"
exec > >(tee -a "$LOG") 2>&1
echo "current_head_training_waiter_started=$(date -Is)"
echo "seed=20261003"

gate_ready() {
    jq -e '.status == "PASS"' "$1" >/dev/null 2>&1
}

while true; do
    for report in \
        "$REPORT_ROOT/CURRENT_HEAD_VIDEO06_PROVENANCE.json" \
        "$REPORT_ROOT/CURRENT_HEAD_VIDEO07_PROVENANCE.json" \
        "$REPORT_ROOT/RUNTIME_FEATURE_PARITY_VIDEO06_CURRENT_HEAD.json" \
        "$REPORT_ROOT/RUNTIME_FEATURE_PARITY_VIDEO07_CURRENT_HEAD_TOL4E5.json"; do
        if [ -f "$report" ] && ! gate_ready "$report"; then
            echo "gate_failed=$report"
            exit 2
        fi
    done
    if gate_ready "$REPORT_ROOT/CURRENT_HEAD_VIDEO06_PROVENANCE.json" && \
       gate_ready "$REPORT_ROOT/CURRENT_HEAD_VIDEO07_PROVENANCE.json" && \
       gate_ready "$REPORT_ROOT/RUNTIME_FEATURE_PARITY_VIDEO06_CURRENT_HEAD.json" && \
       gate_ready "$REPORT_ROOT/RUNTIME_FEATURE_PARITY_VIDEO07_CURRENT_HEAD_TOL4E5.json"; then
        break
    fi
    sleep 30
done
echo "corrected_video06_video07_gates_ready=$(date -Is)"

VIDEO06=$FULL/video06_records.jsonl
VIDEO07=$FULL/video07_records.jsonl
for path in "$VIDEO06" "$VIDEO07"; do
    if [ ! -f "$path" ]; then
        echo "missing_corrected_records=$path"
        exit 3
    fi
done

if [ ! -f "$DATASET/manifest.json" ]; then
    if [ -e "$DATASET" ]; then
        echo "incomplete_compact_output=$DATASET"
        exit 4
    fi
    "$PYTHON" -u "$REPO/reproduction_tools/jev_compact_dataset.py" \
        --input "$VIDEO06" "$VIDEO07" --output "$DATASET"
fi

if [ ! -f "$SPLIT" ]; then
    "$PYTHON" -u "$REPO/reproduction_tools/create_jev_policy_split.py" \
        --input "$VIDEO06" "$VIDEO07" --output "$SPLIT" \
        --seed 20261003 --val-fraction 0.2
fi

mkdir -p "$METHODS"
CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    "$PYTHON" -u "$REPO/reproduction_tools/run_jev_three_way_method.py" \
    --dataset "$DATASET" --split-manifest "$SPLIT" \
    --output "$METHODS/question_threshold" --model question_threshold \
    --hidden-dim 140 --epochs 20 --batch-size 128 --lr 0.001 --device cuda:0 \
    > "$TRAINROOT/question_threshold.log" 2>&1 &
P_THRESHOLD=$!

CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    "$PYTHON" -u "$REPO/reproduction_tools/run_jev_three_way_method.py" \
    --dataset "$DATASET" --split-manifest "$SPLIT" \
    --output "$METHODS/question_conditioned_mlp" --model question_conditioned_mlp \
    --hidden-dim 139 --epochs 20 --batch-size 128 --lr 0.001 --device cuda:0 \
    > "$TRAINROOT/question_conditioned_mlp.log" 2>&1 &
P_MLP=$!

CUDA_VISIBLE_DEVICES=9 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
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
    echo "current_head_three_way_training_failed=$RC"
    exit "$RC"
fi

"$PYTHON" -u "$REPO/reproduction_tools/aggregate_jev_three_way_compact.py" \
    --dataset "$DATASET" --split-manifest "$SPLIT" --methods-root "$METHODS" \
    --output "$REPORT" --markdown "$MARKDOWN"

cd "$REPO"
while [ -e .git/index.lock ]; do sleep 5; done
git add "$REPORT" "$MARKDOWN"
git commit -m "Add current-head corrected video06 video07 three-way results" || true
git push origin HEAD || {
    git pull --rebase origin jev/counterfactual-rng-isolation-v4-20261006
    git push origin HEAD
}
echo "current_head_three_way_complete=$(date -Is)"
