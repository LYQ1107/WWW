#!/usr/bin/env bash
set -u

REPO=/data1/liuyeqiang/WWW
STAGE1_OUT="$REPO/outputs/stage1_single_gpu"
STAGE1_FINAL="$STAGE1_OUT/model_16000.pth"
STAGE1_VALIDATION="$STAGE1_OUT/checkpoint_validation.json"
STAGE2_OUT="$REPO/outputs/stage2_single_gpu"
CHAIN_LOG="$STAGE2_OUT/chain.log"

mkdir -p "$STAGE2_OUT"

# Wait for the canonical 20,000-global-step Stage1 run.  This output directory
# resumes the original global-4000 checkpoint, so its local final filename is
# model_16000.pth while the scheduler reaches epoch 20000.
# Require a stable file size and the matching last_checkpoint
# marker so Stage2 cannot consume a partially written file.
while :; do
    if [[ -s "$STAGE1_FINAL" ]] && [[ "$(tr -d '\n' < "$STAGE1_OUT/last_checkpoint" 2>/dev/null)" == "model_16000.pth" ]]; then
        size_before=$(stat -c '%s' "$STAGE1_FINAL")
        sleep 30
        size_after=$(stat -c '%s' "$STAGE1_FINAL" 2>/dev/null || echo 0)
        if [[ "$size_before" == "$size_after" ]]; then
            break
        fi
    fi
    sleep 60
done

validation_report_ok() {
    [[ -f "$STAGE1_VALIDATION" ]] || return 1
    /home/liuyeqiang/anaconda3/envs/GMT/bin/python - "$STAGE1_VALIDATION" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
if not (
    report.get("status") == "PASS"
    and report.get("iteration") == 16000
    and report.get("scheduler_last_epoch") == 20000
):
    raise SystemExit(1)
PY
}

# A stale or failed report must never authorize Stage2.  The validator refuses
# to overwrite an existing report, so regenerate into a private temporary
# path and atomically replace the report only after validation succeeds.
if ! validation_report_ok; then
    validation_tmp_dir=$(mktemp -d "$STAGE1_OUT/.checkpoint_validation.XXXXXX")
    env PYTHONPATH="$REPO:$REPO/third_party/CenterNet2:$REPO/reproduction_tools" \
        /home/liuyeqiang/anaconda3/envs/GMT/bin/python -u \
        "$REPO/reproduction_tools/validate_training_checkpoint.py" \
        --checkpoint "$STAGE1_FINAL" \
        --expected-iteration 16000 \
        --expected-scheduler-iteration 20000 \
        --output "$validation_tmp_dir/checkpoint_validation.json" \
        >> "$CHAIN_LOG" 2>&1 || exit 1
    mv "$validation_tmp_dir/checkpoint_validation.json" "$STAGE1_VALIDATION"
    rmdir "$validation_tmp_dir"
fi

validation_report_ok || exit 1

# The trainer normally exits after saving the final checkpoint; wait for its
# process to release GPU0 before launching Stage2.
while pgrep -f 'train_net.py.*OUTPUT_DIR outputs/stage1_single_gpu' >/dev/null 2>&1; do
    sleep 30
done

exec env \
    CUDA_VISIBLE_DEVICES=0 \
    GMT_DISTRIBUTED_BACKEND=gloo \
    GMT_CPU_COLLECTIVES=1 \
    GMT_CHECKPOINT_BACKBONE=1 \
    GMT_TRAIN_PROGRESS=1 \
    OMP_NUM_THREADS=1 \
    /home/liuyeqiang/anaconda3/envs/GMT/bin/python -u "$REPO/train_net.py" \
    --num-gpus 1 \
    --config-file "$REPO/configs/VISION_stage2.yaml" \
    SOLVER.IMS_PER_BATCH 1 \
    SOLVER.TRAIN_ITER 20000 \
    MODEL.WEIGHTS "$STAGE1_FINAL" \
    OUTPUT_DIR "$STAGE2_OUT" \
    >> "$CHAIN_LOG" 2>&1
