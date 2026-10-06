#!/usr/bin/env bash

# Wait for the formal same-code video-07 gate, then supervise the authorized
# 24-video canonical H=8 build and its CPU-only aftercare.  This process is
# deliberately fail-closed: no worker is launched while the formal report is
# missing, incomplete, or failed.

set -u

REPO=/data1/liuyeqiang/WWW_rng_fix_v4
RUNTIME=${FULL_H8_RUNTIME_ROOT:-/home/liuyeqiang/WWW_jev_rng_v4_runtime/full_h8_current_head}
PYTHON=/home/liuyeqiang/anaconda3/envs/GMT/bin/python
# The 3-video small_h8_partition_fixed is only the formal gate fixture. The
# canonical production build must use the locked 24-video partition.
PARTITION_SOURCE=/home/liuyeqiang/WWW_jev_full_h8_runtime/partition
PARTITION=$RUNTIME/partition
FORMAL=$REPO/reports/JEV_RNG_V4/FORMAL_GMT_INTRA_VIDEO_CHUNK_EQUIVALENCE_VIDEO07_CURRENT_HEAD_V10.json
QUEUE=$RUNTIME/queue_state.json
OUTPUT_ROOT=$RUNTIME/formal_h8_full
LOG_ROOT=$RUNTIME/scheduler_logs
LOG=$RUNTIME/full_h8_supervisor.log
PUBLISHED_REPORT=$REPO/reports/JEV_RNG_V4/FULL_H8_CURRENT_HEAD_AFTERCARE.json
PUBLISHED_MARKDOWN=$REPO/docs/JEV_RNG_V4_FULL_H8_CURRENT_HEAD_AFTERCARE.md

mkdir -p "$RUNTIME"
exec > >(tee -a "$LOG") 2>&1
echo "full_h8_supervisor_started=$(date -Is)"
echo "formal_gate=$FORMAL"
echo "partition_source=$PARTITION_SOURCE"

if [ ! -f "$PARTITION_SOURCE/partition_manifest.json" ]; then
    echo "missing_partition_manifest=$PARTITION_SOURCE/partition_manifest.json"
    exit 3
fi

# Aftercare expects the partition under the same runtime root as the shards.
# Reuse the immutable prepared partition through a symlink; never copy or
# rewrite its trace/cache files.
if [ -L "$PARTITION" ]; then
    if [ "$(readlink -f "$PARTITION")" != "$(readlink -f "$PARTITION_SOURCE")" ]; then
        echo "partition_symlink_mismatch=$(readlink -f "$PARTITION")"
        exit 4
    fi
elif [ -e "$PARTITION" ]; then
    source_sha=$(sha256sum "$PARTITION_SOURCE/partition_manifest.json" | awk '{print $1}')
    existing_sha=$(sha256sum "$PARTITION/partition_manifest.json" 2>/dev/null | awk '{print $1}')
    if [ -z "$existing_sha" ] || [ "$source_sha" != "$existing_sha" ]; then
        echo "partition_manifest_mismatch source=$source_sha existing=$existing_sha"
        exit 4
    fi
else
    ln -s "$PARTITION_SOURCE" "$PARTITION"
fi

while true; do
    if [ -f "$FORMAL" ]; then
        status=$(jq -r '.status // "INVALID"' "$FORMAL" 2>/dev/null || echo INVALID)
        if [ "$status" = "PASS" ]; then
            set +e
            PYTHONPATH="$REPO:$REPO/reproduction_tools:$REPO/third_party/CenterNet2" \
                "$PYTHON" -c \
                'from pathlib import Path; from jev_full_h8_authorization import read_formal_authorization; read_formal_authorization(Path(__import__("sys").argv[1]))' \
                "$FORMAL"
            authorization_rc=$?
            set -e
            if [ "$authorization_rc" -ne 0 ]; then
                echo "formal_authorization_validation_failed=$authorization_rc"
                exit 5
            fi
            break
        fi
        if [ "$status" = "FAIL" ]; then
            echo "formal_gate_failed; full_h8_not_authorized=$(date -Is)"
            exit 6
        fi
    fi
    sleep 30
done

partition_status=$(jq -r '.status // "INVALID"' "$PARTITION/partition_manifest.json" 2>/dev/null || echo INVALID)
partition_videos=$(jq '.videos | length' "$PARTITION/partition_manifest.json" 2>/dev/null || echo 0)
partition_records=$(jq -r '.total_main_decisions // -1' "$PARTITION/partition_manifest.json" 2>/dev/null || echo -1)
if [ "$partition_status" != "COMPLETE" ] || [ "$partition_videos" -ne 24 ] || [ "$partition_records" -le 0 ]; then
    echo "partition_not_locked status=$partition_status videos=$partition_videos records=$partition_records"
    exit 7
fi

if pgrep -af "run_jev_full_h8_fast_scheduler.py.*$(basename "$QUEUE")" | rg -v "pgrep -af" >/dev/null 2>&1; then
    echo "scheduler_already_live_for_queue=$QUEUE"
    exit 8
fi

echo "formal_authorization_passed=$(date -Is)"
set +e
PYTHONPATH="$REPO:$REPO/reproduction_tools:$REPO/third_party/CenterNet2" \
    "$PYTHON" -u "$REPO/reproduction_tools/run_jev_full_h8_fast_scheduler.py" \
    --partition-root "$PARTITION_SOURCE" \
    --cache /data1/liuyeqiang/WWW/outputs/research_final_v2/off/perception_cache_train \
    --annotations /data/DATASETS/TRACKING/JDE/VisionTrack/annotations/train.json \
    --checkpoint /data1/liuyeqiang/WWW/outputs/stage2_single_gpu/model_20000.pth \
    --config-file "$REPO/configs/VISION_test.yaml" \
    --output-root "$OUTPUT_ROOT" \
    --queue "$QUEUE" \
    --log-root "$LOG_ROOT" \
    --horizon 8 --view-num 2 --history-limit 80 --poll-seconds 20 \
    --formal-gate-report "$FORMAL"
scheduler_rc=$?
set -e
if [ "$scheduler_rc" -ne 0 ]; then
    echo "full_h8_scheduler_failed=$scheduler_rc"
    exit 9
fi

echo "full_h8_scheduler_complete=$(date -Is)"
set +e
PYTHONPATH="$REPO:$REPO/reproduction_tools:$REPO/third_party/CenterNet2" \
    "$PYTHON" -u "$REPO/reproduction_tools/aftercare_jev_full_h8.py" \
    --runtime-root "$RUNTIME" \
    --formal-gate-report "$FORMAL" \
    --poll-seconds 60
aftercare_rc=$?
set -e
echo "full_h8_aftercare_exit=$(date -Is) rc=$aftercare_rc"
if [ "$aftercare_rc" -ne 0 ]; then
    exit "$aftercare_rc"
fi

set +e
PYTHONPATH="$REPO:$REPO/reproduction_tools:$REPO/third_party/CenterNet2" \
    "$PYTHON" -u "$REPO/reproduction_tools/publish_full_h8_aftercare.py" \
    --runtime-root "$RUNTIME" \
    --repo-root "$REPO" \
    --formal-gate-report "$FORMAL" \
    --output "$PUBLISHED_REPORT" \
    --markdown "$PUBLISHED_MARKDOWN"
publication_rc=$?
set -e
if [ "$publication_rc" -ne 0 ]; then
    echo "full_h8_publication_failed=$publication_rc"
    exit 10
fi

cd "$REPO"
while [ -e .git/index.lock ]; do
    sleep 5
done
git add "$PUBLISHED_REPORT" "$PUBLISHED_MARKDOWN" reproduction_tools/publish_full_h8_aftercare.py
git commit -m "Publish full H8 aftercare summary" || true
git push origin HEAD || {
    git pull --rebase origin jev/counterfactual-rng-isolation-v4-20261006
    git push origin HEAD
}
echo "full_h8_publication_complete=$(date -Is)"
exit 0
