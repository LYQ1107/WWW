#!/usr/bin/env bash

# Run the old-vs-corrected dataset audit once the corrected v6/v7 artifacts
# are complete. This is read-only with respect to the builders.

set -u

REPO=/data1/liuyeqiang/WWW_rng_fix_v4
RUNTIME=/home/liuyeqiang/WWW_jev_rng_v4_runtime
NEW_ROOT=$RUNTIME/small_h8_rng_controlled_v4_current_head_75b0aea
OUT=$REPO/reports/JEV_RNG_V4/OLD_VS_RNG_FIXED_DATASET_AUDIT_VIDEO06_VIDEO07_CURRENT.json
LOG=$RUNTIME/current_head_v6_v7_old_vs_new_audit_corrected_waiter.log

exec > >(tee -a "$LOG") 2>&1
echo "watcher_started=$(date -Is)"
while ! jq -e '.status == "COMPLETE" and (.records | tonumber) > 0' \
    "$NEW_ROOT/video_06/manifest.json" >/dev/null 2>&1 || \
      ! jq -e '.status == "COMPLETE" and (.records | tonumber) > 0' \
    "$NEW_ROOT/video_07/manifest.json" >/dev/null 2>&1; do
    sleep 30
done
echo "manifests_ready=$(date -Is)"

set +e
PYTHONPATH="$REPO:$REPO/reproduction_tools:$REPO/third_party/CenterNet2" \
    /home/liuyeqiang/anaconda3/envs/GMT/bin/python -u \
    "$REPO/reproduction_tools/audit_jev_rng_v4_datasets.py" \
    --old-runtime /home/liuyeqiang/WWW_jev_full_h8_runtime \
    --new-runtime "$NEW_ROOT" \
    --output "$OUT" --video-ids 6 7
RC=$?
set -e

cd "$REPO"
while [ -e .git/index.lock ]; do sleep 5; done
git add "$OUT"
git commit -m "Record corrected v6 v7 old-new audit" || true
git push origin HEAD || {
    git pull --rebase origin jev/counterfactual-rng-isolation-v4-20261006
    git push origin HEAD
}
echo "audit_waiter_exit=$RC"
exit "$RC"
