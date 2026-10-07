#!/usr/bin/env bash

# Publish only after the live v2 aftercare reaches a terminal line. This
# waiter never signals, migrates, or changes the builder; it only uses a
# temporary Git index to publish reports to the audit branch.

set -u

REPO=/data1/liuyeqiang/WWW_rng_fix_v4
RUNTIME=/home/liuyeqiang/WWW_jev_rng_v4_runtime
AFTERCARE_LOG=$RUNTIME/formal_current_head_video01_v2_aftercare.log
PUBLISH_LOG=$RUNTIME/formal_current_head_video01_v2_publish_waiter.log
AUDIT_BRANCH=jev/audit-diagnostics-20261007

exec > >(tee -a "$PUBLISH_LOG") 2>&1
echo "publish_waiter_started=$(date -Is)"

while true; do
    if grep -q 'video01_v2_aftercare_complete=' "$AFTERCARE_LOG" 2>/dev/null; then
        terminal_reason=aftercare_complete
        break
    fi
    if grep -q 'v2_manifest_binding_failed=' "$AFTERCARE_LOG" 2>/dev/null; then
        terminal_reason=manifest_binding_failed
        break
    fi
    sleep 30
done

echo "publish_waiter_terminal=$(date -Is) reason=$terminal_reason"

audit_tmp_dir=$(mktemp -d /tmp/www_v2_aftercare_publish_20261007_XXXXXX)
audit_index="$audit_tmp_dir/index"
parent_commit=$(git -C "$REPO" rev-parse "refs/heads/$AUDIT_BRANCH")
GIT_INDEX_FILE="$audit_index" git -C "$REPO" read-tree "$parent_commit"

for report in "$REPO"/reports/JEV_RNG_V4/VIDEO01_V2_*.json; do
    if test -f "$report"; then
        relative=${report#"$REPO"/}
        GIT_INDEX_FILE="$audit_index" git -C "$REPO" add -- "$relative"
    fi
done

if GIT_INDEX_FILE="$audit_index" git -C "$REPO" diff --cached --quiet; then
    echo "publish_waiter_no_new_reports=true"
    exit 0
fi

new_tree=$(GIT_INDEX_FILE="$audit_index" git -C "$REPO" write-tree)
new_commit=$(git -C "$REPO" commit-tree "$new_tree" -p "$parent_commit" -m "Publish video01 v2 aftercare reports")
git -C "$REPO" update-ref "refs/heads/$AUDIT_BRANCH" "$new_commit" "$parent_commit"
git -C "$REPO" push origin "$new_commit:refs/heads/$AUDIT_BRANCH"
echo "publish_waiter_pushed_commit=$new_commit"
echo "publish_waiter_complete=$(date -Is)"
