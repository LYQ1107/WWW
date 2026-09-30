#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
JSON="$ROOT/audit/cache/sanity_subset_test.json"
IMG="$ROOT/datasets/VisionTrack/images/test"
export GMT_AUDIT_TEST_JSON="$JSON" GMT_AUDIT_TEST_IMAGE_ROOT="$IMG" CUDA_VISIBLE_DEVICES=0

# A0 fixed-seed no-jitter comparison (only this seed is required by the spec).
GMT_AUDIT_DISABLE_TEST_JITTER=1 ./audit_tools/run_audit_inference.sh A0_seed_20260930_nojitter \
  SEED 20260930 INPUT.VIDEO.TEST_LEN 40 MODEL.ASSO_HEAD.WITH_BANK True MODEL.ASSO_HEAD.BANK_SIZE 10
unset GMT_AUDIT_DISABLE_TEST_JITTER

# A2/A3 are the predeclared matrix on the automatic three-scene subset.
python3 audit_tools/run_history_sweep.py --subset --matrix all

# A5 uses one fixed-seed, bank-on, TEST_LEN=40 representation dump.
export GMT_AUDIT_DUMP_FEATURES=1
./audit_tools/run_audit_inference.sh A5_feature_dump SEED 20260930 INPUT.VIDEO.TEST_LEN 40 MODEL.ASSO_HEAD.WITH_BANK True MODEL.ASSO_HEAD.BANK_SIZE 10
unset GMT_AUDIT_DUMP_FEATURES

echo "remaining audit inference matrix finished"
