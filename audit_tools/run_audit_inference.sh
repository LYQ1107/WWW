#!/usr/bin/env bash
set -euo pipefail

# Explicitly audit-only.  This wrapper invokes test_net.py (evaluation mode)
# and never imports train_net.py or creates a training output.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GMT_PY="${GMT_PY:-/data3/liuyeqiang/GMT_VisionTrack_repro/tools/miniconda3/envs/GMT/bin/python}"
RUN_NAME="$1"; shift
OUT="$ROOT/audit/runs/$RUN_NAME"
mkdir -p "$OUT"
RAW_ROOT="$ROOT/VISIONT18000_13_640_60_objdetection0.525_multithred0.001_NMS0.65_MINLEN50"
if [[ -e "$OUT/raw_predictions" ]]; then
  echo "existing run: $OUT" >&2
  exit 0
fi
if [[ -e "$RAW_ROOT" ]]; then
  # The repository's inference loop writes this historical fixed root. Keep it
  # as an immutable artifact before starting another audit condition.
  stale="$ROOT/audit/runs/_stale_$(date +%s)"
  mkdir -p "$stale"
  mv "$RAW_ROOT" "$stale/raw_predictions"
fi
export CUDA_VISIBLE_DEVICES=0
export GMT_AUDIT_DIR="$ROOT/audit"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
unset CUDA_LAUNCH_BLOCKING GMT_DISTRIBUTED_BACKEND GMT_CPU_COLLECTIVES GMT_POST_BACKWARD_CPU GMT_CPU_DDP_INIT GMT_SYNC_DDP_BUCKETS GMT_CHECKPOINT_BACKBONE
export PYTHONPATH="$ROOT/third_party/CenterNet2:${PYTHONPATH:-}"
START_NS=$(date +%s)
set +e
"$GMT_PY" "$ROOT/test_net.py" --num-gpus 1 \
  --config-file "$ROOT/configs/VISION_test.yaml" \
  MODEL.WEIGHTS "$ROOT/audit/links/stage2_model_20000.pth" \
  OUTPUT_DIR "$OUT" \
  DATASETS.TEST "('VISION_test_audit',)" \
  "$@" >"$OUT/stdout_stderr.log" 2>&1
RC=$?
set -e
END_NS=$(date +%s)
if [[ -e "$RAW_ROOT" ]]; then
  mv "$RAW_ROOT" "$OUT/raw_predictions"
fi
if [[ -e "$OUT/raw_predictions" && -n "${GMT_AUDIT_TEST_JSON:-}" ]]; then
  "$GMT_PY" "$ROOT/audit_tools/convert_mot_results.py" \
    --raw-root "$OUT/raw_predictions" --annotations "$GMT_AUDIT_TEST_JSON" \
    --output "$OUT/predictions.json" >"$OUT/convert.log" 2>&1
  "$GMT_PY" "$ROOT/audit_tools/evaluate_audit_run.py" \
    --pred-json "$OUT/predictions.json" --annotations "$GMT_AUDIT_TEST_JSON" \
    --gt-root "$ROOT/../GMT_paper_repro_clean/TrackEval/data/gt/mot_challenge/vision-train" \
    --output "$OUT/evaluation" >"$OUT/evaluation.log" 2>&1
fi
python3 - "$OUT" "$RC" "$START_NS" "$END_NS" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1]); rc=int(sys.argv[2]); start=int(sys.argv[3]); end=int(sys.argv[4])
pred=out/'raw_predictions'
manifest={'run':out.name,'output':str(out),'return_code':rc,'start_epoch':start,'end_epoch':end,'runtime_sec':end-start,'raw_predictions':str(pred),'raw_predictions_exists':pred.exists()}
(out/'run_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest,indent=2))
sys.exit(0 if pred.exists() else rc)
PY
