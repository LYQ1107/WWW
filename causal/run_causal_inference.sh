#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GMT_PY="${GMT_PY:-/data3/liuyeqiang/GMT_VisionTrack_repro/tools/miniconda3/envs/GMT/bin/python}"
MODE="${GMT_CAUSAL_AUDIT_MODE:?set GMT_CAUSAL_AUDIT_MODE=log|correction|sham|injection}"
RUN_NAME="${1:?run name}"; shift
OUT="$ROOT/causal/runs/$RUN_NAME"
mkdir -p "$OUT"
if [[ ! -e "$OUT/datasets" ]]; then ln -s "$ROOT/datasets" "$OUT/datasets"; fi
SUBSET="$ROOT/../GMT_challenge_audit/audit/cache/sanity_subset_test.json"
GT_JSON="${GMT_CAUSAL_GT_JSON:-$SUBSET}"
RAW_NAME="VISIONT18000_13_640_60_objdetection0.525_multithred0.001_NMS0.65_MINLEN50"
RAW_ROOT="$OUT/$RAW_NAME"
if [[ -e "$OUT/raw_predictions" ]]; then echo "existing run: $OUT" >&2; exit 0; fi
if [[ -e "$RAW_ROOT" ]]; then mv "$RAW_ROOT" "$OUT/_stale_raw_$(date +%s)"; fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GMT_AUDIT_DIR="$ROOT"
export GMT_AUDIT_TEST_JSON="$GT_JSON"
export GMT_AUDIT_TEST_IMAGE_ROOT="$ROOT/datasets/VisionTrack/images/test"
export GMT_CAUSAL_GT_JSON="$GT_JSON"
export GMT_CAUSAL_AUDIT_MODE="$MODE"
export GMT_CAUSAL_RUN="$RUN_NAME"
export GMT_CAUSAL_EVENT_MANIFEST="$ROOT/causal/manifests/correction_events.json"
export GMT_CAUSAL_INJECTION_MANIFEST="$ROOT/causal/manifests/injection_events.json"
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
unset CUDA_LAUNCH_BLOCKING GMT_DISTRIBUTED_BACKEND GMT_CPU_COLLECTIVES GMT_POST_BACKWARD_CPU GMT_CPU_DDP_INIT GMT_SYNC_DDP_BUCKETS GMT_CHECKPOINT_BACKBONE
export PYTHONPATH="$ROOT/third_party/CenterNet2:${PYTHONPATH:-}"
START_NS=$(date +%s)
set +e
pushd "$OUT" >/dev/null
"$GMT_PY" "$ROOT/test_net.py" --num-gpus 1 \
  --config-file "$ROOT/configs/VISION_test.yaml" \
  MODEL.WEIGHTS "$ROOT/audit/links/stage2_model_20000.pth" \
  OUTPUT_DIR "$OUT" \
  DATASETS.TEST "('VISION_test_audit',)" \
  INPUT.VIDEO.TEST_LEN 40 MODEL.ASSO_HEAD.WITH_BANK True MODEL.ASSO_HEAD.BANK_SIZE 10 SEED 20260930 \
  "$@" >"$OUT/stdout_stderr.log" 2>&1
RC=$?
popd >/dev/null
set -e
END_NS=$(date +%s)
if [[ -e "$RAW_ROOT" ]]; then mv "$RAW_ROOT" "$OUT/raw_predictions"; fi
if [[ -e "$OUT/raw_predictions" ]]; then
  "$GMT_PY" "$ROOT/audit_tools/convert_mot_results.py" --raw-root "$OUT/raw_predictions" --annotations "$GT_JSON" --output "$OUT/predictions.json" >"$OUT/convert.log" 2>&1 || true
fi
# The released Detectron2 evaluator writes the immutable per-image COCO
# prediction file under the configured inference directory. Prefer it for
# causal runs; the historical raw MOT root is retained when present.
COCO_JSON="$OUT/inference_VISION_test_audit/coco_instances_results.json"
if [[ ! -e "$OUT/predictions.json" && -e "$COCO_JSON" ]]; then
  cp "$COCO_JSON" "$OUT/predictions.json"
fi
"$GMT_PY" - "$OUT" "$RC" "$START_NS" "$END_NS" "$MODE" <<'PY'
import json, sys
from pathlib import Path
out=Path(sys.argv[1]); rc=int(sys.argv[2]); start=int(sys.argv[3]); end=int(sys.argv[4])
payload={"run":out.name,"mode":sys.argv[5],"output":str(out),"return_code":rc,
         "start_epoch":start,"end_epoch":end,"runtime_sec":end-start,
         "raw_predictions":str(out/'raw_predictions'),"raw_predictions_exists":(out/'raw_predictions').exists(),
         "predictions":str(out/'predictions.json'),"predictions_exists":(out/'predictions.json').exists(),
         "decisions":str(out/'decisions.jsonl'),"decisions_exists":(out/'decisions.jsonl').exists()}
(out/'run_manifest.json').write_text(json.dumps(payload,indent=2)+'\n')
print(json.dumps(payload,indent=2))
sys.exit(0 if payload['predictions_exists'] and payload['decisions_exists'] else (rc or 1))
PY
