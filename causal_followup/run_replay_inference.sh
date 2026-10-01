#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GMT_PY="${GMT_PY:-/data3/liuyeqiang/GMT_VisionTrack_repro/tools/miniconda3/envs/GMT/bin/python}"
MODE="${1:?mode: dump or load}"
JSON="${2:?COCO json path}"
OUT="${3:?output directory}"
GPU="${CUDA_VISIBLE_DEVICES:-4}"
if [[ "$MODE" != dump && "$MODE" != load ]]; then
  echo "mode must be dump or load" >&2
  exit 2
fi
if [[ -e "$OUT" && -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing to reuse non-empty output: $OUT" >&2
  exit 3
fi
mkdir -p "$OUT"
REPLAY_ROOT="${GMT_ASSOC_REPLAY_DIR:-$ROOT/causal_followup/association_replay}"
TRACE_ROOT="${GMT_ASSOC_REPLAY_TRACE_DIR:-$ROOT/causal_followup/association_trace_$MODE}"
TRACE_POST_ROOT="${GMT_ASSOC_REPLAY_TRACE_POST_DIR:-$ROOT/causal_followup/association_trace_post_$MODE}"
export CUDA_VISIBLE_DEVICES="$GPU"
export GMT_AUDIT_DIR="$ROOT/causal_followup"
export GMT_AUDIT_TEST_JSON="$JSON"
export GMT_AUDIT_TEST_IMAGE_ROOT="$ROOT/datasets/VisionTrack/images/test"
export GMT_ASSOC_REPLAY_DIR="$REPLAY_ROOT"
export GMT_ASSOC_REPLAY_TRACE_DIR="$TRACE_ROOT"
export GMT_ASSOC_REPLAY_TRACE=1
export GMT_ASSOC_REPLAY_TRACE_POST_DIR="$TRACE_POST_ROOT"
export GMT_ASSOC_REPLAY_TRACE_POST=1
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
unset GMT_CAUSAL_AUDIT_MODE GMT_CAUSAL_GT_JSON GMT_CAUSAL_EVENT_MANIFEST GMT_CAUSAL_INJECTION_MANIFEST GMT_CAUSAL_RUN
unset CUDA_LAUNCH_BLOCKING GMT_DISTRIBUTED_BACKEND GMT_CPU_COLLECTIVES GMT_POST_BACKWARD_CPU GMT_CPU_DDP_INIT GMT_SYNC_DDP_BUCKETS GMT_CHECKPOINT_BACKBONE
if [[ "$MODE" == dump ]]; then
  export GMT_ASSOC_REPLAY_DUMP=1
  unset GMT_ASSOC_REPLAY_LOAD
else
  export GMT_ASSOC_REPLAY_LOAD=1
  unset GMT_ASSOC_REPLAY_DUMP
fi
export PYTHONPATH="$ROOT/third_party/CenterNet2:${PYTHONPATH:-}"

set +e
"$GMT_PY" "$ROOT/test_net.py" --num-gpus 1 \
  --config-file "$ROOT/configs/VISION_test.yaml" \
  MODEL.WEIGHTS "$ROOT/audit/links/stage2_model_20000.pth" \
  OUTPUT_DIR "$OUT" \
  DATASETS.TEST "('VISION_test_audit',)" \
  INPUT.VIDEO.TEST_LEN 40 MODEL.ASSO_HEAD.WITH_BANK True MODEL.ASSO_HEAD.BANK_SIZE 10 SEED 20260930 \
  >"$OUT/stdout_stderr.log" 2>&1
rc=$?
set -e
printf '{"mode":"%s","return_code":%d,"output":"%s","cache":"%s","trace":"%s"}\n' "$MODE" "$rc" "$OUT" "$REPLAY_ROOT" "$TRACE_ROOT" > "$OUT/run_manifest.json"
exit "$rc"
