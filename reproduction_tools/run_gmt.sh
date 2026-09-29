#!/usr/bin/env bash
set -eo pipefail
GMT_ROOT=/data3/liuyeqiang/GMT_VisionTrack_repro
if [[ "${1##*/}" == "train_net.py" || "${1##*/}" == "run_observed_training.py" ]]; then
  while [[ -e "$GMT_ROOT/manifests/runtime_selection.pending" ]]; do sleep 2; done
fi
if [[ -f "$GMT_ROOT/manifests/selected_runtime.env" ]]; then
  source "$GMT_ROOT/manifests/selected_runtime.env"
fi
source "$GMT_ROOT/tools/miniconda3/etc/profile.d/conda.sh"
conda activate GMT
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9}"
export CUDA_HOME="$CONDA_PREFIX"
export NCCL_P2P_DISABLE=1
export OMP_NUM_THREADS=1
export GMT_CHECKPOINT_BACKBONE=1
export GMT_DISTRIBUTED_BACKEND=gloo
export CUDA_LAUNCH_BLOCKING="${GMT_CUDA_LAUNCH_BLOCKING:-1}"
export GMT_TRAIN_PROGRESS=1
cd "$GMT_ROOT/code/GMT"
python "$GMT_ROOT/tools/check_training_launch.py" "$$"
exec python "$@"
