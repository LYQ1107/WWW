#!/usr/bin/env bash
set -eo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
GMT_ROOT=/data3/liuyeqiang/GMT_VisionTrack_repro
source "$GMT_ROOT/tools/miniconda3/etc/profile.d/conda.sh"
conda install -n GMT --override-channels --strict-channel-priority -c nvidia/label/cuda-11.8.0 -c conda-forge cuda-libraries-dev=11.8.0 -y
conda activate GMT
export CUDA_HOME="$CONDA_PREFIX"
export MAX_JOBS=8 TORCH_CUDA_ARCH_LIST=8.0 FORCE_CUDA=1
python -m pip install --no-build-isolation "$GMT_ROOT/tools/detectron2"
python -m pip check
conda env export > "$GMT_ROOT/reports/GMT_environment.yml"
python -m pip freeze > "$GMT_ROOT/reports/GMT_pip_freeze.txt"
python "$GMT_ROOT/tools/environment_smoke.py" > "$GMT_ROOT/logs/environment_smoke.log" 2>&1
