#!/usr/bin/env bash
set -eo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
GMT_ROOT=/data3/liuyeqiang/GMT_VisionTrack_repro
source "$GMT_ROOT/tools/miniconda3/etc/profile.d/conda.sh"
conda activate GMT
export CUDA_HOME="$CONDA_PREFIX"
export MAX_JOBS=8
export TORCH_CUDA_ARCH_LIST=8.0
export FORCE_CUDA=1
python -m pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r "$GMT_ROOT/reports/requirements-compatible.txt"
python -m pip install cython cython-bbox tensorboard
python -m pip install --no-build-isolation "$GMT_ROOT/tools/detectron2"
python -m pip check
conda env export > "$GMT_ROOT/reports/GMT_environment.yml"
python -m pip freeze > "$GMT_ROOT/reports/GMT_pip_freeze.txt"
