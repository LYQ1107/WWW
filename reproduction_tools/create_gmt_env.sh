#!/usr/bin/env bash
set -eo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
GMT_ROOT=/data3/liuyeqiang/GMT_VisionTrack_repro
"$GMT_ROOT/tools/miniconda3/bin/conda" create -n GMT --override-channels -c nvidia/label/cuda-11.8.0 -c conda-forge python=3.10 pip 'setuptools<70' wheel ninja cmake 'gcc_linux-64=11' 'gxx_linux-64=11' 'cuda-nvcc=11.8' 'cuda-cudart-dev=11.8' 'cuda-driver-dev=11.8' 'cuda-cccl=11.8' -y
