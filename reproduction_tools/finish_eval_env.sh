#!/usr/bin/env bash
set -eo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
GMT_ROOT=/data3/liuyeqiang/GMT_VisionTrack_repro
"$GMT_ROOT/tools/miniconda3/bin/conda" install -n GMT_eval --override-channels -c conda-forge python=3.9 pandas=1.5.3 -y
