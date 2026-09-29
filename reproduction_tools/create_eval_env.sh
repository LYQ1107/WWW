#!/usr/bin/env bash
set -eo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
GMT_ROOT=/data3/liuyeqiang/GMT_VisionTrack_repro
"$GMT_ROOT/tools/miniconda3/bin/conda" create -n GMT_eval --override-channels -c conda-forge python=3.10 octave 'numpy=1.24.1' 'scipy=1.10.1' 'pandas=2.0.3' pip 'gcc_linux-64=11' 'gxx_linux-64=11' -y
