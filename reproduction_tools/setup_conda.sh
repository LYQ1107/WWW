#!/usr/bin/env bash
set -euo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
GMT_ROOT=/data3/liuyeqiang/GMT_VisionTrack_repro
curl -fL --connect-timeout 15 --max-time 300 --retry 2 -o "$GMT_ROOT/downloads/Miniconda3-latest-Linux-x86_64.sh" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sha256sum "$GMT_ROOT/downloads/Miniconda3-latest-Linux-x86_64.sh" > "$GMT_ROOT/manifests/miniconda.sha256"
bash "$GMT_ROOT/downloads/Miniconda3-latest-Linux-x86_64.sh" -b -p "$GMT_ROOT/tools/miniconda3"
"$GMT_ROOT/tools/miniconda3/bin/conda" create -n GMT --override-channels -c conda-forge -c nvidia/label/cuda-11.8.0 python=3.10 pip 'setuptools<70' wheel ninja cmake 'gcc_linux-64=11' 'gxx_linux-64=11' cuda-toolkit=11.8.0 -y
