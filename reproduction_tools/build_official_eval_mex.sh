#!/usr/bin/env bash
set -eo pipefail
GMT_ROOT=/data3/liuyeqiang/GMT_VisionTrack_repro
source "$GMT_ROOT/tools/miniconda3/etc/profile.d/conda.sh"
conda activate GMT_eval
mkdir -p "$GMT_ROOT/tools/octave_mex"
octave --no-gui --quiet --eval 'disp(version)'
for unit in MinCostMatching clearMOTMex costBlockMex; do
 mkoctfile --mex -std=c++11 -fopenmp -I"$GMT_ROOT/tools/octave_mex/include" "$GMT_ROOT/code/GMT/MOTChallengeEvalKit_cv_test/matlab_devkit/utils/$unit.cpp" -o "$GMT_ROOT/tools/octave_mex/$unit.mex"
done
