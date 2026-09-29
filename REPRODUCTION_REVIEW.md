# GMT / VisionTrack reproduction review bundle

This repository contains the official GMT source at commit `dfa9ca8e0b8da5c2af89ef3d3ac4f9d991162ebb`, the local compatibility fixes, and the runtime/diagnostic scripts used for the VisionTrack reproduction attempt.

The current user-selected execution layout is four GPUs (`0,1,2,3`) with global batch size 4. The unchanged official video loader asserts one sample per rank, so batch 4 is required for four ranks. The requested recipe remains 20,000 updates for each of Stage 1 and Stage 2, with the repository AdamW and learning rates. No test-GT tuning, tracking threshold change, model architecture change, loss change, or evaluator-formula change is included.

The main model files are `train_net.py`, `test_net.py`, and `gtr/`. The `gmt_runtime_*.py` files are regular source files in this review copy so that the checkout is self-contained; they implement activation checkpointing, progress tracing, distributed runtime handling, and the explicit CPU post-backward gradient averaging candidate. The latter is a runtime deviation adopted only after repeated shared-GPU distributed waits; it must be reviewed before treating any result as a strict paper reproduction.

`reproduction_tools/` is a snapshot of the orchestration, smoke-test, diagnostic, inference, and evaluation helpers from the working reproduction directory. It intentionally excludes datasets, pretrained/final weights, logs, conda environments, Baidu cookies, and generated outputs. Most orchestration scripts contain the original absolute working-directory path because they were written for the experiment host; adapt that path before running elsewhere.

The four-GPU 100-update diagnostic subsequently passed with finite losses, and the independent four-rank communication gate passed. A fresh Stage 1 smoke run is now active; there is still no completed Stage 1 or Stage 2 checkpoint and no real VisionTrack metric result. The cross-view evaluator has no MATLAB installation on the host; the original MATLAB/MEX sources are retained and the prepared bridge uses GNU Octave, which is a documented evaluation-environment deviation.
