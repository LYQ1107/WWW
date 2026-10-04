# GMT / VisionTrack reproduction review bundle

This repository contains the official GMT source at commit `dfa9ca8e0b8da5c2af89ef3d3ac4f9d991162ebb`, the local compatibility fixes, and the runtime/diagnostic scripts used for the VisionTrack reproduction attempt.

The active execution layout is the measured single-GPU path on GPU0 with
per-device/global batch size 1. The unchanged official video loader asserts
one sample per rank; short wall-clock tests showed that one GPU is faster per
optimizer iteration than the 2/4/9-GPU Gloo paths on this host. This is a
documented hardware/runtime deviation, not a claim of paper-equivalent
throughput. The requested recipe remains 20,000 updates for each of Stage 1
and Stage 2, with the repository AdamW and learning rates. No test-GT tuning,
tracking threshold change, model architecture change, loss change, or
evaluator-formula change is included.

The main model files are `train_net.py`, `test_net.py`, and `gtr/`. The `gmt_runtime_*.py` files are regular source files in this review copy so that the checkout is self-contained; they implement activation checkpointing, progress tracing, distributed runtime handling, and the explicit CPU post-backward gradient averaging candidate. The latter is a runtime deviation adopted only after repeated shared-GPU distributed waits; it must be reviewed before treating any result as a strict paper reproduction.

`reproduction_tools/` is a snapshot of the orchestration, smoke-test,
diagnostic, inference, decision-contract, and evaluation helpers from the
working reproduction directory. It intentionally excludes datasets,
pretrained/final weights, logs, conda environments, Baidu cookies, and
generated outputs. Historical helpers retain their original absolute paths;
the active monitor and inference/training entry points use the current layout.

The four-GPU 100-update diagnostic and independent four-rank communication
gate passed historically. The corrected single-GPU Stage1 run is active and
has saved recoverable checkpoints through `outputs/stage1_single_gpu/model_13000.pth`;
that checkpoint independently passed iteration/scheduler, finiteness,
optimizer-state, and reload checks. Stage2 is waiting for `model_16000.pth`,
so there is still no completed Stage2 checkpoint or real VisionTrack metric
result. The persistent monitor records this state and can resume either stage
from its latest checkpoint. The
cross-view evaluator has no MATLAB installation on the host; the original
MATLAB/MEX sources are retained and the prepared bridge uses GNU Octave, which
remains a documented evaluation-environment deviation.
