# Reproduction helper snapshot

These files are copied from the working VisionTrack reproduction harness. They include environment setup, data preparation, distributed smoke tests, runtime diagnostics, staged training, inference, prediction conversion, and evaluation helpers.

They are a review snapshot rather than a packaged command-line tool. The
current active checkout is `/data1/liuyeqiang/WWW`; the active data root is
`/data/DATASETS/TRACKING/JDE/VisionTrack`, and the active GMT environment is
`/home/liuyeqiang/anaconda3/envs/GMT`.

The older copied orchestration files that reference `/data3/.../code/GMT`
are historical review artifacts and must not be used for the active run. The
current long-run entry points are `monitor_research.py`,
`chain_research_after_stage2.py`, `run_formal_inference.py`,
`build_jev_counterfactual_dataset.py`, `replay_jev_policy.py`, `train_jev.py`,
`jev_stress_tests.py`, and the `test_jev_*.py` contract tests. The post-Stage2
supervisor writes isolated artifacts under `outputs/research_pipeline/` and
can resume from completed manifests. No dataset, checkpoint, log, generated
output, or authentication state is included in a clean checkout.
