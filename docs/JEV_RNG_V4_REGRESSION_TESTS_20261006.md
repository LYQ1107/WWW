# Current-head regression tests — 2026-10-06

The GMT environment now has the pytest runner required for source-level
regression tests. The current-head test batch completed with **3 passed, 0
failed**.

It covered the intra-video chunk contract, counterfactual RNG isolation,
reactivation semantics, JEV training contracts, and runtime contracts. The
only output warning was an existing Pillow `Image.LINEAR` deprecation warning
inside third-party Detectron2 code.

This is source-level evidence only. It does not replace the live formal
single-vs-chunk equivalence gate, corrected video builders, runtime parity, or
the final tracking comparison.
