# JEV v2 live status

Last refreshed: 2026-10-04 (UTC)

## Code branch

- Worktree: `/data1/liuyeqiang/WWW_jev_v2`
- Branch: `jev/reviewer-proof-v2`
- Base: canonical local `ba3dbdad6f8da0542a24607de9c1d9586c96982e`
- Canonical Stage2 worktree is separate and remains untouched.

## Completed contracts

- typed threshold/MLP baseline family;
- pre-action identity utility and uninformative-memory weighting fixes;
- raw trajectory feature fields and feature-audit tool;
- capacity-match tool;
- constrained global Hungarian REASSOCIATE path;
- sequence policy split and final-selection lock tools;
- val-only temperature calibration;
- frozen perception cache writer/reader;
- mutable-association v2 contract engine and label-builder scaffold;
- formal GMT association-transformer builder backend;
- CPU invariant tests for all of the above.

## Not yet formal evidence

- full cache generation/checksum audit from the fixed GMT proxy checkpoint;
- full-sequence fixed-model GMT association-transformer replay labels;
- v2 train/val policy suite and capacity report;
- final calibration/ablation selection;
- `FINAL_SELECTION_LOCK.json`;
- official-test inference and final tracking metrics.

The v1 proxy traces remain running evidence for scheduling only. The cosine
association backend is a protocol backend and must not be reported as the
formal GMT causal result. A real-cache two-view fixed-model adapter smoke
replay passed on 2026-10-04; it is a contract check, not the final causal
metric.
