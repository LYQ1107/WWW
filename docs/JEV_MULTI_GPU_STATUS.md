# JEV v2 live status

Last refreshed: 2026-10-05 (UTC)

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

The canonical Stage2 checkpoint is fixed to `model_20000.pth` with SHA256
`cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8`.
The historical `model_4500` lock is explicitly proxy-only and has no
official-test authority.

## Not yet formal evidence

- strict same-GPU OFF equality and v2 replay equivalence;
- full-sequence fixed-model GMT association-transformer replay labels;
- strict same-GPU OFF equality and v2 replay equivalence;
- v2 train/val policy suite and capacity report across H=1/8/16/32 and three seeds;
- final calibration/ablation selection;
- canonical `FINAL_SELECTION_LOCK.json`;
- official-test inference and final tracking metrics.

The v1 proxy traces remain running evidence for scheduling only. The cosine
association backend is a protocol backend and must not be reported as the
formal GMT causal result. A real-cache two-view fixed-model adapter smoke
replay passed on 2026-10-04; it is a contract check, not the final causal
metric. The historical cross-GPU OFF report is diagnostic only; the fresh
same-GPU gate is the formal prerequisite.
