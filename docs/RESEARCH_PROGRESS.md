# WWW research progress

Updated: 2026-10-05 UTC

## Verified

- GMT Stage1 and Stage2 completed; canonical Stage2 `model_20000.pth` passed
  reload, finiteness, optimizer-state, and iteration validation.
- Canonical OFF train/test/native inference artifacts are complete and tied to
  that digest. Cache and trace alignment checks passed.
- Canonical OFF TEST TrackEval/CVIDF1/CVMA reporting is recorded in
  `docs/STAGE2_OFF_TRACKING_RESULTS.md`; it is not a policy-selection input.
- Formal v2 code includes filtered/serialized shards, immutable perception
  payload prefetch, bounded adapter tensor cache, and branch cloning that
  shares frozen perception but copies mutable tracker state.
- Selection-gate, cached-perception, and GMT-association adapter invariants pass.

## Running / retained

- Original formal train shard 0 and manually started low-memory train shard 1
  remain running. A pre-lock TEST shard remains as diagnostic evidence only.
- A fresh same-GPU native/traced OFF equality run is in progress on visible GPU0.
- Existing incomplete/OOM logs and pre-lock artifacts are retained; no valid
  formal shard is overwritten or deleted.

## Protocol corrections now enforced

- Final lock creation accepts only canonical `model_20000.pth`.
- The old `model_4500` lock is marked proxy screening only.
- Policy split construction rejects official TEST input before the lock.
- New TEST counterfactual processes fail closed until a canonical lock and
  validation-only selection manifest exist.
- Cross-GPU OFF comparison is marked diagnostic-only; same-GPU exact equality is
  the formal prerequisite.
- Automatic protocol runner added for H=1/8/16/32, all threshold/nonlinear
  threshold and generic-MLP controls, three seeds, equal supervision, and
  calibration for every candidate.

## Open gates

1. Complete TRAIN formal v2 shards and H=1/8/16/32 TRAIN datasets.
2. Finish the selection protocol, feature audit, same-score/different-state and
   reviewer-control reports.
3. Create and validate the canonical selection lock.
4. Only after that lock, generate/read official TEST counterfactuals and run
   Oracle, baseline, JEV, TrackEval, cross-view evaluation, and final report.

The research task remains **IN PROGRESS**; no final GO/NO-GO or causal gain
claim is made.
