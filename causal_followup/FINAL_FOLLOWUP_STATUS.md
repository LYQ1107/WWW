# GMT causal state-recovery follow-up — final status

The follow-up is complete. It did not retrain GMT, start a formal Stage1 or
Stage2 run, modify the released checkpoint, or delete any first-round audit
artifact. All work was performed on branch `jev-gmt-causal-prototype` in
`/data3/liuyeqiang/JEV_GMT`.

## Prerequisite gates

- `REPLAY_FIDELITY.json`: **PASS**. The fixed observation cache has 7,130
  records across `00001garden`, `00003garden`, and `00005garden`. Cache counts
  match the audit JSON and the dump/load pre-filter tracking traces have exact
  track-ID and detection-count equality for every record.
- `SNAPSHOT_ROUNDTRIP.json`: **PASS**. All three scenes have exact serialized
  state digests, exact ten-frame continuation traces, and exact CPU/CUDA/Python
  RNG restoration.
- Fixed stream: seed `20260930`, `TEST_LEN=40`, `WITH_BANK=True`,
  `BANK_SIZE=10`; no GT is passed into the tracker/model.

## Isolated event audit

Every frozen event was replayed from the same pre-event snapshot. Multiple
events sharing a frame/view were separately branched from that shared state.

- C1b: 71/71 events, 5,751 rows. C1b-B had 66 estimable events at each target
  horizon. C1b-C had 60 estimable events because repair collisions were
  explicitly skipped. Assignment-only, quarantine, and oracle repair did not
  produce a statistically significant reduction at all of +5, +10, and +20.
- C2b: 256/256 events, 10,496 rows. The +5 pooled error-rate difference was
  **+0.074 percentage points**, with paired bootstrap 95% CI
  **[−0.011, +0.139] percentage points**. At +10 it was +0.001 pp
  [−0.048, +0.052], and at +20 −0.047 pp [−0.121, +0.027].
- Skip reasons are preserved in the CSVs and statistics: target occupancy,
  repair collision, and scene-end branches are not silently counted as
  effects.

The full paired bootstrap and scene-stratified results are in
`causal_followup/results/FOLLOWUP_STATISTICS.json`; the readable reports are
`causal_followup/reports/C1b_STATE_REPAIR.md` and
`causal_followup/reports/C2b_ISOLATED_INJECTION.md`.

## Final causal decision

**`STOP_NO_STRONG_CAUSAL_GATE`**.

The isolated single-error injection has a small positive direction at +5, but
its confidence interval crosses zero. Neither quarantine nor oracle state
repair has a confidence interval excluding zero across the required +5/+10/+20
horizons. The scene-stratified direction requirement therefore also fails.

The causal challenge does not support proceeding to a JEV-GMT prototype. No
JEV source audit, design, training, or prototype code was started. The earlier
first-round C1/C2 artifacts remain under `causal/` unchanged and are retained
as historical evidence.
