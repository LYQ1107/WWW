# Video01 v2 post-diagnostic audit — 2026-10-07

This report implements the end-of-file requirements in
`www_补充执行指令_20261007.txt`. It is diagnostic evidence, not a paper result.
The existing segmented gate was preserved while the audit was performed; no
second complete video01 builder was started.

## Direct conclusion

The narrow `4108f18` source diff does not justify reusing the old action labels.
On a 20-record same-input replay, state projections are exact (`20/20`), but
outcome/label projections are not (`0/20`). The old artifact also cannot be
reproduced from the recorded 60675 inputs. Therefore all v2 outcomes,
target probabilities, best actions, and sample weights must be regenerated.

The old formal v2 candidate gate is still blocked (`225` capped mismatches,
`100` native-only and `100` replay-only event keys). The independent raw-ReID
repair window removes the score mismatch: candidate scores, IDs/order, chosen
proposal IDs, and legacy OFF actions pass. It still has `8` native-only and `8`
replay-only event keys, so score parity alone is insufficient. The independent
segmented early window (frames 210–216) is stronger evidence: all `70/70`
records are exactly equivalent and both reactivation candidate events pass.
The late 881–890 window was still completing its single-reference comparison
at the report snapshot and is not called PASS.

The raw-ReID/anchor-geometry repair is now committed separately as
`b20d7e2bc26af53095f56bc92179f2e414468da6` on
`jev/video01-reactivation-box-parity-fix-20261007`. The targeted JSON was
generated before that commit and remains diagnostic-only; the commit contains
the tested code, not a claim that the full v2 gate has passed.

## Measured cost and why video01 took hours

The 2,400-record diagnostic measured `3,771.88 s` wall time. It made `102,068`
proposal calls taking `3,562.07 s` in the timing wrapper; this is the dominant
measured cost. It also measured `105,762` steps (`201.82 s`), `8,360` state
clones (`48.11 s`), `100,212` reactivation-context calls (`34.78 s`), and
`6,027` utility rollouts (`4.61 s`). Operation timings are nested, so they are
not summed as independent wall time.

The full formal video01 run started at 04:15:18 UTC, wrote 8,995 records at
11:10:33 UTC, and exited at 11:10:35 UTC: about 6 h 55 min. That is the
requested six-hour evidence, and the machine-readable live snapshot proves
the same PID had already been alive for more than six hours. The derived full
trace workload is approximately 22,514 candidate branches, 393,288 total
steps, 370,764 future proposal calls, and 372,637 reactivation-context calls.

## Current preserved job

PID `15541` is the independent `segmented_gate_20261007_v4` job on GPUs
2,3,5,6,7,8,9, source `1d2711e80ac5fa00806fd9eed30e90cd51df6b30`. At the
snapshot it was finishing the late 881–890 probe: 98/102 records were
complete and its single-reference baseline was still running. The early
210–216 probe already passed exact chunk equivalence and candidate parity.
This job remains untouched.

The formal 8,995-record artifact from `4108f18` is retained for audit only.
Its provenance and checkpoint wrapper passed, but runtime feature parity,
reactivation candidate parity, and three-repeat stability failed; the corrected
closed loop was therefore not run. Full H8 remains paused and noncanonical.

## Minimum authorized next steps

1. Finish the preserved early/late segmented gates and require exact event-key
   coverage, not only candidate-score agreement.
2. Freeze the committed raw-ReID/anchor-box/joint-bank repair together with
   the native event-key/promotion-order fix.
3. Rebuild the complete video01 v2 artifact from that new frozen commit and
   bind every acceptance report to its records SHA, source SHA, input hashes,
   schema, checkpoint, and configuration.
4. Run the three-way closed loop only after all runtime gates pass.
5. Reconsider Full H8 only after a clear GO; no speculative data is eligible
   for policy training or paper metrics.

The complete machine-readable record is
`reports/JEV_RNG_V4/VIDEO01_POST_DIAGNOSTIC_AUDIT_20261007.json`.
