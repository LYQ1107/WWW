# Video01 v2 diagnostic re-audit — 2026-10-07

This is the re-audit requested by the final “运行诊断后的补充执行要求” in
`www_补充执行指令_20261007.txt`. It is diagnostic evidence only; it is not a
paper result and does not authorize Full H8.

## Current state

The original corrected video01 v2 job was preserved and was not restarted. It
has now completed and exited. No second complete video01 builder was started,
and the re-audit found no active WWW/JEV/H8 process. The completed artifact is
still bound to:

- records: `8995`, SHA-256
  `1da2ed9e7eec43c7cf139754d873f78a299c344ff74d65c6866b92f18f3da79`;
- source: `4108f18f5040432f68d55872e81e9f76e9acd08f`;
- checkpoint: `cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8`;
- question counts: MATCH `4524`, MEMORY `4297`, REACTIVATION `174`;
- provenance/schema: PASS; runtime feature parity: FAIL.

The full machine-readable re-audit is
`reports/JEV_RNG_V4/VIDEO01_DIAGNOSTIC_REAUDIT_20261007.json`.

## 1. Does the 4108f18 change require a complete label recomputation?

The direct commit diff is narrow: one file, `11` insertions and `4`
deletions, recording `model.thred_bank` as `bank_threshold` and using it for
reactivation state features. The trace, config, checkpoint, annotations, and
cache-index hashes were equal between the old and new bounded comparisons.

That narrow source diff does not make the old labels reusable. An independent
20-record same-input comparison found:

| Check | Result |
|---|---:|
| event key/order | exact |
| state projection | 20/20 exact |
| outcome/label projection | 0/20 exact |
| repeated current sample | byte-identical |
| 60675 historical replay vs current sample | exact |
| 60675 historical replay vs old artifact | not exact |

The old artifact’s hidden generation provenance therefore cannot be recovered
from its manifest. Reusing `action_outcomes`, `horizon_outcomes`,
`target_probs`, `best_actions`, or sample weights is blocked. A new canonical
v2 build must regenerate all of them; the bounded comparison is not itself a
formal acceptance run.

## 2. Fresh candidate parity result

A fresh replay was run against the completed v2 records only through frame 220;
this did not start a full builder. It compared `5` native reactivation events
with `5` replay events. The result is still `FAIL`:

- two native-only and two replay-only event keys remain;
- on overlapping events, candidate IDs, selected proposal IDs, legacy OFF
  actions, and bank thresholds are exact;
- candidate scores fail at `(214,1,0)`, `(216,1,0)`, and `(217,1,0)`, with
  maximum absolute errors `0.0701287`, `0.0437699`, and `0.0193566`;
- the native and replay bank threshold is `0.4` at those failures.

This specifically rules out treating the `bank_threshold` feature change as a
complete candidate-parity repair. The remaining evidence points to raw-ReID /
anchor-bank semantics and event-key/promotion timing, not to a simple threshold
value mismatch. The current candidate parity gate is BLOCKED.

## 3. Measured cost and six-hour evidence

An independent current-commit 20-record probe took `22.659 s`, with `857`
proposal calls (`8.519 s`) and `908` step calls (`8.044 s`). State cloning took
only `0.018 s` over `51` calls. The larger 2,400-record probe measured
`3,771.883 s`; proposal/model evaluation took `3,562.068 s` over `102,068`
calls, while step timing was `201.817 s` over `105,762` calls. These timings
are nested instrumentation and must not be summed as independent wall time.

The completed full v2 run started at `04:15:18 UTC`, produced all `8,995`
records at `11:10:33 UTC`, and exited at `11:10:35 UTC`: `24,917 s`, about
`6 h 55 min`, at about `21.67 records/min`. The derived workload was roughly
`22,514` candidate branches, `393,288` total steps, `370,764` future proposal
calls, and `372,637` reactivation-context preparations. This is direct timing
evidence, not an ETA extrapolated from GPU utilization.

Conclusion: the dominant measured bottleneck is repeated proposal/model
evaluation multiplied by a full future replay for each legal candidate. State
cloning and utility scoring are not the primary measured bottlenecks. The run
was expensive, but the evidence does not indicate an I/O deadlock.

## 4. Recovery and authorized next step

The repository now contains a future-build recovery protocol in
`reproduction_tools/jev_v2_progress_recovery.py`: atomic progress fields for
frame/key range, completed/total events, branch count, elapsed time, and
last-update time; plus checkpoints containing mutable state, explicit
trajectory RNG state/seed, provenance, and a state signature. That protocol was
not injected into the completed v2 process, so no restart was performed merely
to add progress reporting.

The minimum next step is to freeze the tested raw-ReID/anchor/joint-bank and
native event-key/promotion-order repair, pass bounded early and late exact
equivalence plus candidate parity, then build one new complete video01 v2
artifact with every label regenerated. Until then:

- video01 v2 runtime gate: **NO_GO**;
- candidate-conditioned follow-up: **blocked pending semantic repair**;
- Full H8: **paused and noncanonical**;
- no paper metrics are authorized from this audit.
