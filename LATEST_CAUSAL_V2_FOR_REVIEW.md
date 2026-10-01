# Latest Causal Audit v2 state for review

This is the independent post-hoc **Causal Audit v2** requested after the
published causal follow-up. It adds no training, no JEV-GMT implementation,
and no VisionTrack test run. The old `causal/`, `causal_followup/`, and
`LATEST_CAUSAL_FOLLOWUP_FOR_REVIEW.md` artifacts are unchanged.

## Decision

`STATE_CAUSAL_GO = NO-GO` remains the previous pre-registered result.

`MECHANISM_VALIDATED = NO` for Causal Audit v2.

C1c complete state repair passes its recovery gate, but C2c isolated target
injection does not pass the propagation gate. Therefore the state-recovery /
contamination motivation is stopped. A separate decision-formulation audit may
be planned later without training; this publication does not start it.

## Frozen design and endpoint

- C2c uses the exact 256-event `causal_followup/manifests/C2b_events.json`.
  Events were not reselected.
- C1c uses the exact 71-event `causal_followup/manifests/C1b_events.json`.
- Horizons are +1, +2, +5, +10, and +20 frames.
- The resampling unit is the event. Paired bootstrap uses 10,000 resamples,
  seed `20260930`.
- The primary endpoint is the treated GT identity only. A horizon with no
  matched treated GT observation is `target_not_observable`, never zero error.
- C2b has frozen `p_baseline` and `p_wrong` but no separate `p_correct`; for
  C2c only, the frozen baseline identity is therefore `p_correct := p_baseline`.
  No majority label was recomputed.
- The old global frame error is retained as a secondary metric.

The implementation audit is in
[`causal_v2/CODE_AUDIT_V2.md`](causal_v2/CODE_AUDIT_V2.md). It documents why
the old `_frame_error` was global, why C1b-C did not repair GMT's sliding
history, and the `history -> ids -> unique_ids -> id_inds -> traj_score` path.
V2 uses copied internal history overrides; public/emitted `instances[].track_ids`
are not rewritten. Full repair synchronizes the sliding history, re-ID rows,
counts, `gt_history`, candidate eligibility, and relevant `old_reids` proxies,
with explicit `repair_conflict` and `C1c-bankoff` outcomes.

## Micro and state-repair sanity

[`causal_v2/MICRO_TRACE.md`](causal_v2/MICRO_TRACE.md) is the first 10 C2b
events in lexicographic event-key order. The target-centric micro check passes:
all seven executable injections show SHAM at the frozen identity and treatment
at `p_wrong` at the event frame, and future rows track only the treated GT.
Three events are explicitly skipped as `baseline_anchor_mismatch` because the
replay's current decision did not equal the frozen baseline; they were not
forced: `00001garden|179|1|11`, `00001garden|196|1|7`, and
`00001garden|208|1|3`.

[`causal_v2/STATE_REPAIR_SANITY.md`](causal_v2/STATE_REPAIR_SANITY.md) and
`STATE_REPAIR_SANITY_FULL.md` show the five-event sanity and the full-run
assertions. In the five-event sanity, moved wrong-history rows were 1, 20, 20,
20, and 0. Every successful branch reports public history unchanged, unique
frame/view IDs, and `id_count_dict == len(id_reid_dict)` for both repaired IDs;
no relevant `old_reids` entry was present in those five branches.

## C1c — full internal state repair

There are 71 frozen events. Sixty-three treatment branches were executable;
eight were explicit `repair_conflict` skips. The treatment-side accounting is
`repair_conflict`: 8 events / 160 rows, `target_not_observable`: 31 events /
246 rows, and `scene_end`: 1 event / 19 rows. The paired estimable event count
varies by horizon because observability is evaluated separately.

Primary treated-identity error (`treatment - SHAM`; pooled delta is the
count-weighted difference):

| Horizon | n | SHAM | Repair | Event delta | Pooled delta | 95% CI |
|---:|---:|---:|---:|---:|---:|---|
| +1 | 53 | 0.720588235294 | 0.632352941176 | -0.0660377358491 | -0.0882352941176 | [-0.122641509434, -0.0188679245283] |
| +2 | 48 | 0.761904761905 | 0.68253968254 | -0.0520833333333 | -0.0793650793651 | [-0.09375, -0.0104166666667] |
| +5 | 50 | 0.784615384615 | 0.676923076923 | -0.1 | -0.107692307692 | [-0.18, -0.03] |
| +10 | 47 | 0.688524590164 | 0.540983606557 | -0.13829787234 | -0.147540983607 | [-0.234042553191, -0.0531914893617] |
| +20 | 49 | 0.703125 | 0.59375 | -0.102040816327 | -0.109375 | [-0.183673469388, -0.030612244898] |

Target correct-rate deltas are respectively `+0.0660377358491`,
`+0.0520833333333`, `+0.1`, `+0.13829787234`, and `+0.102040816327`, with
95% CIs `[0.0188679245283, 0.122641509434]`,
`[0.0104166666667, 0.09375]`, `[0.03, 0.18]`,
`[0.0531914893617, 0.234042553191]`, and
`[0.0408163265306, 0.183673469388]`.

Target wrong-injected-ID deltas are `-0.0566037735849`, `-0.104166666667`,
`-0.07`, `-0.0851063829787`, and `-0.0918367346939`, with CIs
`[-0.122641509434, 0.00943396226415]`, `[-0.177083333333, -0.0416666666667]`,
`[-0.13, -0.02]`, `[-0.170212765957, 0]`, and
`[-0.163265306122, -0.030612244898]`.

Target fragment/other-ID deltas are `-0.00943396226415`, `+0.0520833333333`,
`-0.03`, `-0.0531914893617`, and `-0.0102040816327`, with CIs
`[-0.0754716981132, 0.0566037735849]`, `[0, 0.11484375]`, `[-0.1, 0.02]`,
`[-0.13829787234, 0.0106382978723]`, and `[-0.0612244897959, 0.030612244898]`.

For completeness, the SHAM → repair rates are: correct
`0.279411764706→0.367647058824`, `0.238095238095→0.31746031746`,
`0.215384615385→0.323076923077`, `0.311475409836→0.459016393443`,
`0.296875→0.40625`; wrong-ID `0.161764705882→0.0735294117647`,
`0.285714285714→0.15873015873`, `0.230769230769→0.138461538462`,
`0.180327868852→0.0655737704918`, `0.140625→0.03125`; fragment/other-ID
`0.558823529412→0.558823529412`, `0.47619047619→0.52380952381`,
`0.553846153846→0.538461538462`, `0.508196721311→0.475409836066`,
`0.5625→0.5625` (horizon order +1, +2, +5, +10, +20).

Target cross-view same-ID consistency (SHAM → repair, delta, 95% CI) is:
`+1: 0.333333333333→0.533333333333, +0.2, [-0.0666666666667, 0.466666666667]`;
`+2: 0.333333333333→0.533333333333, +0.2, [-0.133333333333, 0.533333333333]`;
`+5: 0.2→0.4, +0.2, [-0.0666666666667, 0.466666666667]`;
`+10: 0.285714285714→0.5, +0.214285714286, [-0.0714285714286, 0.5]`;
`+20: 0.4→0.533333333333, +0.133333333333, [-0.133333333333, 0.4]`.

The secondary global-frame-error deltas are `-0.00253776301913`,
`-0.00134062621966`, `-0.00297578019352`, `-0.00458156091221`, and
`-0.0026703515045`; their CIs are respectively
`[-0.00923294595616, 0.00460337607814]`,
`[-0.00963967293907, 0.00751487513383]`,
`[-0.00862217233185, 0.00210228839261]`,
`[-0.0100754608295, 0.000189996159754]`, and
`[-0.00739731757548, 0.00143174380552]`.

Scene-level target-error deltas (scene order `00001garden`, `00003garden`,
`00005garden`) are:

| Horizon | Scene deltas (n events) |
|---:|---|
| +1 | -0.222222222222 (9), 0 (13), -0.0483870967742 (31) |
| +2 | -0.25 (8), 0 (11), -0.0172413793103 (29) |
| +5 | -0.25 (8), -0.142857142857 (14), -0.0357142857143 (28) |
| +10 | -0.25 (8), -0.142857142857 (14), -0.1 (25) |
| +20 | -0.1875 (8), -0.0769230769231 (13), -0.0892857142857 (28) |

The full machine-readable result is
[`causal_v2/results/C1c_target_repair.csv`](causal_v2/results/C1c_target_repair.csv);
the complete metric tables are in
[`causal_v2/reports/C1c_TARGET_STATE_REPAIR.md`](causal_v2/reports/C1c_TARGET_STATE_REPAIR.md).

## C2c — isolated single-error injection

There are 256 frozen events. Treatment branches were 159 executable and 97
explicitly non-executable at the event frame: 96 `baseline_anchor_mismatch`
events and one `current_target_occupied` event. Eighteen events had at least
one `target_not_observable` horizon. Treatment-side accounting is
`baseline_anchor_mismatch`: 96 events / 1,920 rows, `current_target_occupied`:
1 event / 20 rows, and `target_not_observable`: 18 events / 155 rows.

Primary treated-identity error (`injection - SHAM`; pooled delta is the
count-weighted difference):

| Horizon | n | SHAM | Injection | Event delta | Pooled delta | 95% CI |
|---:|---:|---:|---:|---:|---:|---|
| +1 | 155 | 0.0445544554455 | 0.0544554455446 | 0.0129032258065 | 0.00990099009901 | [0, 0.0322580645161] |
| +2 | 155 | 0.0295566502463 | 0.0394088669951 | 0.0129032258065 | 0.00985221674877 | [0, 0.0322580645161] |
| +5 | 156 | 0.0445544554455 | 0.0594059405941 | 0.0160256410256 | 0.0148514851485 | [0, 0.0352564102564] |
| +10 | 148 | 0.0364583333333 | 0.046875 | 0.0135135135135 | 0.0104166666667 | [0, 0.0337837837838] |
| +20 | 149 | 0.0510204081633 | 0.0663265306122 | 0.0167785234899 | 0.015306122449 | [0, 0.0402684563758] |

Target correct-rate deltas are `-0.0129032258065`, `-0.0129032258065`,
`-0.0160256410256`, `-0.0135135135135`, and `-0.0167785234899`, with CIs
`[-0.0322580645161, 0]`, `[-0.0322580645161, 0]`, `[-0.0384615384615, 0]`,
`[-0.0337837837838, 0]`, and `[-0.0369127516779, 0]`.

Target wrong-injected-ID deltas are `+0.0161290322581`, `+0.0129032258065`,
`+0.0128205128205`, `+0.0135135135135`, and `+0.0134228187919`, with CIs
`[0, 0.0387096774194]`, `[0, 0.0322580645161]`, `[0, 0.0320512820513]`,
`[0, 0.0337837837838]`, and `[0, 0.0335570469799]`.

Target fragment/other-ID deltas are `-0.00322580645161`, `0`,
`+0.00320512820513`, `0`, and `+0.00335570469799`, with CIs
`[-0.00967741935484, 0]`, `[0, 0]`, `[0, 0.00961538461538]`, `[0, 0]`, and
`[0, 0.010067114094]`.

For completeness, the SHAM → injection rates are: correct
`0.955445544554→0.945544554455`, `0.970443349754→0.960591133005`,
`0.955445544554→0.940594059406`, `0.963541666667→0.953125`,
`0.948979591837→0.933673469388`; wrong-ID
`0→0.0148514851485`, `0→0.00985221674877`, `0→0.00990099009901`,
`0.00520833333333→0.015625`, `0.00510204081633→0.015306122449`; and
fragment/other-ID `0.0445544554455→0.039603960396`,
`0.0295566502463→0.0295566502463`, `0.0445544554455→0.049504950495`,
`0.03125→0.03125`, `0.0459183673469→0.0510204081633` (horizon order
+1, +2, +5, +10, +20).

Target cross-view same-ID consistency (SHAM → injection, delta, 95% CI) is:
`+1: 0.872340425532→0.872340425532, 0, [0, 0]`;
`+2: 0.916666666667→0.916666666667, 0, [0, 0]`;
`+5: 0.869565217391→0.847826086957, -0.0217391304348, [-0.0652173913043, 0]`;
`+10: 0.909090909091→0.909090909091, 0, [0, 0]`;
`+20: 0.872340425532→0.851063829787, -0.0212765957447, [-0.063829787234, 0]`.

The secondary global-frame-error deltas are `+0.0000470354243939`,
`+0.000251572327044`, `+0.000278798769365`, `+0.000323166032764`, and
`-0.0000116561751782`; their CIs are respectively
`[-0.00183438155136, 0.0014715368489]`, `[0, 0.000754716981132]`,
`[-0.000619673827221, 0.00126385145253]`,
`[-0.000476462740614, 0.00131027253669]`, and
`[-0.000787337516897, 0.000752398341235]`.

Scene-level target-error deltas (scene order `00001garden`, `00003garden`,
`00005garden`) are:

| Horizon | Scene deltas (n events) |
|---:|---|
| +1 | 0 (24), 0 (26), +0.0190476190476 (105) |
| +2 | 0 (24), 0 (26), +0.0190476190476 (105) |
| +5 | 0 (24), +0.0192307692308 (26), +0.0188679245283 (106) |
| +10 | 0 (24), 0 (26), +0.0204081632653 (98) |
| +20 | +0.0208333333333 (24), 0 (26), +0.020202020202 (99) |

The full machine-readable result is
[`causal_v2/results/C2c_target_injection.csv`](causal_v2/results/C2c_target_injection.csv);
the complete metric tables are in
[`causal_v2/reports/C2c_TARGET_INJECTION.md`](causal_v2/reports/C2c_TARGET_INJECTION.md).

## Gate evaluation

The post-hoc gate was not changed:

1. **C2c propagation:** fail. At +2, +5, +10, and +20 the target-error CI
   lower bounds are `0`, `0`, `0`, and `0`, not strictly greater than zero.
   +5 and +20 have 2/3 positive scene effects, but the strict CI condition
   still fails.
2. **C1c recovery:** pass at +2, +5, +10, and +20. Each has upper CI below
   zero and at least 2/3 negative scene effects; +5, +10, and +20 are negative
   in all three scenes.
3. The conjunctive rule requires both C2c and C1c. Therefore
   `MECHANISM_VALIDATED = NO`.
4. The old `STATE_CAUSAL_GO = NO-GO` is preserved and is not renamed or
   overwritten.

The generated gate and evidence matrix are
[`causal_v2/MECHANISM_GATE.json`](causal_v2/MECHANISM_GATE.json),
[`causal_v2/MECHANISM_EVIDENCE_MATRIX.md`](causal_v2/MECHANISM_EVIDENCE_MATRIX.md),
and [`causal_v2/FINAL_MECHANISM_DECISION.md`](causal_v2/FINAL_MECHANISM_DECISION.md).

## Provenance and hashes

| Item | Value |
|---|---|
| Branch | `jev-gmt-causal-prototype` |
| Commit before this v2 publication | `d4878693913e0c7fae11b39732517ae9443077ae` |
| Previous GMT causal base commit | `416d543b24b7f04d06add750ced95f4a24c9f1f4` |
| Stage2 checkpoint SHA256 | `d1ab611670aa0284b910b25fd251f84b4cb13ba406d7ab89226adcd6ca702e17` |
| Test annotation SHA256 | `1c47323b7f4c3536495bd985d52dd7ad0c3396a929e9e1b8199eb1887afb19a9` |
| Fixed seed | `20260930` |
| TEST_LEN | `40` |
| BANK_SIZE | `10` |
| WITH_BANK | `true` |
| C1b event manifest SHA256 | `8a4f791afb77d346f8b4827c3bc463f69e914f195a97d6eb432fb102a5ec2ca0` |
| C2b event manifest SHA256 | `1848d1ca39dc73e790d5829ae912a6790eae67aa325448785feb85e0d08da6a3` |
| C1c CSV SHA256 | `79267a7b507ff57ba0e79e5d633d00b319106cc9291d4b64ec72fcc7c3bdae3d` |
| C2c CSV SHA256 | `ea9fa85044d4ad225fd6c35ee329fa60c00f965d0ac1a3d9e9b50f6f4c8e4af4` |
| C1c statistics SHA256 | `854f0cad8bcf7bfcab97061f3308006c9471d72dc42f91f61d9d1fe2ca979922` |
| C2c statistics SHA256 | `0b30bd5e90a3c1c46c6ac3057dfbf563ec721a40870b1adc3657f7770c2e910c` |

The final commit hash is intentionally reported by `git rev-parse HEAD` after
this file is committed; placing a commit hash into its own content would
change that hash. No datasets, checkpoints, feature dumps, raw images,
credentials, or large replay caches are part of the v2 publication.
