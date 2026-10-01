# Latest GMT causal follow-up for review

This is a publication summary of the already completed causal follow-up. No
new experiment, training run, JEV-GMT implementation, or gate change was made
while preparing this file. The source worktree is
`/data3/liuyeqiang/JEV_GMT`, on branch `jev-gmt-causal-prototype`.

The old root `FINAL_PROJECT_STATUS.md` is retained unchanged. It records the
first-round, pre-isolation status (`CONDITIONAL GO`). The latest isolated
follow-up decision in this document and in
`causal_followup/FINAL_FOLLOWUP_STATUS.md` supersedes that preliminary status.

All rates below are fractions in `[0,1]`; multiplying a delta by 100 gives
percentage points. A paired delta is the mean of event-level
`treatment_rate - sham_rate`. A pooled delta is the difference between pooled
treatment and SHAM wrong/total rates. Confidence intervals are paired event
bootstrap 95% percentile intervals, seed `20260930`, 4,000 resamples. The
canonical +5/+10/+20 C1b/C2b values are copied from
`causal_followup/results/FOLLOWUP_STATISTICS.json`. The +1/+2 rows are a
supplementary offline aggregation of the committed CSVs with the same paired
event procedure; no replay was run.

## A. Replay fidelity

Source files: `causal_followup/REPLAY_FIDELITY.json`,
`causal_followup/REPLAY_TRACE_COMPARISON.json`, and
`causal_followup/REPLAY_POSTFILTER_TRACE_COMPARISON.json`.

| Scene | Baseline records | Replay records | Pre-filter trace | Track decision/count equivalence |
|---|---:|---:|---|---|
| 00001garden | 2010 | 2010 | exact, first diff `null` | `true` |
| 00003garden | 3320 | 3320 | exact, first diff `null` | `true` |
| 00005garden | 1800 | 1800 | exact, first diff `null` | `true` |
| **Total** | **7130** | **7130** | **PASS** | **true** |

The fixed baseline metrics are the replay reference metrics because the
pre-filter association traces are exactly equal:

| Metric | Baseline | Replay/reference | Difference |
|---|---:|---:|---:|
| HOTA | 83.06787536626467 | 83.06787536626467 | 0 |
| AssA | 81.92432823733957 | 81.92432823733957 | 0 |
| MOTA | 79.62186419480615 | 79.62186419480615 | 0 |
| IDF1 | 82.63379469998145 | 82.63379469998145 | 0 |
| number_predictions | 48005 | 48005 | 0 |
| sequence_count | 6 | 6 | 0 |

Replay fidelity: **PASS**. The post-filter trace artifact explicitly reports
`NOT_COLLECTED`; pre-filter exact trace equality is the declared fidelity
criterion. Fixed stream: seed `20260930`, `TEST_LEN=40`, `WITH_BANK=true`,
`BANK_SIZE=10`.

## B. Snapshot roundtrip

Source file: `causal_followup/SNAPSHOT_ROUNDTRIP.json`.

| Scene | Snapshot SHA256 | Original digest | Loaded digest | Exact ID/trace | RNG exact | Frames |
|---|---|---|---|---|---|---:|
| 00001garden | `6be6fdb9a86eef7c08c07409bbcf030c0e95bfce11d7088351c34d426b1e88e6` | `b917cfb98a307c437ceebf80f6ed377f5e7ef1ea3da7f45f22e833f544e7fae8` | same | `true` | `true` | 10 |
| 00003garden | `43f452140e053479dcc0bf73ace8afbbac759085d063703a363504112191f4c3` | `ee3e7054fb47fe66f3dc6ce3037471d14e03fae881d1f753f28ca0b5595ab25a` | same | `true` | `true` | 10 |
| 00005garden | `ed7291ded21f4f9297d57aa44758bcee85b2617217c28699af8c9cd03c9df5e7` | `aa26b85dc389d67260f8b5bda5e62d4c6e678f1ecca75d220f8d9f169b50b2c6` | same | `true` | `true` | 10 |

Snapshot roundtrip: **PASS**. The stored continuation traces have no first
difference and no nonzero box/score discrepancy was recorded (effective
maximum discrepancy `0`, hence within any `1e-6` tolerance). The artifact does
not contain a separate floating-point tolerance sweep; exact trace equality is
the recorded check.

## C. C1 original: long-rollout assignment correction

Source: `causal/reports/C1_ORACLE_CORRECTION.md`,
`causal/results/C1_oracle_correction.csv`, and the C1 evaluation metric JSONs.

Frozen events: **71**. Intervention events applied: **70**. SHAM events
observed: **71**. Thus one intervention event was unapplied; the original
report does not record a skip reason for that event.

| Horizon | Events | SHAM error | Assignment-correction error | Delta (treatment - SHAM) | Paired 95% CI | Same-direction fraction | Cross-view difference |
|---:|---:|---:|---:|---:|---|---:|---:|
| +1 | 60 | 0.558333 | 0.641667 | 0.0833333 | [0.008333333333333333, 0.16666666666666666] | 0.166667 | 0.368421 |
| +2 | 55 | 0.618182 | 0.690909 | 0.0727273 | [-0.00909090909090909, 0.15454545454545454] | 0.163636 | 0.411765 |
| +5 | 57 | 0.666667 | 0.72807 | 0.0614035 | [-0.02631578947368421, 0.14912280701754385] | 0.175439 | 0.411765 |
| +10 | 54 | 0.472222 | 0.583333 | 0.111111 | [0.009259259259259259, 0.2222222222222222] | 0.222222 | 0.411765 |
| +20 | 56 | 0.491071 | 0.607143 | 0.116071 | [0.026785714285714284, 0.21428571428571427] | 0.178571 | 0.315789 |

Tracking metrics:

| Condition | AssA | IDF1 | HOTA | MOTA | number_predictions | sequence_count |
|---|---:|---:|---:|---:|---:|---:|
| SHAM | 81.92432823733957 | 82.63379469998145 | 83.06787536626467 | 79.62186419480615 | 48005 | 6 |
| Assignment correction | 82.34255591701813 | 82.80216182406349 | 83.18228396182761 | 79.29816817479585 | 47949 | 6 |

The old C1 metrics are rollout-level results and are not used as evidence for
the isolated state-repair gate.

## D. C1b isolated state repair

Source: `causal_followup/results/C1b_state_repair.csv`,
`causal_followup/results/FOLLOWUP_STATISTICS.json`, and
`causal_followup/reports/C1b_STATE_REPAIR.md`.

All three C1b branches use **71 frozen events** and **5,751 CSV rows**. Skip
counts below are treatment CSV rows with no estimable total (they are not
silently converted into effects):

| Branch | Skip rows |
|---|---|
| SHAM reference / assignment-only pairing | `scene_end=19`, `current_target_occupied=80` |
| Assignment correction only (C1b-A) | `scene_end=19`, `current_target_occupied=80` |
| Correction + Quarantine (C1b-B) | `scene_end=19`, `current_target_occupied=80` |
| Oracle State Repair (C1b-C) | `scene_end=19`, `repair_conflict=180`, `current_target_occupied=20` |

### SHAM rates used for the paired comparisons

The SHAM rate is calculated on the same estimable event subset as its paired
treatment. This is why C1b-C has a different SHAM denominator after repair
conflicts are excluded.

| Pairing | +1 | +2 | +5 | +10 | +20 |
|---|---:|---:|---:|---:|---:|
| C1b-A / C1b-B SHAM | 0.07742461287693561 | 0.08116883116883117 | 0.07061688311688312 | 0.06382978723404255 | 0.05906148867313916 |
| C1b-C SHAM | 0.07538601271571299 | 0.081374321880651 | 0.06890299184043518 | 0.06369426751592357 | 0.06057866184448463 |

### Assignment correction only (C1b-A)

| Horizon | Executable events | SHAM error | Treatment error | Paired delta estimate | Pooled delta | Paired 95% CI |
|---:|---:|---:|---:|---:|---:|---|
| +1 | 67 | 0.07742461287693561 | 0.07654723127035831 | -0.0006218905472636816 | -0.0008773816065772988 | [-0.0018656716417910447, 0] |
| +2 | 66 | 0.08116883116883117 | 0.08211382113821138 | 0.0006851196673331061 | 0.0009449899693802083 | [-0.0017266922800519635, 0.0035433517679071052] |
| +5 | 66 | 0.07061688311688312 | 0.06904955320877336 | -0.000913837540978516 | -0.0015673299081097603 | [-0.005352160161119318, 0.0038305762943444104] |
| +10 | 66 | 0.06382978723404255 | 0.06382978723404255 | 0.0010774410774410772 | 0 | [-0.0018181818181818182, 0.00505050505050505] |
| +20 | 66 | 0.05906148867313916 | 0.05825242718446602 | -0.0005808080808080809 | -0.0008090614886731365 | [-0.001893939393939394, 0.00015151515151515122] |

### Correction + Quarantine (C1b-B)

| Horizon | Executable events | SHAM error | Treatment error | Paired delta estimate | Pooled delta | Paired 95% CI |
|---:|---:|---:|---:|---:|---:|---|
| +1 | 67 | 0.07742461287693561 | 0.07579462102689487 | -0.001218905472636816 | -0.0016299918500407434 | [-0.003059701492537313, 0] |
| +2 | 66 | 0.08116883116883117 | 0.0836038961038961 | 0.00182012432012432 | 0.002435064935064929 | [0, 0.003796328671328702] |
| +5 | 66 | 0.07061688311688312 | 0.0698051948051948 | -0.0005611672278338944 | -0.0008116883116883189 | [-0.0016835016835016834, 0] |
| +10 | 66 | 0.06382978723404255 | 0.0630114566284779 | -0.0007974481658692185 | -0.0008183306055646461 | [-0.0023923444976076554, 0] |
| +20 | 66 | 0.05906148867313916 | 0.05825242718446602 | -0.0006313131313131313 | -0.0008090614886731365 | [-0.001893939393939394, 0] |

### Oracle State Repair (C1b-C)

| Horizon | Executable events | SHAM error | Treatment error | Paired delta estimate | Pooled delta | Paired 95% CI |
|---:|---:|---:|---:|---:|---:|---|
| +1 | 61 | 0.07538601271571299 | 0.07622504537205081 | 0.000628415300546448 | 0.0008390326563378209 | [-0.0020491803278688526, 0.003934426229508196] |
| +2 | 60 | 0.081374321880651 | 0.08152173913043478 | 0.00005918718962197242 | 0.00014741724978378778 | [-0.002497010236140671, 0.0026410256410256414] |
| +5 | 60 | 0.06890299184043518 | 0.06799637352674524 | -0.0002805836139169474 | -0.0009066183136899331 | [-0.00516273849607183, 0.0049382716049382715] |
| +10 | 60 | 0.06369426751592357 | 0.06551410373066424 | 0.002574074074074074 | 0.001819836214740675 | [-0.0013333333333333333, 0.00787037037037037] |
| +20 | 60 | 0.06057866184448463 | 0.059674502712477394 | -0.0006388888888888889 | -0.0009041591320072331 | [-0.0020833333333333333, 0.00016666666666666636] |

Per-scene pooled deltas for the canonical target horizons are:

| Branch | +5: 00001 / 00003 / 00005 | +10: 00001 / 00003 / 00005 | +20: 00001 / 00003 / 00005 |
|---|---|---|---|
| C1b-A | `0, 0, -0.0019392738339585186` | `0, 0.007575757575757583, -0.0010162601626016177` | `0, 0, -0.0010090817356205803` |
| C1b-B | `0, 0, -0.0010050251256281395` | `0, 0, -0.0010162601626016177` | `0, 0, -0.0010090817356205803` |
| C1b-C | `0, 0, -0.001141552511415525` | `0, 0.008196721311475412, 0.0011481056257175715` | `0, 0, -0.0011494252873563288` |

No C1b branch produced new rollout-level HOTA, IDF1, AssA, or MOTA. The
C1b endpoint is the isolated future identity-error rate. Recovery time,
cross-view cascade, and contamination counts were not collected as separate
C1b fields.

## E. C2 original: compound injection

Source: `causal/reports/C2_ERROR_INJECTION.md`,
`causal/results/C2_error_injection.csv`, and the C2 evaluation metric JSONs.

Frozen events: **939**. Injection events applied: **535**. SHAM events
observed: **939**. Thus 404 injection events were unapplied in the long
rollout; the original report does not provide a skip-reason breakdown.

| Horizon | Events | SHAM error | Injection error | Delta (treatment - SHAM) | Paired 95% CI | Same-direction fraction | Cross-view difference |
|---:|---:|---:|---:|---:|---|---:|---:|
| +1 | 503 | 0.0308151 | 0.415507 | 0.384692 | [0.34393638170974156, 0.42646620278329983] | 0.389662 | 0.0819672 |
| +2 | 504 | 0.0327381 | 0.409722 | 0.376984 | [0.3353174603174603, 0.41964285714285715] | 0.382937 | 0.0737705 |
| +5 | 510 | 0.0411765 | 0.417647 | 0.376471 | [0.33431372549019606, 0.41862745098039217] | 0.382353 | 0.0901639 |
| +10 | 490 | 0.0408163 | 0.403061 | 0.362245 | [0.3193877551020408, 0.4051020408163265] | 0.367347 | 0.0813008 |
| +20 | 472 | 0.0423729 | 0.398305 | 0.355932 | [0.3125, 0.399364406779661] | 0.364407 | 0.0806452 |

Tracking metrics:

| Condition | AssA | IDF1 | HOTA | MOTA | number_predictions | sequence_count |
|---|---:|---:|---:|---:|---:|---:|
| SHAM | 81.92432823733957 | 82.63379469998145 | 83.06787536626467 | 79.62186419480615 | 48005 | 6 |
| Compound injection | 76.74941156489075 | 79.29562015200187 | 80.16668909823407 | 77.95004781873023 | 47732 | 6 |

The +5 delta is the exact value behind the earlier approximate statement
“about +0.376”; it is not the isolated C2b estimate.

## F. C2b isolated single-error injection

Source: `causal_followup/results/C2b_isolated_injection.csv`,
`causal_followup/results/FOLLOWUP_STATISTICS.json`, and
`causal_followup/reports/C2b_ISOLATED_INJECTION.md`.

The isolated condition has **256 frozen events** and **10,496 CSV rows**.
Skip rows are `current_target_occupied=280` and `scene_end=8`. The paired
executable event count is 242 at +1, +2, +5, and +10, and 241 at +20.

| Horizon | Executable events | SHAM error | Injected error | Paired delta estimate | Pooled absolute delta | Paired 95% CI |
|---:|---:|---:|---:|---:|---:|---|
| +1 | 242 | 0.04786485218207415 | 0.04834545881248533 | 0.00012125921216830307 | 0.00048060663041118307 | [-0.0009194509194509196, 0.0009535918626827718] |
| +2 | 242 | 0.04715868898844612 | 0.047461629279811096 | 0.0003214826256507895 | 0.00030294029136497536 | [-0.00027614610506274166, 0.0010204568009544654] |
| +5 | 242 | 0.039782145394269476 | 0.04052132701421801 | 0.0006006720352749817 | 0.0007391816199485354 | [-0.00010825492230450906, 0.0013889518225217114] |
| +10 | 242 | 0.04199103562160887 | 0.042000943841434636 | 0.00001697321855424617 | 0.000009908219825764675 | [-0.00047829567592413453, 0.0005217294051286144] |
| +20 | 241 | 0.037230258477590705 | 0.036755987668958975 | -0.0004027616214632555 | -0.0004742708086317299 | [-0.001211225987773221, 0.00027497005211904045] |

Per-scene pooled absolute deltas and direction consistency:

| Horizon | 00001garden | 00003garden | 00005garden | Strictly positive scenes |
|---:|---:|---:|---:|---:|
| +1 | -0.000005652783430560963 | -0.0020105059630751328 | 0.0010040378732075542 | 1/3 |
| +2 | 0 | 0.00006469923482371742 | 0.0004293371765707113 | 2/3 (one zero) |
| +5 | 0 | 0.00006165988407941594 | 0.00102297169030887 | 2/3 (one zero) |
| +10 | 0 | 0 | 0.000017284579583935755 | 1/3 |
| +20 | -0.000015136378772742133 | 0 | -0.0006322091829535081 | 0/3 |

The isolated C2b endpoint did not collect separate recovery-time, error-streak,
or cross-view-spread fields. Those are therefore **not available** for C2b;
the table reports the required future identity-error endpoint and its
per-scene direction.

## G. Final pre-registered gate

The machine-readable result is
`STOP_NO_STRONG_CAUSAL_GATE` in `FOLLOWUP_STATISTICS.json`.
The requested three-level label is:

**`STATE_CAUSAL_GO = NO-GO`**

The pre-registered checks are unchanged:

1. **Does isolated C2b significantly increase future error?** No. Every C2b
   target-horizon paired 95% CI crosses zero: +5
   `[-0.00010825492230450906, 0.0013889518225217114]`, +10
   `[-0.00047829567592413453, 0.0005217294051286144]`, and +20
   `[-0.001211225987773221, 0.00027497005211904045]`. The supplementary +1
   and +2 intervals also cross zero.
2. **Does C1b quarantine or oracle state repair significantly reduce future
   error?** No. C1b-B intervals touch or cross zero at +5/+10/+20; C1b-C
   intervals cross zero at all three horizons. No interval has the required
   strict upper bound below zero.
3. **Is there at least one +5/+10/+20 horizon whose CI does not cross zero?**
   No for the isolated C2b and C1b follow-up conditions. The strong C2 result
   in the older compound rollout is not an isolated estimate and cannot be
   substituted into this gate.
4. **Are at least 50% of estimable scenes direction-consistent?** C2b has
   two strictly positive scenes at +5, but its pooled CI still crosses zero;
   C1b-B has only one strictly negative scene at each target horizon and C1b-C
   does not maintain two negative scenes across the target horizons. The
   conjunctive gate therefore fails.

No JEV-GMT implementation or follow-up method is authorized by this result.

## H. Provenance

| Item | Value |
|---|---|
| Branch | `jev-gmt-causal-prototype` |
| Current commit at summary generation (before publication commit) | `bdf899250722511ea620558e0689b24c8c0db7ff` |
| GMT base commit (parent) | `416d543b24b7f04d06add750ced95f4a24c9f1f4` |
| Stage2 checkpoint SHA256 | `d1ab611670aa0284b910b25fd251f84b4cb13ba406d7ab89226adcd6ca702e17` |
| Test annotation SHA256 | `1c47323b7f4c3536495bd985d52dd7ad0c3396a929e9e1b8199eb1887afb19a9` |
| Fixed seed | `20260930` |
| TEST_LEN | `40` |
| BANK_SIZE | `10` |
| WITH_BANK | `true` |
| Original C1 correction manifest SHA256 | `4d73c06aa3add46e6b033ca79c16c47f621cf9562ad6a28e8b4081a88c2e3906` |
| Original C2 injection manifest SHA256 | `ac5f3bfee795987570090994bab21c195584833e30fa09008018e9acb27904e3` |
| C1b event manifest SHA256 | `8a4f791afb77d346f8b4827c3bc463f69e914f195a97d6eb432fb102a5ec2ca0` |
| C2b event manifest SHA256 | `1848d1ca39dc73e790d5829ae912a6790eae67aa325448785feb85e0d08da6a3` |

The publication commit hash is intentionally reported by `git rev-parse HEAD`
after this file is committed, since inserting a commit hash into its own
content would change that hash.
