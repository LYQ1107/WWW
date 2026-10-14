# JEV closed-loop fastcheck final decision — 2026-10-14

`JEV_INDEPENDENT_ADVANTAGE_NOT_ESTABLISHED`. The bounded fastcheck is complete; scientific success and original master completion are not established.

All rows below use the same full1034-frame two-camera TRAIN16, one frozen seed. References are exact frozen prior inputs and are separate from historical DEV17–19 three-seed scores. TRAIN16 is gradient-free for new modules but pre-exposed by initializers, so this is development evidence.

| Method | HOTA | AssA | CVIDF1 | IDSW |
|---|---:|---:|---:|---:|
| frozen parent | 69.710 | 69.867 | 87.960 | 1013 |
| frozen threshold | 70.605 | 71.106 | 89.825 | 1002 |
| frozen mlp | 68.536 | 66.989 | 86.763 | 1180 |
| frozen attention | 69.727 | 69.690 | 88.805 | 1103 |
| frozen jev | 68.556 | 67.023 | 86.455 | 1455 |
| attention_D0 | 67.425 | 65.181 | 85.784 | 1279 |
| attention_D1 | 68.892 | 68.073 | 88.038 | 1299 |
| direct_attention_D1 | 63.097 | 57.003 | 74.757 | 1692 |
| direct_jev_D1 | 67.052 | 64.476 | 82.406 | 1093 |
| jev_D0 | 66.525 | 63.674 | 82.574 | 1581 |
| jev_D1 | 66.657 | 64.001 | 83.511 | 1891 |

Data benefit is stronger in ordinary Attention; JEV aggregation did not protect identity stability. Direct JEV lowers switches but increases global-map fragmentation and lowers CVIDF1. Past-certificate duration and complete-video fixed-map duration are separate diagnostics; neither is called a unique physical causal share.

| Live model | Scene FPS | FPS change vs Fixed | Stage2 trial-mean p50/p95 ms | Deployed parameters | New trainable parameters |
|---|---:|---:|---|---:|---:|
| attention_D1 | 1.927 | -29.64% | 156.13/207.81 | 36165044 | 169092 |
| direct_jev_D1 | 2.652 | -3.17% | 90.17/112.17 | 36165107 | 169155 |
| parent | 2.739 | +0.00% | 82.43/103.06 | 35995952 | 0 |
| threshold | 1.962 | -28.35% | 154.88/185.57 | 36126227 | 130275 |

Every reported FPS uses three complete live-image runs on the same physical GPU8, V10032GB, with detector/VFCE/actual Gallery/Bank/assignment and no GT journals/evaluator in the timed loop. Native identity state resets between trials; the loaded model is reused. Common dormant pruning is disclosed and is not a JEV scientific gain. Original instantiated system53515309 parameters; common deployed Fixed35995952; Direct JEV36165107 (net+169155 versus common Fixed); Direct Attention36164979 (net+169027).

GPU budget: 3.091/24 registered GPU-wall hours (includes loading/IO; small manual setup/contract probes are outside this registry, so this is not total active-kernel time). Missing live timings: [].

## A

Data aggregation helped ordinary Attention relative to its D0 fine-tune (+1.467 HOTA); JEV improved only +0.132 HOTA and IDSW rose +310. A1 remains below its original Attention initializer; J1 remains below strong Fixed/Threshold. CLOSED_LOOP_NOT_SUFFICIENT_FOR_CURRENT_JEV; no general claim that DAgger cannot work.

## B

Paired native prefixes had identical initial input/state SHA but diverged at TRAIN12/24/1, TRAIN13/225/0, TRAIN14/6/0. Direction differs by video. Full data intervention helped ordinary control more than JEV. Missing/UNKNOWN/Bank-only support and correlated person histories remain. Current targets certify first-past native-ID/person compatibility. Multiple native IDs can remain compatible with the same person, and neither the certificate nor local WHO correctness is a unique long-term Global ID or future safety guarantee. This is a supervision/decision-alignment limitation, not an established exclusive architecture cause.

## C

Keep Strong Fixed and honest frozen JEV evidence. The current new Recovery JEV module lacks support for deployment or continued scale-up. Direct JEV remains an auditable diagnostic, not a successful replacement.

## D

The sole direct lawful score/choice repair was executed. Against unchanged J1: HOTA +0.396, IDSW 1891->1093, but CVIDF1 83.511->82.406 and full-video identity-inconsistency duration 6957->7741 camera-frames. Lower IDSW does not establish stable identity benefit. Comparing repair to M0 also changes loss/readout/solver. Matched direct M1/M2 compares complete typed/untyped variants including reader allocation, not question embeddings alone, and M1 still loses to the stronger existing ordinary Attention.

## E

Best matched TRAIN16 reference: frozen Pair Threshold HOTA70.605/AssA71.106/CVIDF1=89.825/IDSW1002. Best new P2 model: ordinary Attention D1 HOTA68.892/AssA68.073/CVIDF1=88.038/IDSW1299. Direct JEV:67.052/64.476/82.406/1093. Same-GPU live-image costs are recorded separately; no cached FPS imputation. Three complete live-image trial means on the same GPU: Threshold 1.962, Fixed 2.739, Attention D1 1.927, Direct JEV 2.652 scene FPS. Unchanged JEV D1 live FPS was not measured.

## F

No new JEV exceeds Strong Fixed, Threshold or best ordinary Attention. Direct JEV exceeds the matched direct-Attention variant (HOTA63.097) but that variant is weaker than the original/P2 Attention and is not the strongest ordinary baseline.

## G

All new results are seed20261009, reused authorized TRAIN, and pre-exposed TRAIN16. No independent complete video, cross-seed reproduction, new source permission, or future safety qualification was established. DEV/TEST/sealed datasets were not newly used.

## H

Prefer the evidence-supported Strong Fixed/ordinary Threshold baseline and retain the stronger ordinary Attention as research control. Stop adding JEV Questions/Readers. Any next research scope should first resolve lawful action/identity-preservation supervision and independent validation; no new run is authorized by this final decision.
