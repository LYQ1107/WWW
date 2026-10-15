
# CV-PIL V2: separation from all-TRAIN error mining

2026-10-15. Status: P0/P1_COMPLETE; INSUFFICIENT_NATIVE_OPPORTUNITY / NO_UNIQUE_PREDICTIVE_EFFECT on the frozen 1000/300 cohort; P2 WAITING_EXTERNAL_FROZEN_MINER_RELEASE. See [the actual matched-control results](CV_PIL_V2_REAL_RESULTS_20261015_ZH.md). The protocol below remains the inherited research plan; no native tracking success is claimed.

## Division of labor

**All-TRAIN original-GMT miner (another branch, controlled by the user)**:
- Collects original GMT pre-commit score/candidate/history and post-commit native decisions on authorized original TRAIN scenes.
- Identifies known correct/wrong, missed known positive, top1 wrong but assigned correctly, candidate absent, Bank/birth, impure owner/alias/UNKNOWN separately.
- Preserves real video/frame/physical-camera/detection/native-ID keys; original checkpoint/config/data SHA; source commit; GT label derivation; original native candidate and full solver outcome. For TRAIN-only hard selection, document use of GT. Validation selection must be frozen without using labels.
- Exports a manifest plus compact event+feature references and counts by scene/person/causal episode. Never changes source GT or constructs an oracle candidate gallery.

**This CV-PIL branch (this research worktree)**:
- Develops a schema adapter, model, matched controls, pre-commit integration and metrics; it does not rerun all-TRAIN mining, extraction, nor download videos or re-run the large foundation teacher.
- Until the new miner publishes outputs, use only READ-ONLY event files already available from TrackVGGT branch (fit TRAIN4/5/6/7/8: 1000 selected; TRAIN12 development: 300) to test schema mapping and training mechanics. TrackVGGT event counts are historical, not automatically equivalent to the new miner's results.
- Never merge TrackVGGT AG/QD weights, scripts or experimental configs into CV-PIL.
- No GT, future image or true person ID can become a deployed predictor input.

## Why the existing +1/2517 is not a sufficient GO

Existing CV-PIL TRAIN5/8 -> TRAIN12 screening: native GTA 2500/2517 (99.3246%); predictive 2501/2517 (99.3643%); all changed choices are from just one person. A camera-removed predictor and a cached DINOv2 teacher tie the predictive score. Fit has four scene-person groups and TRAIN12 diagnosis just six persons. Conditional GT-filtered candidate ranking is not a full native online Hungarian/Bank outcome, nor HOTA. Preserve prior report and checkpoints unchanged.

TrackVGGT's TRAIN12 selected cohort reportedly contains 18 known Top1 ranking errors but no selected actual-primary wrong/missed outcomes. A Top1 error is not necessarily an error in the final native ID. Before triggering full-video testing, require demonstrably legal, GT-certifiable **actual native committed wrong/missed** corrections. Never turn UNKNOWN or absent-right-candidate into correctable examples.

## Immediate tasks while miner runs

P0: Freeze old positive/negative evidence and SHA. Write a READ-ONLY adapter to existing TrackVGGT event payload (native feature dim declared as 1152) and an input-schema validator for the next mined events. Do not silently slice arbitrary 1152-D vectors to1024-D without verifying VFCE/RPCE layout. Check t(history)<t(query), legal original candidate IDs, GT-free inference inputs, exactly matching original checkpoint/config/score scales, and dependence-aware video/person grouping. Fail closed for mismatches.

P1: Prepare three matched tiny heads using the exact SAME causal native history/candidate set:
- C2 rank16 low-rank candidate-ranking/hard-negative learner, no prediction loss.
- C3 same architecture/budget/samples plus future opposite-camera latent prediction auxiliary loss, with stop-gradient on future target, used only in TRAIN and NEVER in inference.
- C4 same C3 but removing explicit destination camera conditioning; optional C5 removes temporal-gap conditioning. Two-camera target identity can be inferred from source camera, so C3=C4 alone cannot determine camera invariance.
Keep C0 original GMT, C1 native GMT+simple historical cosine, and frozen GMT+MLP as separate deployment reference. Do not enlarge teacher or retrain GTA. Optional teacher is NOT an early-stage rescue.

P2: When miner publishes a legally authorized frozen split, ingest compact metadata/features by hash and run at most 300 matched training updates per core model, initially 1 seed. Report native-score Top1 correctness separately from actual matched/Bank final correctness, candidate coverage, GT label UNKNOWN/alias rates, and per-unique-person/scene net fixes. No training on diagnostic cohorts or post-hoc threshold selection. Q2-style KEEP/APPLY certification is not required here; pair identity supervision with legal candidate filtering is the learning objective.

P3: Only consider native deployment if (1) the evaluation cohort contains certifiable actual native errors reachable by lawful candidates; (2) C3 produces net improvements over C2 on >=2 distinct person-dependent groups and does not increase known harms; (3) benefit has a falsifiable predictive-specific attribution beyond camera/time/linear controls. One repeated correct clause does not qualify.

P4: If P3 passes, connect bounded new score correction AFTER full frozen GTA, preserving Hungarian and Gallery/Bank. Test OFF=original exact, no future/GT actor inputs, same image detections and checkpoint. On same authorized full native videos, compare GMT, GMT+MLP, C2, C3 and report raw+stock HOTA, AssA, CVIDF1, IDSW, Frag, full-image FPS. No cache timing reported as full FPS. Prior TRAIN12/16 are exposed development, not independent blind validation.

## Budgets, logging and stopping

While the miner runs, no new GPU jobs in this branch. P0/P1 and tests on CPU with read-only compact existing caches. After supervision qualification, apply a separately registered initial <=1 GPU-wall-hour training budget (not resetting old ledgers). Keep output small and on a disk with >=30 GiB free. Avoid GPUs occupied by independent research jobs, particularly TrackVGGT.

If no lawful actual native errors exist, report INSUFFICIENT_NATIVE_OPPORTUNITY. If C3 does not beat C2 under same conditions or remains identical to no-condition, report NO_UNIQUE_PREDICTIVE_EFFECT. A positive development result without full video and independent validation is DEVELOPMENT_POSITIVE_PENDING_FULL_NATIVE, not SCIENTIFIC_SUCCESS. No indefinite experiments, no newly unsealed TEST, and no 'JEPA/JEV' novelty from a label alone.

Deliverables: frozen V1 manifest, schema adapter and validation unit tests, model+matched-control contracts, pre-registration JSON, source binding/provenance, per-person error report, then contingent native full-video report. Push each completed stage and keep both branches' experiments isolated.
