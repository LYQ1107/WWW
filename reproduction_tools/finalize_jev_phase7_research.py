"""Conditional Phase VII research verdict from completed P0.5 and native gates."""
import json
from jev_phase7_common import *

def read(name):return json.loads((REPORTS/name).read_text())

def main():
    protect();p05=read('PHASE7_MATCH_MECHANISM_GO_NO_GO.json');dataset=read('NATIVE_CAUSAL_MINISET_V1_AUDIT.json')
    prior=read('PRIOR_CANONICAL_LEARNABILITY_AUDIT.json');memory=read('MEMORY_H64_IDENTIFIABILITY_AUDIT.json')
    tests=read('MINISET_TESTS.json');assert tests['status']=='PASS'
    propagation=read('IDENTITY_ERROR_PROPAGATION_AUDIT.json')
    assert p05['attribution_status']=='COMPLETE'and dataset['construction_status']=='COMPLETE_VALIDATED_BOUNDED_MINISET'and memory['execution_status']=='COMPLETE'
    assert not dataset['gate']['pass']and memory['status']=='BLOCKED_MEMORY_IDENTIFIABILITY'
    learnability={'status':'COMPLETE_WITH_QUESTION_GATES_BLOCKED','MATCH':{
        'status':dataset['status'],'bounded_train_val':dataset['roles'],'unscreened_population':dataset['unscreened_native_population'],
        'joint_conflict_audit':dataset['joint_conflict_audit'],'historical_labels':prior['historical_MATCH'],'historical_labels_native_truth':False,
        'heldout_paired_source':{'events':11,'unknown_target_events':11,'GT_unmatched_target_fitting_weight':0},
        'hypothesis_scope':'candidate identity choice not supported by selected labels; complete train stream has 28 raw proposal errors with known correct candidate, complete validation stream has zero'},
        'MEMORY':{'status':memory['status'],'preserved_canonical':prior['native_MEMORY'],'H64_followup':memory,
          'M0_M5_source_sha256':prior['memory_baseline_evidence_sha256'],'M2_equals_M0_preserved':True,
          'fixed_representation_baselines':'M3 EMA, M4 confidence EMA and M5 gallery can change trajectories; this does not create single-WRITE causal labels or justify a learned head'},
        'REACTIVATION':{'status':'BLOCKED_NATIVE_RELATIVE_REACT_HOOK','events':prior['REACT_total_events'],
          'preserved_canonical':prior['native_REACTIVATION'],'multiple_threshold_valid_stale':0,'assigned_wrong_with_valid_correct_alternative':0,
          'original_native_binary_space_preserved':True,'synthetic_candidates_added':False,
          'existing_head_validation':'Phase VI train24 improves but validation23 degrades; disabled, not refit'},
        'native_transition_parity':{'MATCH':'PASS_PRESERVED_11_SNAPSHOTS_AND_NEW_32_COMPLETE_STATE_FORKS',
          'MEMORY':'PASS_CONTROL_WRITE_COMPLETE_STATE_FOLLOWUP; NO_IDENTIFIABLE_UTILITY',
          'relative_REACT':'NOT_IMPLEMENTED_IN_PRODUCTION'},
        'WHAT_DID_WE_LEARN':'existence of state divergence or large candidate pools does not establish trainable corrective action labels; opportunity coverage and production semantics remain the limiting factors',
        'binding':binding(),'official_test_read':False}
    save(REPORTS/'LEARNABILITY_AUDIT.json',learnability);save(REPORTS/'PHASE7_LEARNABILITY_AUDIT.json',learnability)
    parity=read('NATIVE_TRANSITION_PARITY.json');parity['phase7_completed_evidence']={
        'heldout_full_video_runs':69,'paired_prefix_events':11,'paired_prefix_branches':187,'new_train_val_prefixes':32,
        'new_control_factual_complete_mutable_state_parity':True,'new_MEMORY_control_WRITE_complete_state_events':8,
        'heldout_legacy_digest_omits_counters_and_archival_assignments':True,
        'new_digest_every_mutable_field_including_counters_assignments':True,
        'source_audits':{'paired_sha256':sha(REPORTS/'PAIRED_REASSOCIATION_AUDIT.json'),
                        'miniset_sha256':sha(REPORTS/'NATIVE_CAUSAL_MINISET_V1_AUDIT.json'),
                        'memory_sha256':sha(REPORTS/'MEMORY_H64_IDENTIFIABILITY_AUDIT.json')}}
    parity['relative_REACT_production_parity']='BLOCKED_NATIVE_RELATIVE_REACT_HOOK';save(REPORTS/'NATIVE_TRANSITION_PARITY.json',parity)
    architecture={'status':'NOT_RUN_NATIVE_LEARNABILITY_GATES_FAILED','candidate_model_trained':False,'candidate_prototype_implemented':False,
        'Unified_trained':False,'MEMORY_head_trained':False,'REACT_head_refit':False,
        'reason':'MATCH lacks corrective validation opportunities and diverse supported winners; MEMORY H64 all ties; relative REACT hook absent',
        'P05_state_only_comparisons':'COMPLETE; fixed/dynamic/33969-parameter MLP/34203-parameter LayerNorm MLP/Binary/frozen B2 with identical native operator',
        'planned_candidate_parameter_count':32093,'planned_parameter_matched_variant':33887,'planned_candidate_MACs':'13184*N+17088; dense linear estimate only',
        'candidate_latency_calibration_state_perturbation':'NOT_MEASURED_BECAUSE_NO_ELIGIBLE_FIT',
        'supervision_ablation':'NOT_RUN; historical labels are not relabelled native truth',
        'lifecycle_transfer_and_error_propagation_factorial':'BLOCKED; no assertion of learned three-question benefit',
        'binding':binding(),'official_test_read':False}
    save(REPORTS/'ARCHITECTURE_ABLATION.json',architecture)
    plan=read('PHASE7_ARCHITECTURE_ABLATION_PLAN.json');plan['planned_parameter_matched_variant']=33887;plan['matched_variant_MACs']='13992*N+18018';plan['status']='BLOCKED_NATIVE_LEARNABILITY_GATES_FAILED';plan['reason']=architecture['reason'];save(REPORTS/'PHASE7_ARCHITECTURE_ABLATION_PLAN.json',plan)
    verdict={'status':'NO_GO_PHASE7_ARCHITECTURE_AND_UNIFIED_EXTENSION','task_execution_status':'COMPLETE_BOUNDED_RESEARCH_AND_CONDITIONAL_GATES',
        'P0_5':p05['status'],'PHASE_A':'COMPLETE_QUESTION_GATES_BLOCKED','PHASE_B':'COMPLETE_32_RECORD_NATIVE_MINISET_BLOCKED_FOR_FITTING',
        'PHASE_C':'NOT_RUN_A_B_GATES_FAILED','PHASE_D':'P05_STATE_ONLY_BASELINES_COMPLETE; CANDIDATE_AND_SHARED_ABLATIONS_NOT_RUN',
        'PHASE_E':'NATIVE_MATCH_AND_MEMORY_INTERVENTIONS_COMPLETE; LEARNED_THREE_QUESTION_FACTORIAL_BLOCKED',
        'PHASE_F':'COMPLETE_NEW_CONTROLLER_HELDOUTS_09_10_11; FOUNDATION_TRAIN_OVERLAP_DISCLOSED',
        'candidate_conditioning_recommended_as_next_hypothesis':True,'candidate_conditioning_proved_better':False,
        'current_state_only_MATCH_extension_allowed':False,'Full24_or_official_TEST_allowed':False,
        'native_label_coverage':{'MATCH':dataset['gate'],'MEMORY_informative_H64':0,'relative_REACT_production_hook':False},
        'research_recommendation':'freeze current B2; preregister a genuinely corrective validation distribution and candidate/native choice interface before fitting; diagnose bank/threshold effects before learned MEMORY; establish relative REACT commit parity before reactivation sharing',
        'architecture_superiority_claim_allowed':False,'commercial_Jev_or_RLCD_reproduction_claim_allowed':False,
        'checks':{'protected_B2_sha256':sha(B2),'PhaseV_head':PHASE5,'PhaseVI_head':BASE,'official_test_read':False,'Full24_started':False},
        'available_MATCH_to_MEMORY_error_chain':propagation['counts'],
        'results_sha256':{name:sha(REPORTS/name)for name in ['IDENTITY_ERROR_PROPAGATION_AUDIT.json','PHASE7_REASSOCIATION_ABLATION.json','PHASE7_BYTETRACK_COMPARISON.json',
            'PHASE7_MATCH_MECHANISM_GO_NO_GO.json','LEARNABILITY_AUDIT.json','CAUSAL_DATASET_AUDIT.json','NATIVE_TRANSITION_PARITY.json','ARCHITECTURE_ABLATION.json','HELDOUT_RESULTS.json']},
        'WHAT_DID_WE_LEARN':'strong ordinary controls explain the current MATCH gain; causal data distinguish forced damage from feasible correction; single memory updates alter bank scores without identity benefit through H64; Unified remains unqualified',
        'binding':binding()}
    save(REPORTS/'FINAL_GO_NO_GO.json',verdict)
    train=dataset['roles']['train'];val=dataset['roles']['validation'];mem=memory['counts'];p=read('PHASE7_REASSOCIATION_ABLATION.json')['pooled_metrics']
    text=f'''# Phase VII final research report

**NO-GO for current state-only MATCH expansion, candidate fitting on this MiniSet, and Unified three-question claims.** All authorized bounded attribution and learnability experiments are complete. Conditional architecture/lifecycle training did not start because the preregistered gates failed. This is a research conclusion, not an incomplete run disguised as a success.

## P0.5: the architecture claim fails the ordinary-controller test

69 full-video runs, six actual camera sequences in three preregistered TRAIN controller-heldouts (09/10/11), eleven same-prefix events / 187 branches, ten H8/H16/H32 live continuations. Fixed GMT perception, source scoring/candidate functions and native transition rules; paired current tensors/candidate orders are exact, while downstream online states mutate naturally. First and second native features/actions/interfaces are audited. All three legal actions are available to rules and learned controllers, using the same constrained Hungarian and binary second pass.

| Primary method | HOTA | AssA | IDF1 | IDSW | MOTA |
|---|---:|---:|---:|---:|---:|
| GMT | {p['GMT']['HOTA']:.4f} | {p['GMT']['AssA']:.4f} | {p['GMT']['IDF1']:.4f} | {p['GMT']['IDSW']:.0f} | {p['GMT']['MOTA']:.4f} |
| Fixed rule | {p['FIXED']['HOTA']:.4f} | {p['FIXED']['AssA']:.4f} | {p['FIXED']['IDF1']:.4f} | {p['FIXED']['IDSW']:.0f} | {p['FIXED']['MOTA']:.4f} |
| Binary JEV with disclosed rejection adapter | {p['BINARY']['HOTA']:.4f} | {p['BINARY']['AssA']:.4f} | {p['BINARY']['IDF1']:.4f} | {p['BINARY']['IDSW']:.0f} | {p['BINARY']['MOTA']:.4f} |
| Frozen B2 | {p['B2']['HOTA']:.4f} | {p['B2']['AssA']:.4f} | {p['B2']['IDF1']:.4f} | {p['B2']['IDSW']:.0f} | {p['B2']['MOTA']:.4f} |
| Ordinary LayerNorm MLP | {p['MLP_LN']['HOTA']:.4f} | {p['MLP_LN']['AssA']:.4f} | {p['MLP_LN']['IDF1']:.4f} | {p['MLP_LN']['IDSW']:.0f} | {p['MLP_LN']['MOTA']:.4f} |
| B2, global reassociation disabled | {p['B2_VALIDATION_ONLY']['HOTA']:.4f} | {p['B2_VALIDATION_ONLY']['AssA']:.4f} | {p['B2_VALIDATION_ONLY']['IDF1']:.4f} | {p['B2_VALIDATION_ONLY']['IDSW']:.0f} | {p['B2_VALIDATION_ONLY']['MOTA']:.4f} |

These are true pooled TrackEval, not averages. Fixed rules obtain ΔHOTA +0.4066/ΔAssA +0.8184, exceeding B2 +0.3515/+0.7083. Under exactly equal 5/3/3 trigger budgets, fixed still exceeds B2 by +0.0291/+0.0652. B2 exceeds normalized MLP by +0.0582/+0.1182 raw and +0.0823/+0.1596 at the same budget, below preregistered +0.1/+0.2 margins. Three videos and one learned-controller seed cannot establish statistical significance. Collapsed dynamic/unnormalized MLP runs are retained and do not justify dismissing their whole families.

The B2 global on/off × own/legacy validator factorial attributes −0.0065 HOTA/−0.0133 AssA to global reassociation with its own validator, −0.0037/−0.0077 to learned revalidation with global solve, and −0.0048/−0.0097 to their interaction. Effects are small and sequence-dependent. Full raw and diagnostic budget runs remain separate. Most gain is explained by ordinary first association control and reduced fragmentation; the native re-solve does not demonstrate an architectural benefit on these new heldouts.

All eleven B2-triggering targets are GT-unmatched. Original target utilities are not eligible identity labels. Paired externality diagnostics show some local B2 protection against alternative policies, but do not overturn the complete-video rule result. Born-anchor wrong-ID time is 16 camera-frames for GMT, fixed, B2 and normalized MLP, while extra identity fragments are respectively 182/13/37/152. This distinguishes identity contamination duration from fragmentation; an always-new tracker can have zero born-anchor wrong time and disastrous IDSW. Episodes, unknown/unanchored counts, fixed global identity alignment and gap/end censoring are reported separately, without inventing seconds.

ByteTrack is official pinned code, not a synonym for banned-edge global re-solve. At threshold 0.5 no retained (0.1,0.5) detections exist. Paper-derived 0.6 uses 100 genuine low-score detections and accepts 36 low-stage matches, with no fabricated input. The frozen detector floor around .525 truncates the stream. Per-camera Kalman tracking and absent GMT cross-camera fusion prevent an architectural interpretation of its lower pooled AssA. The 0.6 supplement is disclosed as source-derived after some primary outcomes.

An exportable [scientific figure](../reports/JEV_PHASE7/figures/PHASE7_REASSOCIATION_ATTRIBUTION.png) and [PDF](../reports/JEV_PHASE7/figures/PHASE7_REASSOCIATION_ATTRIBUTION.pdf) show the principal raw and same-budget contrasts, without invented confidence intervals. See [full attribution](PHASE7_REASSOCIATION_ATTRIBUTION.md) and JSONs under `reports/JEV_PHASE7` for all 23 conditions, trigger counts, banned edges, changed assignments, corrections/regressions, ACCEPT externalities, second acceptance, timings, per-video metrics and error episodes.

## Phase A/B: real native data exist, but corrective supervision is missing

The preserved 4,303 historical MATCH labels use legacy H8 shortcuts/frozen OFF futures and remain ineligible as new native causal truth. They support like-for-like controls only. Their common +0.25 margins can reflect the birth penalty; large raw state values exceed 3,000, motivating the pre-outcome normalization control.

The new Native Causal Lifecycle MiniSet v1 contains 16 train07 and 16 val06 records, actual full candidate evidence/matrices, all legal current-row actions, CONTROL, common live B2 H8/H16/H32 futures, independent mutable state/RNG snapshots, complete-field fingerprints, and four joint conflict groups. All control/factual complete-state and committed-ID checks pass. The loader and adverse-record tests are executable; future/GT fields cannot enter its inference input. Records, raw forks, snapshots, labels, population distributions and manifests are published.

Train has {train['raw_unique_best']} raw unique-best labels, {train['informative_eligible']} eligible after quarantining one unanchored prefix; validation has {val['raw_unique_best']} / {val['informative_eligible']}, five ties and one unanchored prefix. Eligible winners are ACCEPT only (15 train / 11 validation), ESS 15/11. Every current action can force native state divergence, but selected corrective alternatives are 0/0. The complete unscreened train stream has 28 proposal errors with a known correct candidate; complete validation has zero. This separates sampling coverage from validation-distribution coverage. The minimum informative-count gate passes; diverse-winning-action and corrective-opportunity gates fail. We do not reselect favourable events or replace validation after seeing labels.

Joint rejection interactions include +5 utility units in a train and validation conflict group: edge values are not generally additive. An early +0.25 interaction is quarantined with unanchored prefix identity. Per-horizon effects are preserved, and no local margin is labelled as HOTA or long-term architecture value.

WHAT DID WE LEARN? Forced damage teaches an easy ACCEPT policy. A candidate selector needs feasible wrong-proposal/correct-alternative examples in both train and validation, not merely many possible IDs or different branch hashes.

## MEMORY: representation changes do not yield single-write identity targets

Phase VI's 20 WRITE/SKIP sources remain tied; M0–M5 fixed-representation evidence is retained, including exact M2/M0 prediction equality. Eight predeclared known-target READ-enriched sources receive common H8/H16/H32/H64 continuation, with no branch-specific stop selection. All eight change the gallery and complete mutable state; all eight later change the actually read prototype and candidate scores. Across {mem['shared_READ_payloads']} shared actual READ payloads, committed IDs and raw predictions remain identical, and all eight H64 utilities still tie. Bank prototypes may update at later promotion rather than immediately at WRITE. Only two of the eight events contain actual target-identity queries; the other six are right-censored for target READ in H64, even though the edited prototype is used against other queries. This is missing target observability, not six negative target-effect labels. The native latest10 mean dilutes a mature-bank single observation; exact score shifts, actual target versus other/unknown query identities and threshold context are published.

This blocks learned MEMORY identifiability here. It does not prove all memory representation strategies are irrelevant. For the two events with real target queries, longer H64 still gives no identity consequence; for six others the target has not been queried and the bounded window cannot identify its recovery effect. Increasing model size or manufacturing nonzero labels would not fix either limitation.

## REACTIVATION and lifecycle sharing

All 1,503 preserved events are audited: 1,058 train24 and 445 val23. Raw bank pools can be large, yet no event has two candidates above the actual native 0.4 bank threshold; assigned-wrong plus valid correct alternate is 0/0. Canonical binary labels retain 16/9 informative examples. The existing learned REACT improves train24 and degrades val23, so it remains disabled. Relative production REACT is absent; the research resolver cannot prove native relative commit parity. Do not add synthetic stale candidates or turn a raw pool-size count into legal action support.

MEMORY and REACT therefore block shared three-question training. No Unified model, learned lifecycle factorial, candidate superiority, commercial Jev architecture or RLCD is claimed. [Candidate design](PHASE7_ARCHITECTURE_DESIGN.md) specifies online fields, invariant/equivariant set handling, semantic NEW, one-to-one constraints, distinct correctness/Q heads, planned 32,093 parameters and dense MAC estimate. It also identifies the unverified direct candidate-commit interface. Implementation/calibration/latency of that conditional prototype is NOT RUN because A/B failed.

## Available identity error propagation

A posthoc audit of the already executed legal MATCH interventions finds seven wrong existing-ID commits. All seven receive an actual same-payload MEMORY WRITE, establishing an observed MATCH-to-memory contamination link in mutable native state. No later REACT query exposes those edited identities in H32, so bank-mediated REACT propagation is not established. The audit reports true consecutive camera-frame episodes for all affected known targets. It does not infer shared-JEV benefit or isolate memory mediation without a lifecycle factorial.

## WWW relevance and reviewer limits

Actual predicted-identity index lookups report precision/recall, false joins, trajectory contamination, cross-camera continuity and lookup latency. Query-to-primary-predicted-ID alignment uses offline oracle GT; this is an optimistic controlled index audit, not an implemented visual query encoder or deployed WWW search service. A WWW application contribution is not established by the current tracker numbers.

Foundation GMT may have trained on TRAIN videos; the new splits are controller-heldouts, not official benchmark test generalization. Thirteen required primary papers/preprints plus ByteTrack were method-audited, with checked source commits where available and explicit unavailable repositories. Preprints are not called accepted papers. Candidate association, adaptive memory and typed decision mechanisms have prior art. A causal decision audit/data toolkit is concrete work here; a superior structured lifecycle architecture is not supported.

## Changes, checks and publication

Only isolated Phase VII research tools/docs/reports and archive ignore rules changed. Production `gtr/*`, protected Phase V/VI branches, old predictions and permanent B2 are preserved. New controls, new snapshots/labels, raw predictions, segment commits, native journals and evaluation outputs are archived with source/archive SHA manifests. An early in-progress ByteTrack compressed archive in commit 8016a68 was corrected without changing its original predictions; `ARCHIVE_COMPLETION_NOTE.json` records the correction. Later archives publish atomically from ignored staging.

Four native/controller/solver/clone/temporal tests and four actual MiniSet/adverse-record tests pass. Runtime assertions validate exact native tensors/masks/features, factual/control commit and state parity, legal actions, finite utility/probabilities, candidate ordering, split independence and no GT/future inference inputs. A final publication verifier checks archive/tracked-file integrity and immutable anchors. Source bindings record start plans as well as end manifests; documentation/publication commits progressed during long runs, so an end HEAD is not falsely presented as the execution start revision.

Research branch: `jev/www-jev-phase7-causal-structured-20261008`. Base Phase VI: `{BASE}`. Protected B2 SHA: `{B2_SHA}`. No Full24, million-record build, official TEST, old checkpoint overwrite or forced push.

## Final decision

Stop expanding current MATCH from its earlier positive results. Candidate conditioning remains a justified next hypothesis, conditional on a preregistered validation distribution with actual corrective opportunities and a verified native candidate-choice interface. Diagnose bank/threshold and genuine observable memory effects before learned MEMORY; establish relative REACT commit parity before any sharing. All failed/blocked gates are explicit. More compute alone does not establish the WWW/CVPR architecture claim.
'''
    (ROOT/'docs/PHASE7_FINAL_RESEARCH_REPORT.md').write_text(text)
    short=f'''# Phase VII learnability audit

MATCH: bounded native 16 train / 16 validation records; eligible unique-best counts 15/11, all ACCEPT. Corrective selected opportunities 0/0. Complete population opportunities 28 train / 0 validation. Two early unanchored-prefix labels are quarantined. Gate: BLOCKED_MATCH_CORRECTIVE_CANDIDATE_AND_ACTION_SUPPORT.

MEMORY: preserved 20 sources tied; new eight known-target common H64 forks all change gallery/read prototypes/scores, but none changes identities/predictions or utility. Gate: BLOCKED_MEMORY_IDENTIFIABILITY. Enriched sources are not population prevalence estimates.

REACT: all 1,503 events audited, no multiple threshold-valid candidates and no assigned-wrong/valid-correct alternative. 16/9 informative binary labels remain, but the relative production hook is absent and old validation degraded. Gate: BLOCKED_NATIVE_RELATIVE_REACT_HOOK.

Native state divergence alone is insufficient. Full JSON reports margins, tie/unknown counts, ESS, train/val roles, candidate coverage, true state/commit consequences and source SHAs. See [final report](PHASE7_FINAL_RESEARCH_REPORT.md) for WHAT DID WE LEARN at every phase.
'''
    (ROOT/'docs/PHASE7_LEARNABILITY_AUDIT.md').write_text(short)
    protect();print(json.dumps({'status':verdict['status'],'task_execution_status':verdict['task_execution_status']}))

if __name__=='__main__':main()
