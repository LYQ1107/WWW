"""P0.5 attribution gate, requiring every preregistered full-video and paired run."""
from collections import Counter
import json
from jev_phase7_common import *

PRIMARY=['GMT','FIXED','DYNAMIC','MLP','MLP_LN','BINARY','B2','B2_VALIDATION_ONLY']
LEGACY=[f'{p}_LEGACY_VALIDATOR'for p in ['FIXED','DYNAMIC','MLP','MLP_LN','BINARY','B2']]+['B2_ORIGINAL_LEGACY']
BUDGET=[f'{p}_BUDGET'for p in ['FIXED','DYNAMIC','MLP','MLP_LN','BINARY','B2']]
TAGS=PRIMARY+LEGACY+BUDGET+['BYTETRACK','BYTETRACK_PAPER06']


def main():
    protect();summaries={t:json.loads((REPORTS/'individual_results'/f'{t}.json').read_text())for t in TAGS}
    assert all(d['status']=='COMPLETE'and len(d['runs'])==3 for d in summaries.values())
    paired=json.loads((REPORTS/'PAIRED_REASSOCIATION_AUDIT.json').read_text());assert paired['status']=='COMPLETE'
    tests=json.loads((REPORTS/'CONTRACT_TESTS.json').read_text());assert tests['status']=='PASS'
    for stage in ['PRIMARY','CONTROLS','BUDGET','SUPPLEMENTS']:
        assert json.loads((REPORTS/f'EXECUTION_RESULTS_{stage}.json').read_text())['status']=='COMPLETE'
    pooled={t:d['true_pooled_TrackEval']for t,d in summaries.items()}
    def delta(a,b):return {k:pooled[a][k]-pooled[b][k]for k in ['HOTA','AssA','IDF1','IDSW','MOTA']}
    stats={};temporal={};query={}
    initial={}
    for tag,data in summaries.items():
        counts=Counter();correctness=Counter();durations=[];episodes=0;known=unknown=0;false_joins=tp=gtobs=retrieved=0
        for run in data['runs']:
            if 'initial_state_fingerprint'in run:
                video=run['video'];initial.setdefault(video,set()).add(run['initial_state_fingerprint'])
            counts.update(run.get('mechanism_counts',{}))
        for audit in data['offline_identity_diagnostics'].values():
            correctness.update(audit.get('mechanism_correctness',{}).get('counts',{}))
            durations.extend(e['duration_frames']for e in audit['wrong_ID_episodes']);episodes+=len(audit['wrong_ID_episodes'])
            known+=audit['known_observations'];unknown+=audit['unknown_GT_observations']
            q=audit['identity_index_query'];false_joins+=q['false_identity_joins']
            for row in q['queries']:tp+=row['tp'];gtobs+=row['gt_observations'];retrieved+=row['retrieved']
        rows=counts.get('match_rows',0);validations=counts.get('second_validations',0)
        stats[tag]={'native_counts':dict(counts),'descriptive_proposal_change_correctness':dict(correctness),
            'trigger_rate_per_MATCH_row':counts.get('reassociate_rows',0)/rows if rows else None,
            'second_validation_acceptance_rate':counts.get('second_validation_accepted',0)/validations if validations else None,
            'solver_ms_per_trigger_payload':1000*counts.get('solver_seconds',0)/counts['reassociate_payloads']if counts.get('reassociate_payloads')else None,
            'controller_seconds':sum(r.get('controller_seconds',0)for r in data['runs']),
            'complete_video_wall_seconds':sum(r['wall_seconds_excluding_eval']for r in data['runs']),
            'latency_scope':'measured on shared hardware, excludes final TrackEval; no dedicated speed superiority claim'}
        temporal[tag]={'birth_anchor_wrong_ID_camera_frames':sum(durations),'episodes':episodes,'maximum_frames':max(durations,default=0),
                       'known_GT_observations':known,'unknown_GT_observations':unknown,
                       'extra_birth_fragments':sum(a['extra_birth_fragments']for a in data['offline_identity_diagnostics'].values()),
                       'false_merge_tracks':sum(a['false_merge_tracks']for a in data['offline_identity_diagnostics'].values()),
                       'global_unmapped_fragment_observations':sum(a['global_unmapped_fragment_observations']for a in data['offline_identity_diagnostics'].values()),'units':'consecutive observed camera-frames; gaps/end censored; no seconds'}
        query[tag]={'precision_micro':tp/retrieved if retrieved else 0,'recall_micro':tp/gtobs if gtobs else 0,'false_identity_joins':false_joins,
                   'scope':'actual predicted-identity index lookup; offline oracle GT-to-primary-ID query alignment; not a deployed visual search service'}
    assert all(len(values)==1 for values in initial.values())
    gain_b2=delta('B2','GMT');gain_fixed=delta('FIXED','GMT')
    explained={k:gain_fixed[k]/gain_b2[k]if gain_b2[k]>0 else None for k in ['HOTA','AssA']}
    fair_budget=delta('B2_BUDGET','MLP_LN_BUDGET');budget_vs_rule=delta('B2_BUDGET','FIXED_BUDGET')
    factorial={}
    for metric in ['HOTA','AssA','IDF1','IDSW','MOTA']:
        y11=pooled['B2'][metric];y01=pooled['B2_VALIDATION_ONLY'][metric]
        y10=pooled['B2_LEGACY_VALIDATOR'][metric];y00=pooled['B2_ORIGINAL_LEGACY'][metric]
        factorial[metric]={'global_effect_with_own_validation':y11-y01,'global_effect_with_legacy_validation':y10-y00,
                          'learned_validation_effect_with_global':y11-y10,'learned_validation_effect_without_global':y01-y00,
                          'global_validation_interaction':y11-y10-y01+y00,
                          'definition':'complete-video intervention; policy is fixed B2, states and future trigger timing mutate naturally'}
    byte={t:{'pooled':pooled[t],'videos':[{'video':r['video'],'parameters':r['parameters'],'low_detections':r['low_detections'],
             'low_stage_status':r['low_stage_status'],'accepted_low_matches':r.get('low_stage_accepted_matches'),
             'nonempty_low_association_payloads':r.get('low_stage_nonempty_track_and_detection_payloads'),
             'predictions_sha256':r['predictions_sha256'],'official_source_sha256':r['official_source_sha256']}for r in summaries[t]['runs']]}for t in ['BYTETRACK','BYTETRACK_PAPER06']}
    byte.update(status='COMPLETE_AVAILABLE_INPUT_COMPARISON',paper_threshold=.6,official_tracker_commit='d1bf0191adff59bc8fcfeaa0b33d3d1642552a99',
                low_detection_stream='genuine but truncated by GMT cache detector floor around .525; lower discarded boxes unavailable',
                algorithm_distinction='official high/low confidence cascade with Kalman state and cost-limited LAP versus rejected-edge constrained global Hungarian plus binary learned revalidation',
                comparison_scope='independent per-camera traditional tracker, no GMT cross-camera fusion; not same-solver architecture attribution',
                fabricated_detections=False,official_test_read=False)
    b2_ln=delta('B2','MLP_LN')
    video_positive={}
    for video in (9,10,11):
        b2=next(r for r in summaries['B2']['runs']if r['video']==video)['metrics']
        mlp=next(r for r in summaries['MLP_LN']['runs']if r['video']==video)['metrics']
        video_positive[str(video)]=all(b2[k]>mlp[k]for k in ['HOTA','AssA'])
    architecture_gate=(all(video_positive.values())and b2_ln['HOTA']>=.1 and b2_ln['AssA']>=.2 and fair_budget['HOTA']>=.1 and fair_budget['AssA']>=.2)
    assert not architecture_gate,'unexpected gate outcome; reviewer must inspect frozen criterion'
    verdict={'status':'NO_GO_CURRENT_STATE_ONLY_MATCH_ARCHITECTURE_EXPANSION','attribution_status':'COMPLETE','binding':binding(),
        'same_solver_legal_action_contract':'PASS; native call asserts identical raw score tensor, banned edges and first features; all rules/learned policies receive A/R/NEW and own A/NEW second validation',
        'complete_video_runs':len(TAGS)*3,'heldout_videos':[9,10,11],'paired_events':paired['events'],'paired_branches':paired['branches'],
        'A_rule_gain_vs_GMT':gain_fixed,'A_rule_fraction_of_positive_B2_gain':explained,
        'B_B2_minus_normalized_MLP_raw':b2_ln,'B_B2_minus_normalized_MLP_same_budget':fair_budget,
        'B_B2_minus_fixed_rule_same_budget':budget_vs_rule,'B_statistical_significance':'NOT_ESTABLISHED; only three preregistered independent videos, one learned-controller seed; overlapping event windows are not independent replicates',
        'C_B2_global_validation_factorial':factorial,'C_first_policy_comparison_scope':'first actions also change future state; matched common-validator full-video controls and identical-prefix interventions separate this from second validation',
        'D_stop_current_MATCH_expansion':True,'D_candidate_conditioning':'research direction only; bounded native data/identifiability gate required before fitting',
        'architecture_superiority_demonstrated':False,'normalization_control_gate_passed':architecture_gate,
        'WHAT_DID_WE_LEARN':'ordinary fixed rules exceed frozen B2 gains; native global re-solve has no positive heldout B2 effect here; bounded candidate supervision must be audited before architecture work',
        'full24_authorized':False,'official_test_read':False}
    ablation={'status':'COMPLETE','binding':binding(),'pooled_metrics':pooled,'mechanisms':stats,'true_identity_error_propagation':temporal,
        'identity_index_queries':query,'deltas':{'B2_minus_GMT':gain_b2,'FIXED_minus_GMT':gain_fixed,'B2_minus_BINARY':delta('B2','BINARY'),
        'B2_minus_VALIDATION_ONLY':delta('B2','B2_VALIDATION_ONLY'),'B2_BUDGET_minus_FIXED_BUDGET':budget_vs_rule,
        'B2_BUDGET_minus_MLP_LN_BUDGET':fair_budget},'B2_factorial':factorial,'paired_audit_sha256':sha(REPORTS/'PAIRED_REASSOCIATION_AUDIT.json'),
        'budget_diagnostic':'retrospective factual B2 row totals 5/3/3 with exogenous uniform cumulative quota; timing changes; preserved all raw runs plus B2 quota run; not a deployed policy',
        'score_and_candidate_freezing':'same perception and unchanged score/candidate functions in closed loop; exact current tensors/sets in paired prefixes; downstream differences caused by mutated state are retained',
        'methods':TAGS,'raw_runs':{t:[{'video':r['video'],'predictions_sha256':r['predictions_sha256'],'mechanisms_sha256':r.get('mechanisms_sha256')}for r in d['runs']]for t,d in summaries.items()},
        'official_test_read':False}
    save(REPORTS/'PHASE7_REASSOCIATION_ABLATION.json',ablation);save(REPORTS/'PHASE7_BYTETRACK_COMPARISON.json',byte)
    save(REPORTS/'PHASE7_MATCH_MECHANISM_GO_NO_GO.json',verdict)
    save(REPORTS/'HELDOUT_RESULTS.json',{'status':'ALL_PREREGISTERED_ATTRIBUTION_RUNS_COMPLETE','results':summaries,'controller_heldouts':[9,10,11],'official_test_read':False})
    header='| Method | HOTA | AssA | IDF1 | IDSW | MOTA | R rows |\n|---|---:|---:|---:|---:|---:|---:|\n'
    table=''.join(f"| {t} | {pooled[t]['HOTA']:.4f} | {pooled[t]['AssA']:.4f} | {pooled[t]['IDF1']:.4f} | {pooled[t]['IDSW']:.0f} | {pooled[t]['MOTA']:.4f} | {stats[t]['native_counts'].get('reassociate_rows',0)} |\n"for t in TAGS)
    text=f'''# Phase VII P0.5 — Reassociation Attribution

**NO-GO for expansion of the current state-only MATCH architecture.** Attribution is complete: {len(TAGS)*3} complete-video runs on preregistered TRAIN controller-heldouts 09/10/11, 11 identical-prefix events / 187 branches, 10 common H8/H16/H32 continuations. Phase VI native mechanism results are retained, not overwritten. Full24 and official TEST remain untouched.

## A. Ordinary rules with the same native solver

Fixed threshold improves pooled HOTA by {gain_fixed['HOTA']:+.4f}, AssA by {gain_fixed['AssA']:+.4f}, compared with B2 {gain_b2['HOTA']:+.4f}/{gain_b2['AssA']:+.4f}. This equals {100*explained['HOTA']:.2f}% / {100*explained['AssA']:.2f}% of the positive B2 gain. Fixed rules beat B2 by {-delta('B2','FIXED')['HOTA']:.4f}/{-delta('B2','FIXED')['AssA']:.4f}. Rules trigger only 1/0/0 reassociations, B2 5/3/3: most ordinary-rule improvement cannot be attributed to frequent global re-solving. The raw score threshold and second-pass semantics are documented; GMT's track-length-scaled default threshold differs.

## B. Same solver and budget

B2 exceeds the normalized ordinary MLP by raw {b2_ln['HOTA']:+.4f}/{b2_ln['AssA']:+.4f}, and under equal 5/3/3 trigger budgets by {fair_budget['HOTA']:+.4f}/{fair_budget['AssA']:+.4f}. Both fail preregistered +0.1 HOTA/+0.2 AssA margins. Under the same budget B2 minus fixed rules is {budget_vs_rule['HOTA']:+.4f}/{budget_vs_rule['AssA']:+.4f}. Dynamic threshold and unnormalized MLP collapsed on these frozen historical labels; preserve their negative results, but do not infer that their entire model families are ineffective. The LayerNorm MLP supplement was frozen from source-scale analysis before reading heldout outcomes. Three videos and one seed do not establish statistical significance.

## C. First judgment, second validation, global assignment, interactions

B2 minus VALIDATION_ONLY is {delta('B2','B2_VALIDATION_ONLY')['HOTA']:+.4f}/{delta('B2','B2_VALIDATION_ONLY')['AssA']:+.4f}: global reassociation has a slightly negative complete-video effect here. The JSON provides the full global on/off × own/legacy validator factorial, plus every controller under a common legacy validator. These are policy interventions; their future states and trigger timing mutate naturally. Current-payload paired interventions instead hold the exact prefix, RNG, score matrix and candidate order fixed, then use common live B2 future policy.

All eleven B2-triggering target rows are GT-unmatched: their original target utilities are unavailable for fitting, not positive causal identity supervision. The paired audit separately measures effects on other known observations. At H32, fixed policy contributes 3 more wrong camera-frames than B2 at this enriched set, while fixed equal-current-trigger budget contributes 0; normalized MLP contributes 48. This local selected-event protection does not reverse the complete-video fixed-rule advantage. Windows overlap and are not population rates. Actual identity error episodes use fixed prefix/birth anchors, consecutive observed camera-frames and gap/end censoring; they are not a scalar proxy renamed as seconds. All full-video error episode records are published.

The mechanism journals retain score matrices, candidate semantic order, first actions/features, banned edges, second inputs/outputs, actual committed IDs, ACCEPT rows moved, solver/validator timings, corrections/regressions and segment commits. The legacy paired state digest includes identity/history/RNG/banks/metadata and omits diagnostic counters/archival assignments; a complete-field digest is implemented and clone-tested for the bounded native dataset. Complete-video inference is genuinely mutated-state and every future proposal is rebuilt. Score/candidate *functions* and exogenous perception are frozen; later score/candidate values may change as a causal consequence of changed state.

## D. Research decision

Stop expanding current MATCH on the basis of its earlier positive result. Ordinary rules explain more than all of the B2 gain in this heldout set, and Binary JEV is also stronger than B2 pooled. Candidate conditioning is a hypothesis to investigate only after bounded native learnability and MiniSet gates. No candidate training, Unified success, commercial Jev reproduction or RLCD claim is made.

## Complete pooled results

{header}{table}
Pooled metrics come from one TrackEval run over all six camera sequences, not a mean of per-video metrics. Raw runs remain primary; budget-matched runs are retrospective diagnostics that alter trigger timing. Latency was measured on shared hardware, so it does not support a dedicated throughput comparison. Query diagnostics use an actual predicted-identity index but offline oracle GT-to-primary-ID alignment; this is not a deployed visual retrieval system.

## Traditional ByteTrack comparison

Pinned official source d1bf0191adff59bc8fcfeaa0b33d3d1642552a99 is executed unchanged except an old NumPy compatibility alias. At the original 0.5 preset, (0.1,0.5) contains no retained detections. At the paper-derived supplementary 0.6 threshold, 16/41/43 genuine low-score detections produce 4/10/22 accepted low-stage matches. No detection is fabricated. The cache detector floor near 0.525 truncates the original low stream. Independent per-camera Kalman tracks, cost-limited LAP and no GMT cross-camera fusion are different from the project's rejected-edge global re-solve plus learned binary revalidation. ByteTrack's lower AssA does not prove JEV architecture superiority. The supplementary 0.6 choice is source-derived and disclosed as made after some primary outcomes, not an independent primary preregistration.

## WHAT DID WE LEARN?

A native solver can change accepted assignments as an externality, but its presence does not establish learned architecture value. Ordinary normalized controls and same-solver rule controls remove the main current superiority claim. Phase VI's positive native-versus-validation-only difference is sequence-dependent and is not independently reproduced on these three new controller-heldouts. Further work should first establish feasible corrective candidates, informative action consequences and valid native support for each lifecycle question.
'''
    (ROOT/'docs/PHASE7_REASSOCIATION_ATTRIBUTION.md').write_text(text)
    save(REPORTS/'P05_COMPLETION.json',{'status':'COMPLETE','verdict_sha256':sha(REPORTS/'PHASE7_MATCH_MECHANISM_GO_NO_GO.json'),
        'ablation_sha256':sha(REPORTS/'PHASE7_REASSOCIATION_ABLATION.json'),'report_sha256':sha(ROOT/'docs/PHASE7_REASSOCIATION_ATTRIBUTION.md'),
        'next_authorized_step':'bounded native learnability/MiniSet audit only; architecture fitting still gated','binding':binding()})
    protect();print(json.dumps({'status':verdict['status'],'rule_gain':gain_fixed,'same_budget_B2_minus_fixed':budget_vs_rule}))

if __name__=='__main__':main()
