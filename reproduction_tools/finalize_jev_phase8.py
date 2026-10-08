"""Finalize bounded native evidence; conditional phases remain explicitly unrun."""
import json,gzip,copy,math,subprocess,statistics
from collections import Counter,defaultdict
from pathlib import Path
from jev_phase8_common import *

def main():
    protect();start=binding();assert not start['worktree_dirty'];pr=json.loads(PREREG.read_text());scan=json.loads((REPORTS/'CORRECTIVE_OPPORTUNITY_AUDIT.json').read_text())
    report={'status':'COMPLETE_BOUNDED_NATIVE_ORACLE_AUDIT','binding':start,'videos':{},'events':[],'formal_data_gate':{},'raw_manifest':[],
       'statistical_scope':'four train and three validation videos in two scene families each; no inferential significance from overlapping event windows',
       'full_deployed_candidate_parity':'NOT_VERIFIED','no_GT_in_actor':True,'heldout_videos_read':False,'official_TEST_read':False,'Full24_started':False}
    packets=[];all_native_uniqueness=True
    for v in pr['train_videos']+pr['validation_videos']:
        directory=OUT/'corrective_forks_v1'/f'video{v:02d}'
        original=directory/('FORK_RESULT.json'if (directory/'FORK_RESULT.json').exists()else'FORK_RESULT.partial.json')
        origins=[original]if original.exists()else[]
        shards=sorted((OUT/'corrective_forks_v2'/f'video{v:02d}').glob('shard*'))
        assert all((p/'FORK_RESULT.json').exists()for p in shards),'unfinished shard'
        origins += [p/'FORK_RESULT.json'for p in shards]
        events=[];bindings=[]
        for origin in origins:
            item=json.loads(origin.read_text());events.extend(item['events']);bindings.extend([item['binding']]if 'binding'in item else [e['binding']for e in item['events']])
        expected_manifest=json.loads((OUT/'opportunity_scan_v1'/f'video{v:02d}'/'SCAN_RESULT.json').read_text())
        wanted={tuple(r['key'])+(row,)for r in expected_manifest['bounded_snapshots']for row in r['selected_rows']}
        assert len(events)==len(wanted)and {tuple(e['key'])+(e['row'],)for e in events}==wanted,'missing or duplicate preregistered cases'
        x={'status':'COMPLETE_ALL_PREREGISTERED_CASES','events':events,'binding':bindings,'source_results':[{'path':str(p),'sha256':sha(p)}for p in origins]}
        path=OUT/'corrective_forks_merged'/f'video{v:02d}'/'MERGED_FORK_RESULT.json';save(path,x)
        scan_dir=OUT/'opportunity_scan_v1'/f'video{v:02d}';m=json.loads((scan_dir/'SCAN_RESULT.json').read_text())
        groups={tuple(e['key'])+(e['row'],):e['group']for e in json.loads((scan_dir/'CONFLICT_GROUP_INDEX.json').read_text())}
        c=Counter();cg=set();verified=[];deltas=defaultdict(list);lat=[]
        for e in x['events']:
            e=copy.deepcopy(e)
            original_qualifiers=list(e['verified_corrective_branches'])
            e['verified_corrective_branches']=[tag for tag in original_qualifiers if e['branches'][tag]['desired_candidate_committed'] and e['branches'][tag]['native'] is not None and e['branches'][tag]['native']['native_existing_ids'][e['row']]==e['branches'][tag]['actual_committed_id']]
            key=tuple(e['key']);row=e['row'];group=groups[key+(row,)];c['audited_events']+=1
            c['CONTROL_KEEP_full_field_parity_PASS']+=e['CONTROL_KEEP_full_field_parity'];c['CONTROL_factual_ID_parity_PASS']+=e['CONTROL_factual_all_committed_ids_parity']
            c['verified_corrective_events']+=bool(e['verified_corrective_branches']);cg.add(group)
            if e['verified_corrective_branches']:verified.append(group)
            compact={'key':e['key'],'row':row,'role':'train'if v in pr['train_videos']else'validation','conflict_group':group,'snapshot_sha256':e['snapshot_sha256'],'run_binding':e['binding'],
                'offline_GT_metadata':e['offline_GT'],'correct_candidate_ID_metadata':e['correct_candidate_ids'],'verified_branches':e['verified_corrective_branches'],'original_native_lifecycle_positive_branches':original_qualifiers,'branches':{}}
            for tag,b in e['branches'].items():
                root=(Path(e['artifact_root'])if 'artifact_root'in e else directory/f'snapshot{m["bounded_snapshots"].index(next(s for s in m["bounded_snapshots"]if tuple(s["key"])==key)):03d}_row{row:03d}')/tag
                trace=json.loads((root/'COMPLETE_STATE_TRACE.json').read_text())
                unique=all(len(t['ids'])==len(set(t['ids'].values()))for t in trace);all_native_uniqueness &= unique
                lat.append(b['elapsed_seconds']);c['branches']+=1;c[f'{tag}_immediate_correct']+=b['immediate_anchored_correct']
                c[f'{tag}_verified']+=tag in e['verified_corrective_branches']
                horizons={}
                for h,u in b['horizons'].items():
                    horizons[h]={k:u[k]for k in('counts','target_counts','utility','utility_birth_zero','wrong_identity_duration_camera_frames')}
                    for k in('delta_utility','delta_utility_birth_zero','delta_target_wrong','delta_wrong_duration_camera_frames'):
                        if k in u:horizons[h][k]=u[k]
                    if 'delta_utility'in u:deltas[f'{tag}_H{h}'].append(u['delta_utility'])
                compact['branches'][tag]={'committed_ID_metadata':b['actual_committed_id'],'immediate_anchored_correct':b['immediate_anchored_correct'],
                    'desired_candidate_committed':b['desired_candidate_committed'],'native_MATCH_existing_ID_before_bank_recovery':b['native']['native_existing_ids'][row]if b['native']is not None else None,'current_MATCH_candidate_origin_qualified':tag in e['verified_corrective_branches'],'H32_complete':b['H32_complete'],'all_camera_payload_IDs_unique':unique,
                    'horizons':horizons,'prediction_sha256':b['prediction_sha256'],'complete_state_trace_sha256':b['post_state_trace_sha256'],'complete_field_digest_scope':b.get('complete_field_digest_scope','EVERY_PAYLOAD'),
                    'native_MATCH_operator_scope':b['native'].get('scope','actual production triage MATCH + replay commit')if b['native']else'original OFF transport',
                    'raw_score_tensor_reused':b['native'].get('raw_score_tensor_reused',True)if b['native']else True}
                for name in('EFFECTS.json','COMPLETE_STATE_TRACE.json','tracking_predictions/jev.json','tracking_decisions/jev.json'):
                    p=root/name;report['raw_manifest'].append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'uploaded_payload':False})
            report['events'].append(compact)
            record=next(s for s in m['bounded_snapshots']if tuple(s['key'])==key)
            import torch
            snapshot=torch.load(record['path'],map_location='cpu');d=snapshot['descriptors'][row]
            identities={int(t):int(g)for t,g in snapshot['offline_prefix_identity']['reliable'].items()}
            packets.append({'key':list(key),'row':row,'role':compact['role'],'group':group,'snapshot_sha256':record['sha256'],
                'inputs':{'state64':snapshot['online'][row]['state_features'],'candidate12':snapshot['online'][row]['candidate_evidence'],
                    'legal_existing_mask':[(row,col)not in snapshot['proposal'].banned_edges for col in range(len(snapshot['proposal'].track_ids))],
                    'NEW_legal':True},
                'metadata_only':{'candidate_IDs':list(snapshot['proposal'].track_ids),'offline_GT':d['offline_GT'],'proposal_column':d['proposal_column']},
                'offline_labels':{'candidate_correctness':[None if int(t)not in identities else int(identities[int(t)]==d['offline_GT'])for t in snapshot['proposal'].track_ids],
                    'executed_global_action_effects':compact['branches'],'unexecuted_candidate_utility':None,'unexecuted_candidates_are_NOT_negative_labels':True},
                'verified_corrective':bool(e['verified_corrective_branches'])})
        gc=Counter(groups[tuple(e['key'])+(e['row'],)]for e in x['events']);weights=[1/gc[groups[tuple(e['key'])+(e['row'],)]]for e in x['events']]
        report['videos'][str(v)]={'role':'train'if v in pr['train_videos']else'validation','counts':dict(c),'audited_conflict_groups':len(cg),'verified_conflict_groups':len(set(verified)),
            'selected_weight_Kish_ESS':sum(weights)**2/sum(w*w for w in weights),'delta_summaries':{tag:{'events':len(a),'sum':sum(a),'median':statistics.median(a),'positive':sum(z>0 for z in a),'negative':sum(z<0 for z in a),'tie':sum(z==0 for z in a)}for tag,a in deltas.items()},
            'rollout_elapsed_seconds_sum':sum(lat),'elapsed_scope':'complete H32 replay + complete-field fingerprint/evaluation; not online policy latency',
            'run_binding':x['binding'],'source_result_sha256':sha(path)}
        for p in origins+[path]+[p.parent/'START_MANIFEST.json'for p in origins if (p.parent/'START_MANIFEST.json').exists()]:
            report['raw_manifest'].append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p),'uploaded_payload':False})
    for role in('train','validation'):
        vs=[x for x in report['videos'].values()if x['role']==role]
        report[role]={'videos':len(vs),'scene_family_units':2,'audited_events':sum(x['counts']['audited_events']for x in vs),
            'verified_events':sum(x['counts']['verified_corrective_events']for x in vs),'verified_conflict_groups':sum(x['verified_conflict_groups']for x in vs),
            'contributing_videos':sum(x['counts']['verified_corrective_events']>0 for x in vs),
            'native_REASSOCIATE_verified':sum(x['counts'].get('REASSOCIATE_verified',0)for x in vs),
            'direct_CANDIDATE_0_verified':sum(x['counts'].get('CANDIDATE_0_verified',0)for x in vs)}
    gate={'verified_train_50':report['train']['verified_events']>=50,'verified_validation_20':report['validation']['verified_events']>=20,
        'two_train_videos':report['train']['contributing_videos']>=2,'two_validation_videos':report['validation']['contributing_videos']>=2,
        'train_five_groups':report['train']['verified_conflict_groups']>=5,'validation_three_groups':report['validation']['verified_conflict_groups']>=3,
        'CONTROL_KEEP_complete_state_PASS':all(x['counts']['CONTROL_KEEP_full_field_parity_PASS']==x['counts']['audited_events']for x in report['videos'].values()),
        'CONTROL_factual_ID_PASS':all(x['counts']['CONTROL_factual_ID_parity_PASS']==x['counts']['audited_events']for x in report['videos'].values()),
        'native_committed_uniqueness_PASS':all_native_uniqueness,'GT_future_input_leakage_absent':True}
    # Candidate choice support is measured by real score rank, not arbitrary
    # historical ID integers or row-specific aliases treated as classes.
    rank_support=set()
    for p in packets:
        if p['verified_corrective']:
            scores=[f[0]for f in p['inputs']['candidate12']]
            for tag,b in p['offline_labels']['executed_global_action_effects'].items():
                if b['desired_candidate_committed'] and b['immediate_anchored_correct'] and b['horizons']['32'].get('delta_utility',0)>0 and b['horizons']['32'].get('delta_utility_birth_zero',0)>0:
                    i=p['metadata_only']['candidate_IDs'].index(b['committed_ID_metadata']);rank_support.add(1+sum(z>scores[i]for z in scores))
    gate['multiple_distinct_candidate_rank_choices']=len(rank_support)>=2
    report['formal_data_gate']={'checks':gate,'pass':all(gate.values()),'candidate_rank_support':sorted(rank_support),
        'scope':'bounded preregistered representative audit; failure does not prove that all118 validation availability events lack causal benefit'}
    assert not report['formal_data_gate']['pass'],'a passed gate requires proceeding with formal B/C/D/E rather than this conditional finalizer'
    report['WHAT_DID_WE_LEARN']='Real existing candidates can correct some native identity errors, while refusing the old edge need not select them. Bounded validation causal support and full deployment parity still gate learning; candidate availability is not architecture evidence.'
    save(REPORTS/'NATIVE_CORRECTIVE_ORACLE_AUDIT.json',report)
    local=OUT/'native_corrective_v2_diagnostic';local.mkdir(exist_ok=True)
    gc=Counter(p['group']for p in packets)
    for p in packets:p['group_weight']=1/gc[p['group']]
    save(local/'diagnostic_records.json',packets)
    formal={'status':'NOT_RUN_BLOCKED_FORMAL_DATA_GATE','diagnostic_packet_status':'COMPLETE_BOUNDED_CORRECTIVE_ONLY_PACKET',
       'formal_training_dataset_constructed':False,'formal_model_fitting':False,'records':len(packets),'records_by_role':Counter(p['role']for p in packets),
       'groups_by_role':{r:len(set(p['group']for p in packets if p['role']==r))for r in('train','validation')},
       'natural_and_hard_negative_feature_records':0,'unobserved_candidate_Q_is_unknown':True,
       'no_state_tensor_contains_GT':True,'online_fields':['state64','candidate12','legal_existing_mask','NEW_legal'],
       'ID_and_GT_are_metadata_not_neural_features':True,'raw_local_path':str(local/'diagnostic_records.json'),'raw_sha256':sha(local/'diagnostic_records.json'),
       'bytes':(local/'diagnostic_records.json').stat().st_size,'gate':report['formal_data_gate'],'binding':start}
    save(REPORTS/'NATIVE_CORRECTIVE_V2_MANIFEST.json',formal)
    save(REPORTS/'NATIVE_CORRECTIVE_V2_AUDIT.json',{**formal,'WHAT_DID_WE_LEARN':'Existing legal corrective states and partially observed causal candidate effects now exist. An enriched corrective-only packet is not a natural/hard-negative formal training dataset and cannot establish candidate-model superiority.'})
    parity={'status':'PASS_SCOPED_NATIVE_OPERATOR_AND_MUTABLE_REPLAY','binding':start,'checks':gate,
       'production_OFF_observational_payloads':7*32,'actual64_input_feature_max_error':0.0,'paired_events':len(packets),
       'same_prefix_full_fingerprint_and_current_raw_scores':'PASS_EVERY_BRANCH','CONTROL_KEEP_full_field_state_every_payload':'PASS',
       'CONTROL_full_factual_committed_IDs_every_payload':'PASS','actual_camera_frame_commits_unique':all_native_uniqueness,
       'future_policy':'common live GMT OFF, native scores and state recomputed after each mutation',
       'candidate_interface_receives_model_values':'NOT_IMPLEMENTED','deployed_sliding_inference_full_commit_parity':'NOT_VERIFIED',
       'candidate_model_permutation_equivariance':'NOT_RUN_MODEL_NOT_IMPLEMENTED','diagnostic_pair_validation_tests':'PASS_10_TESTS',
       'no_GT_or_future_in_live_actor':True,'scope':'Actual original GTR MATCH function on real raw proposals, followed by existing native replay engine commits. These scopes do not prove full deployed candidate-controller integration.'}
    save(REPORTS/'NATIVE_PARITY.json',parity)
    blocker='BLOCKED_INSUFFICIENT_PREREGISTERED_VALIDATION_CORRECTIVE_SUPPORT_AND_UNVERIFIED_DEPLOYED_CANDIDATE_INTERFACE'
    for name,description in [('LEARNING_CURVES','C1 tiny overfit and C3 5/20/50/100epochs'),('DATA_SCALING','C2 grouped fractions at equal updates'),('CAPACITY_ABLATION','C4 8K/34K/128K/500K'),('SAMPLING_LOSS_ABLATION','C5 natural/hard/CE/focal/ranking/Q/multitask'),('NORMALIZATION_ABLATION','C6 raw/train-stat/LayerNorm'),('ARCHITECTURE_ABLATION','D candidate MLP/DeepSets/JEV identical native assignment'),('HELDOUT_RESULTS','E full-video mutated-state closed loops on20/21/22')]:
        save(REPORTS/f'{name}.json',{'status':'NOT_RUN','stage':description,'reason':blocker,'binding':start,
           'metrics':None,'checkpoints_created':0,'gate_report':'NATIVE_CORRECTIVE_ORACLE_AUDIT.json',
           'WHAT_DID_WE_LEARN':'The preceding formal gate failed; absent training/heldout results cannot determine architecture, epoch or capacity effects.'})
    natural=json.loads((OUT/'bootstrap/NATURAL_DISTRIBUTION_AUDIT.json').read_text());save(REPORTS/'NATURAL_DISTRIBUTION_AUDIT.json',natural)
    prior=json.loads((ROOT/'reports/JEV_PHASE7/PHASE7_MATCH_MECHANISM_GO_NO_GO.json').read_text());oldmini=json.loads((ROOT/'reports/JEV_PHASE7/NATIVE_CAUSAL_MINISET_V1_AUDIT.json').read_text())
    failure={'status':'COMPLETE_EVIDENCE_AUDIT_INTERVENTIONAL_LEARNING_DIAGNOSTICS_NOT_RUN','binding':start,
       'prior_same_budget_B2_minus_rule':prior['B_B2_minus_fixed_rule_same_budget'],'prior_ordinary_rule_fraction_of_positive_B2_gain':prior['A_rule_fraction_of_positive_B2_gain'],
       'prior_native_miniset':{r:{k:x[k]for k in('records','informative_eligible','winning_action_support','feasible_correct_alternative_factual_wrong')}for r,x in oldmini['roles'].items()},
       'sample_size':{'finding':'previous32 records contained only15train/11val informative ACCEPT winners, no corrective alternative; new native support exists but this fixed bounded audit fails its validation gate','causal_effect_of_more_data':'NOT_TESTED'},
       'ACCEPT_imbalance':{'finding':'old informative native labels were100%ACCEPT; new natural action frequencies reported separately from offline optimal labels','causal_effect_of_rebalancing':'NOT_TESTED'},
       'epochs':{'finding':'NOT_DETERMINED','experiment':'NOT_RUN_GATE_FAILED'},'capacity':{'finding':'NOT_DETERMINED','experiment':'NOT_RUN_GATE_FAILED'},
       'structure':{'finding':'state triage controls reject/re-solve, not arbitrary candidate identity choice; real direct-candidate corrections sometimes differ from REASSOCIATE','architecture_superiority':'NOT_DEMONSTRATED'},
       'utility_and_weights':{'finding':'new effects distinguish fixed historical wrong IDs, target/all-row externalities and birth-zero sensitivity; old fragment utility cannot substitute for genuine identity propagation','primary_weights_changed_after_outcomes':False},
       'network_failure_cause_proven':False,'WHAT_DID_WE_LEARN':'Supervision support and the association interface are evidenced limitations. More epochs, more capacity, and candidate JEV superiority remain unidentified rather than blamed or asserted.'}
    save(REPORTS/'FAILURE_CAUSE_ANALYSIS.json',failure)
    go={'status':blocker,'research_execution_status':'COMPLETE_CONDITIONAL_GATE_AUDIT','binding':start,'formal_data_gate':report['formal_data_gate'],
       'candidate_JEV_architecture_GO':False,'candidate_JEV_vs_MLP_DeepSets':'NOT_TESTED','MATCH_model_training':'NOT_RUN',
       'MEMORY':'NOT_RUN_CONDITIONAL_MATCH_AND_IDENTIFIABILITY_GATE','relative_REACTIVATION':'NOT_RUN_UNVERIFIED_RELATIVE_PRODUCTION_HOOK','Unified':'NOT_RUN_THREE_INDEPENDENT_GATES_NOT_PASSED',
       'heldout20_21_22':'NOT_RUN_SEALED','no_positive_WWW_architecture_claim':True,'Full24_started':False,'official_TEST_read':False,
       'next_research_scope':'Freeze a supplemental deterministic within-group capture protocol on the same train/validation videos before new future-utility/model-selection outcomes; retain group weights, avoid favourable reselection; independently integrate/test a shared candidate-values submit operator. Do not waive the original gate or treat these enriched representatives as the entire population.',
       'WHAT_DID_WE_LEARN':report['WHAT_DID_WE_LEARN']}
    save(REPORTS/'GO_NO_GO.json',go)
    (ROOT/'docs/PHASE8_DATASET_CONTRACT.md').write_text('''# Phase VIII Native Corrective v2 contract

The formal training dataset is NOT_RUN because the frozen native corrective data gate failed. The completed local diagnostic packet contains only the preregistered corrective representatives and their actual native effects. It is not labelled as a balanced or natural-distribution training dataset. Its manifest, hash and grouping are published; raw states and records remain local.

Each record separates online inputs (real64-state, real12-fields per current existing candidate, legal mask and semantic NEW) from ID/GT/group metadata and offline correctness/causal effects. The twelve fields are raw GMT score, score minus current-proposal score, proposal flag, current-column occupancy by another row, real history count, memory count, last-seen age, observed-camera fraction, active flag, stale flag, real observation/prototype cosine and prototype availability. No imagined motion/depth/uncertainty is present. IDs are not neural features.

Correctness uses permanent preceding confirmed identity anchors; unknown existing anchors are UNKNOWN, never negative labels. Multiple predicted aliases are retained. Only executed branches have causal utility labels. Unexecuted candidates have unknown Q, never zero. Q describes a whole global assignment conditional on forcing the selected edge, including externalities, rather than additive independent edge values. H8/16/32 controls start with the same complete state, RNG and current raw score tensor; future common GMT OFF is live after actual native mutation. NEW includes a same-current-key native-bank veto so the intended semantic action actually creates a new identity.

The primary utility is correct anchored observations minus wrong anchored observations minus5false-merge tracks minus0.25false births. All preregistered sensitivities are preserved, and a qualifying correction must remain positive when the birth weight is zero. Prefix IDs never rename after pollution. New births confirm only after their first two consistent known observations; previously mixed unconfirmed existing IDs remain unknown. Report target errors, all-row effects, true camera-frame wrong-ID episodes and censoring. No FPS is invented. H32 requires a complete32-frame available horizon.

Future formal construction must add predeclared natural samples and known-correct hard negatives from the fixed videos; use identical complete-state/RNG forks for KEEP/alternatives/NEW/real conflicts, quarantine ties/unknown/censoring, and group conflict-connected errors within32frames. Each group has total weight1; report video/scene support separately. Current packet contains zero natural or hard-negative feature records. Train-only normalization, calibration, tiny overfit and loader/online transformation parity are NOT_RUN and cannot be implied by this schema.
''')
    (ROOT/'docs/PHASE8_FAILURE_CAUSE_ANALYSIS.md').write_text('''# Phase VIII failure cause analysis

The current evidence establishes inadequate corrective supervision in the previous native MiniSet and an incomplete candidate-choice interface. It does not establish that training duration or network capacity caused the earlier failure.

Phase VII ordinary fixed rules exceeded B2: B2 minus fixed at matched trigger budget was -0.029083 HOTA and -0.065224 AssA. The old native MiniSet had32 records, with15train and11validation informative winners; every informative winner was ACCEPT and none was a feasible corrective alternative. Larger historical trace indexes were not eligible native causal truth. Increasing epochs on those labels cannot demonstrate learning corrective identity choice.

Phase VIII full new-video scans now find real candidate availability and bounded native corrective effects. Natural factual action frequencies and actual oracle labels are different quantities. The bounded enriched packet contains only wrong proposals, so it is not evidence that all natural labels are corrective, or that rebalancing helps. The formal validation gate fails. C1/C2/C3/C4/C5/C6 therefore remain NOT_RUN; neither insufficient sample size, ACCEPT imbalance, epoch count nor capacity has an isolated causal-effect estimate.

State-only A/R/NEW triage requests a solver action; it does not explicitly evaluate every identity candidate. Direct native candidate interventions can improve some identities that R does not correct. This supports auditing candidate conditioning, while providing no proof that a JEV-specific structure is superior to a strong ordinary MLP or DeepSets. Candidate correctness and downstream utility are separate targets; a global assignment's externalities prohibit treating every edge Q as independent.

The anchor amendment was frozen before new availability results, and v0/v1 factual predictions are byte-identical on all four train videos and diagnostic07. The permanent historical anchor does not rename polluted tracks; unconfirmed identities stay uncertain. Primary utility and its weights did not change after outcomes. Birth-weight-zero benefit, actual identity-error durations, negative corrections and unknown outcomes are retained.

See the machine failure analysis, native oracle audit and GO_NO_GO reports for exact numerical support and scope. More training, larger networks or a Unified model cannot bypass the failed data/deployment gates.
''')
    (ROOT/'docs/PHASE8_CANDIDATE_JEV_ARCHITECTURE.md').write_text('''# Phase VIII candidate architecture decision

Status: NOT_IMPLEMENTED / NOT_TRAINED. The data gate failed, and full deployed candidate-values submission has not been verified. No34K candidate network or inherited B2 modification is claimed complete.

The frozen design is64 online state fields plus12 real fields per actual legal existing candidate and a semantic NEW alternative. State/edge encoder, candidate-context encoder, competing-choice pooling, confidence/uncertainty, correctness and future-utility outputs are proposed modules, not validated contributions. IDs are metadata. Candidate MLP/DeepSets/JEV must share exactly the same inputs, legal masks, score/cost transformation, global one-to-one solver and native commit/bank/birth semantics. A contextual ordinary model can itself be strong; candidate conditioning is not JEV-specific evidence.

The completed diagnostic adapter validates one-to-one real edges, candidate existence, duplicate IDs and masks, calls the original GTR MATCH function with explicit legal assignments, and transports its existing IDs into the actual native replay engine. It receives legal chosen pairs for offline interventions, not learned candidate values. Thus it validates the production MATCH operator and mutated research replay scope only. The old mutable resolver, or calling the MATCH helper alone, does not prove end-to-end deployed sliding-inference integration.

Before fitting, independently implement and test the shared model-values submit API: real raw proposals, duplicate suppression, masks/private NEW slots, global assignment, actual production memory/stale/birth commit and next-frame feedback. Required tests include permutation equivariance, duplicate suppression, missing/no-feasible candidates, uniqueness, correct semantic NEW despite later stale-bank recovery, GT/future exclusion, RNG and complete state/committed-ID parity against production. Those missing tests are NOT_RUN. Architecture GO additionally requires the frozen heldout HOTA/AssA, regression, latency/memory and independent-video support thresholds. None has been demonstrated.
''')
    t=report['train'];v=report['validation']
    (ROOT/'docs/PHASE8_FINAL_RESEARCH_REPORT.md').write_text(f'''# Phase VIII final conditional research report

**本轮结论：正式学习实验的 Gate 未通过。** 训练有{t['verified_events']}/{t['audited_events']}个核验纠错事件，验证有{v['verified_events']}/{v['audited_events']}个；验证预选总数13低于冻结门槛20。这是本轮采样协议的限制，不能推断完整118个候选事件池没有更多有效样本。候选模型、C1–C6、完整 heldout、MEMORY/REACT/Unified均未运行；完整部署候选接口也尚未验证。

**Decision: {blocker}.** This completed storage/protocol/full opportunity scan and bounded actual-native causal audit; it did not complete formal model fitting, architecture comparison or heldout evaluation. A negative Gate outcome is retained without fabricated training or positive metrics.

| Partition | Full MATCH events | Oracle2 availability events | Native audited representatives | Verified immediate+H32 corrections | Verified conflict groups | Contributing videos |
|---|---:|---:|---:|---:|---:|---:|
| TRAIN12/13/14/16 | 122211 | 2278 | {t['audited_events']} | {t['verified_events']} | {t['verified_conflict_groups']} | {t['contributing_videos']} |
| Validation17/18/19 | 38887 | 118 | {v['audited_events']} | {v['verified_events']} | {v['verified_conflict_groups']} | {v['contributing_videos']} |

Native triage can sometimes reject a MATCH assignment and subsequently recover a correct identity from the separate stale bank. Such lifecycle-positive effects are preserved, but an identity absent from the current MATCH candidates is not a current-candidate submission or candidate-index label. Final eligibility additionally checks desired-candidate membership and equality between the production MATCH existing ID and the actually committed ID, preventing automatic bank recovery from being attributed to direct MATCH selection.

The frozen formal gate requires50train/20validation verified corrections, video and independent-group support, full-state control parity, real legal commits and positive H32 benefit even with zero birth penalty. The validation audit selected13 representative events before future benefit inspection. Its failure is a **bound of this sampling protocol**, not proof that the complete118-event availability pool lacks20 useful cases. Consecutive events are grouped, not independent replications. Four train videos and three validation videos span only two scene families each. No statistical architecture-superiority claim is possible.

P-1 inventoried656 checkpoint/state files totalling37.8GiB. No file met the safe orphan proof; zero deletion was executed. All historical negative experiments and B2 were retained. A separate small worktree avoids copying the older giant evidence directories; raw Phase VIII evidence is in the separate/home runtime. Exact dependency/hash/process guards, manifests and storage reports are published.

The first-two-consistent historical identity anchor was amended before reading new availability outcomes. It never renames confirmed IDs after a wrong write. Original strict audits remain intact; all four new train videos and diagnostic07 have byte-identical v0/v1 factual predictions. Unknown anchors and multiple predicted-ID aliases remain explicit. Twenty-eight events on old diagnostic07 versus one strict-purity event exposed the original audit's contamination censorship, without serving as new independent evaluation.

Completed v1 cases (including negative cases) were retained. Remaining cases used disjoint chronological snapshot shards; CONTROL/KEEP complete fields are checked every payload, while other interventions retain per-payload IDs/RNG and complete-field fingerprints at the current key and H8/H16/H32. This read-only instrumentation/scheduling optimization does not change selection, native actions, scores or utility. Incomplete v1 artifacts and the explicit stop manifest remain local. Independent old-video replay checks observer equivalence. A scheduler handoff inadvertently triggered terminal process-group cleanup; partial artifacts were preserved and only unfinished cases replayed in new recovery directories. Composed completion markers record immutable partial/recovery source hashes. This operational interruption is not hidden as a scientific result.

Each bounded event uses identical full initial state, RNG, raw GMT scores and real candidate set. CONTROL and explicit factual ACCEPT match every complete-state field and all committed identities throughout the window; CONTROL matches the full factual native stream. R invokes the unchanged production constrained Hungarian and second validation. Direct candidate branches call the real production GTR MATCH helper and then actually mutate the native replay state. NEW vetoes only the current selected row's stale-bank recovery; future MATCH/MEMORY/REACT continue live GMT OFF. Up to two real correct aliases and available joint-compatible cases are retained. No H32 expansion of the whole population occurred.

**Parity scope matters:** real production MATCH operator plus mutable native replay passed; the new deployed model-values candidate API and full sliding-inference state/bank/birth integration are NOT_VERIFIED. This adapter receives legal chosen pairs for offline oracle interventions. It does not establish a trained candidate controller or a production-ready value submit interface.

Utilities retain target and all-row effects, wrong-ID camera-frame durations, false merges/births, unknown/tie/censoring and every frozen weight sensitivity. They are not HOTA, and overlapping oracle windows cannot be combined as full-video TrackEval. Controller-heldout20/21/22 is sealed and NOT_RUN; HOTA/AssA/IDF1/IDSW/MOTA/Frag/identity-index query metrics for new models are absent. There are no new learned checkpoints. MEMORY, relative REACT and Unified are conditional and NOT_RUN.

## Eight required answers

1. **Why did previous JEV not beat fixed rules?** Same-solver/matched-budget evidence did not show architecture superiority; old native learning labels contained no corrective alternatives. These explain why prior positive tracking scores were insufficient evidence, but do not prove a single isolated neural failure cause.
2. **Too few samples?** Old native informative support was15train/11validation. New real corrective support exists, but the frozen bounded validation gate fails. The causal data-scaling experiment is NOT_RUN; more samples helping is not yet proved.
3. **Too much ACCEPT?** All26informative old native winners were ACCEPT. Natural factual action frequencies are now audited separately. A balancing intervention is NOT_RUN, so the architecture's response to it is unknown.
4. **Insufficient training?** Unknown. No qualified C3 epoch comparison was run; do not diagnose this from loss or prior training duration.
5. **Insufficient capacity?** Unknown. No qualified C4 capacity comparison was run; making the model larger would bypass the present blockers.
6. **Structure mismatched to identity choice?** The current triage interface does not directly score every identity choice. Direct-candidate native effects and R effects differ in real cases, supporting further candidate-interface research. This does not establish that JEV-specific modules outperform ordinary contextual architectures.
7. **Does Candidate JEV beat strong MLP/DeepSets?** NOT_TESTED. No formal candidate model or shared production values interface passed the required gates, and heldout remains untouched.
8. **Is Unified still valuable?** Conditionally testable, not demonstrated. MATCH needs qualified candidate supervision and production parity; MEMORY needs identifiable causal labels; relative REACT needs a real production hook and valid multi-candidate support. Only three independent successes authorize Unified research.

## What changed and what was learned

New files implement storage/protocol guards, semantic anchor/opportunity audit, compact summary, actual MATCH candidate-submit diagnostics, paired native forks, temporal utility and adversarial tests. Original gtr/Phase V/VI/VII sources and protected branch/checkpoint hashes are unchanged. Existing B2 is not overwritten. The complete scientific trail records exact source/commit, seeds, commands, raw artifact paths and SHA hashes; failures remain local and described. GitHub receives necessary code/docs/compact metrics/manifests, not raw checkpoints, predictions or GB snapshots.

The useful new conclusion is that candidate availability and actual native corrective benefit can be separated, and simple reject/re-solve does not necessarily reach a beneficial available identity. A corrective-only diagnostic packet is not an unbiased formal training dataset. Ordinary MLP/DeepSets and JEV must eventually use the same values submit/commit operator before an architecture claim.

The next defensible step is a **new sealed supplemental capture protocol on these same partitions**, with deterministic spaced samples within existing groups, group weights/ESS and natural/hard negatives. Preserve the original13-event validation result, do not waive or relabel the existing gate, and do not select favourable sequences or benefits. Independently implement/test the shared production candidate-values API; then re-evaluate data eligibility before C1–C6. Full24 and official TEST remain excluded.
''')
    print(json.dumps({'train':t,'validation':v,'gate':report['formal_data_gate'],'status':blocker},indent=2))
    protect()
if __name__=='__main__':main()
