"""Aggregate every declared P0 run without treating overlapping windows as samples."""
import collections
import gzip
from jev_phase16_common import *

def joined_consequences(path):
    pending={};groups=collections.defaultdict(collections.Counter);counts=collections.Counter()
    evidence_loss_risk=collections.Counter();examples=[]
    with gzip.open(path,'rt') as stream:
        for line in stream:
            item=json.loads(line);key=tuple(item['key']),item['row']
            if item.get('record_type')!='COMMIT':
                pending[key]=item
                if item['high_risk'] and item['current_GT_OFFLINE_ONLY'] is not None:
                    for identity in item['owner_anchored_summary_hidden_IDs']:
                        score=item['raw_and_summary_scores'][str(identity)]
                        evidence_loss_risk['owner_hidden_options']+=1
                        evidence_loss_risk['crosses_native075_support_threshold']+=score['raw_target']>=.75 and score['four_summary']<.75
                        evidence_loss_risk['raw_target_margin_positive']+=score['raw_target']>score['raw_foreign']
                        evidence_loss_risk['raw_target_margin_ge002']+=score['raw_target']-score['raw_foreign']>=.02
                    if item['owner_anchored_summary_hidden_IDs'] and len(examples)<4:
                        examples.append(dict(key=item['key'],row=item['row'],GT_OFFLINE_ONLY=item['current_GT_OFFLINE_ONLY'],
                            selected_ID=item['selected_ID'],owner_anchored_summary_hidden_IDs=item['owner_anchored_summary_hidden_IDs'],
                            scores={str(i):item['raw_and_summary_scores'][str(i)] for i in item['owner_anchored_summary_hidden_IDs']}))
                continue
            query=pending.pop(key);gt=query['current_GT_OFFLINE_ONLY'];identity=item['committed_ID']
            prior=query['raw_and_summary_scores'].get(str(identity));owner=prior['owner_GT_OFFLINE_ONLY'] if prior else None
            observed_wrong=gt is not None and owner is not None and owner!=gt
            new_mixing=gt is not None and prior is not None and prior['known_GT_count']>0 and prior['target_count']==0
            birth_proxy=item['action']=='START_NEW' and bool(query['raw_target_content_IDs'])
            false_split=item['action']=='START_NEW' and bool(query['legal_pure_correct_IDs'])
            counts['commits']+=1;counts['START_NEW']+=item['action']=='START_NEW'
            counts['REACTIVATE']+=item['action']=='REACTIVATE'
            counts['anchored_owner_wrong_observations']+=observed_wrong
            counts['new_cross_GT_mixing']+=new_mixing
            counts['previously_observed_GT_birth_proxy']+=birth_proxy
            counts['false_split_birth_with_legal_pure_correct_ID']+=false_split
            for flag,present in query['flags'].items():
                if not present:continue
                groups[flag]['commits']+=1;groups[flag]['wrong_owner']+=observed_wrong
                groups[flag]['new_mix']+=new_mixing;groups[flag]['START_NEW']+=item['action']=='START_NEW'
                groups[flag]['birth_proxy']+=birth_proxy;groups[flag]['false_split_birth']+=false_split
            # Multiple MATCH/REACT query observations can share one final
            # commit. Drop earlier payloads, retaining only the latest one.
            for old in [k for k in pending if k[0]<key[0]]:del pending[old]
    return dict(counts=dict(counts),flag_action_correlations={k:dict(v) for k,v in groups.items()},
        high_risk_owner_hidden_diagnostics=dict(evidence_loss_risk),examples=examples,
        owner_definition='immutable earliest>=3/.8 pure past anchor; never relabel a mixture as a wrong WHO candidate',
        stage_scope='final native action joined to its last actual MATCH or REACT query')

def main():
    protect();protocol=read(REPORTS/'PREREGISTRATION.json');inputs=[];cases=[];groups={}
    expected=[('full',r['policy'],r['video']) for r in protocol['P0']['full_replays']]
    expected += [('windows',p,v) for p in ['v1','v2','v3'] for v in TRAIN]
    for scope,policy,video in expected:
        path=OUT/'P0_r2'/scope/policy/f'video{video:02d}/RESULT.json';result=read(path)
        assert result['status']=='COMPLETE';inputs.append(ref(path))
        group_key=f'{scope}_{policy}_'+('TRAIN' if video in TRAIN else 'DEV')
        group=groups.setdefault(group_key,dict(counts=collections.Counter(),flags=collections.Counter(),
            primary=collections.Counter(),clusters=collections.defaultdict(set),action=collections.Counter(),cases=0))
        for item in result['cases']:
            assert scope!='windows' or item['parity']=='ALL_XV_BRANCH_IDS_EXACT'
            joined=joined_consequences(item['queries']['path']);summary=item['summary']
            group['counts'].update(summary['counts']);group['flags'].update(summary['multi_label_high_risk'])
            group['primary'].update(summary['primary_high_risk']);group['action'].update(joined['counts']);group['cases']+=1
            for name,values in summary['clusters'].items():
                group['clusters'][name].update(tuple(x) for x in values if x[0]!=14)
            cases.append(dict(scope=scope,policy=policy,video=video,tag=item['tag'],source=ref(path),
                query_trace=item['queries'],parity=item['parity'],summary=summary,consequences=joined))
    compact={}
    for name,g in groups.items():
        n=g['counts']['current_GT_known'];risk=g['counts']['high_risk_GT_known_queries']
        compact[name]=dict(counts=dict(g['counts']),high_risk_primary=dict(g['primary']),
            high_risk_multi_label=dict(g['flags']),native_action_consequences=dict(g['action']),cases=g['cases'],
            cluster_counts={k:len(v) for k,v in g['clusters'].items()},
            observed_correct_content_rate=g['counts']['observed_correct_content_exists']/n if n else None,
            anchored_owner_content_rate=g['counts']['anchored_correct_owner_exists']/n if n else None,
            selected_correct_on_GT_known_rate=g['counts']['selected_certified_correct']/n if n else None,
            high_risk_GT_known_queries=risk,query_counts_are_not_independent_clusters=True)
    primary=groups['full_v3_TRAIN'];clusters=primary['clusters']['owner_anchored_summary_loss']
    high_risk_clusters=primary['clusters']['high_risk_owner_anchored_summary_loss']
    # Gate uses TRAIN only, exactly the preregistered raw threshold/gap. The
    # diagnostic gate does not establish correct retrieval or safe recovery.
    candidate=P0_GO=len(clusters)>=8
    source=binding(inputs=inputs,evaluator='all raw query traces and final actual native commits',
        scope='full TRAIN primary v3 plus all XV policy/window/DEV diagnostics; no SGD or candidate modification')
    common=dict(status='COMPLETE',binding=source,namespace='P0_r2',
        cases=len(cases),whole_native_runs=19,H32_branches=72,
        early6_engineering_windows_not_added_as_independent_samples=True,
        unknown_is_not_wrong=True,raw_person_content_is_not_correct_Global_ID=True,
        taxonomy_has_overlapping_mechanisms=True,
        actual_REACT=dict(learned=False,cosine_slots=[0,1,2],DEFER=.75,
            latent_four_summary_has_other_camera_slot_not_used_by_native_REACT=True))
    save(REPORTS/'EVIDENCE_AVAILABILITY_AUDIT.json',dict(**common,groups=compact,
        individual_cases=cases,
        denominator_scope='all MATCH/REACT queries separately; final actions separately; counterfactual windows overlap'))
    save(REPORTS/'RAW_GALLERY_RECOVERABILITY.json',dict(**common,primary_TRAIN_v3=compact['full_v3_TRAIN'],
        strict_eligible_owner_summary_loss_clusters=len(clusters),
        strict_eligible_high_risk_owner_summary_loss_clusters=len(high_risk_clusters),
        local_evidence_person_match_does_not_resolve_global_identity_ownership=True,
        prototype_trial_eligible=candidate,safe_recovery_proven=False,
        extra_tokens_need_paired_positive_and_false_activation_validation=True))
    save(REPORTS/'CANDIDATE_FAILURE_TAXONOMY.json',dict(**common,groups={k:dict(primary=v['high_risk_primary'],
        overlapping_flags=v['high_risk_multi_label'],actions=v['native_action_consequences']) for k,v in compact.items()},
        primary_precedence=['E','C','D','F','B','A','G'],
        A_scope='absence of observed target content, not proof that UNKNOWN past observations are another person',
        F_scope='raw contains both target and another observed GT; no automatic wrong label',
        B_scope='same target-containing ID raw-vs-four-summary cosine gap; not a demonstrated decision benefit'))
    save(REPORTS/'PHASE16_P0_GO_NO_GO.json',dict(status='GO_BOUNDED_FROZEN_RETRIEVAL_TRIAL' if P0_GO else 'NO_GO_PROTOTYPE_EXTENSION',
        binding=source,P0_complete=True,P1_frozen_prototype_trial_qualified=P0_GO,
        strict_TRAIN_owner_anchored_clusters=len(clusters),high_risk_clusters=len(high_risk_clusters),
        safe_recovery_proven=False,new_network_training_qualified=False,formal_training_qualified=False,
        decision='Test bounded four-slot GT-free segment evidence, then causal memory factor' if P0_GO else
            'Do not add prototypes without identifiable past evidence; inspect candidate lifecycle and information insufficiency',
        false_activation_and_full_closed_loop_still_required=True))
    (ROOT/'docs/JEV_PHASE16_EVIDENCE_RECOVERY.md').write_text(
        '# Phase XVI evidence availability and recovery\n\n'+
        'The original raw Gallery and hits remain unchanged. Annotation is an offline observer only. '
        'A person appearing in an ID history is different from that Global ID having a certified past owner. '
        'The owner anchor is established only at the earliest past moment with >=3 known observations, '
        '>=0.8 coverage and one GT, then remains immutable. Histories mixed before such an anchor remain unanchored.\n\n'+
        'P0 includes19 complete native replays and72 H32 branches at all24 original TRAIN prefixes under all three XV checkpoints. '
        'The additional six early provenance windows are retained engineering qualification, not independent evidence. '
        'A-G flags overlap; the primary category precedence is E/C/D/F/B/A/G. Unknown observations cannot be labelled wrong. '
        'A denotes no observed target content, with UNKNOWN-history ambiguity disclosed.\n\n'+
        'Default REACT is untrained: actual Bank IDs are offered only after MATCH DEFER, then '
        'max(last, global mean, own-camera mean) cosine is compared through the same one-to-one solver with0.75 terminal. '
        'The fourth other-camera token is used by the WHO model but excluded from this fallback. '
        'No Bank candidate or detection is fabricated. Every query logs actual frozen logits and both summary score scopes.\n\n'+
        'Primary V3 TRAIN diagnostic:\n\n'+json.dumps(compact['full_v3_TRAIN'],ensure_ascii=False,indent=2)+
        '\n\nBounded retrieval trial eligible: '+str(P0_GO)+
        '. This is an information-availability gate; it is not safe-recovery or learned-JEV success. '
        'Continuous cosine gaps need positive-retrieval and false-activation controls before native benefit can be claimed. '
        'Development results never set training labels, thresholds or eligibility.\n')
    print('PHASE16_P0_COMPLETE',compact['full_v3_TRAIN'],'prototype_trial',P0_GO,flush=True)

if __name__=='__main__':main()
