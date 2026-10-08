"""Portable bounded dataset plus preregistered support gate; retain quarantined labels."""
from collections import Counter,defaultdict
import copy,gzip,json,math
import numpy as np
from jev_phase7_common import *
from jev_phase7_miniset import validate,NativeCausalMiniSet,CANDIDATE_FIELDS,ACTIONS
from jev_phase7_offline import IdentityEvaluator


def quantiles(values):
    return dict(zip(['p0','p25','p50','p75','p90','p100'],map(float,np.quantile(values,[0,.25,.5,.75,.9,1]))))if values else None


def population_audit(video,factual,population):
    observed=sorted(factual.items());cursor=0;counts=defaultdict(Counter);report=Counter();by_key=defaultdict(list);events=[]
    for record in population:by_key[tuple(record['key'])].append(record)
    for key,records in sorted(by_key.items()):
        local=key[1:]
        while cursor<len(observed)and observed[cursor][0][:2]<local:
            _,r=observed[cursor]
            if r['gt']is not None:counts[r['id']][r['gt']]+=1
            cursor+=1
        mapping={t:c.most_common(1)[0][0]for t,c in counts.items()};known_correct_targets=set()
        for record in records:
            r=factual[(key[1],key[2],record['row'])];target=r['gt'];candidates=record['candidate_ids'];col=record['proposed_column']
            proposal=candidates[col]if col is not None else None;correct=[t for t in candidates if target is not None and mapping.get(t)==target]
            assessable=target is not None and proposal in mapping
            report['events']+=1;report['known_GT']+=target is not None;report['unknown_GT']+=target is None
            report['assessable_proposal']+=assessable;report['candidate_recall_lower_bound_count']+=bool(correct)
            report['proposal_correct']+=assessable and mapping[proposal]==target
            opportunity=assessable and mapping[proposal]!=target and bool(correct)
            report['proposal_wrong_with_correct_candidate']+=opportunity
            report['confirmed_candidate_miss']+=target is not None and not correct and all(t in mapping for t in candidates)
            report['candidate_identity_unanchored']+=target is not None and not correct and any(t not in mapping for t in candidates)
            report['legal_three_action']+=len(record['legal'])==3
            if correct:known_correct_targets.add(target)
            if opportunity:events.append({'key':list(key),'row':record['row'],'offline_GT':target,'proposal_identity':proposal,'correct_candidates':correct})
        # Candidate availability is an edge upper bound, not proof of joint Hungarian feasibility.
        report['payloads']+=1;report['distinct_GT_recall_edge_upper_bound']+=len(known_correct_targets)
    return {'counts':dict(report),'candidate_recall_lower_bound':report['candidate_recall_lower_bound_count']/max(1,report['known_GT']),
            'corrective_opportunity_events':events,'oracle_scope':'known prefix-identity candidate-edge coverage upper bound; global one-to-one/validator feasibility is not guaranteed',
            'source':'complete unscreened live-native B2 MATCH stream; GT mapping updated only from preceding payloads offline'}


def main():
    protect();roles={};manifest_files={};populations={};total_joint=[]
    for video,role in [(7,'train'),(6,'validation')]:
        root=OUT/'miniset_v1'/f'video{video:02d}';result=json.loads((root/'MINISET_RESULT.json').read_text());assert result['status']=='COMPLETE'
        collection=json.loads((root/'COLLECTION_MANIFEST.json').read_text());records=[]
        for source in result['records']:
            validate(source);r=copy.deepcopy(source);offline=r['offline_only']
            unanchored=offline['proposal_correct']is None or offline['candidate_oracle_uncertain']
            eligible=offline['informative']and not unanchored and not offline['censored_horizons']
            offline['effective_training_weight']=float(eligible)
            r['data_gate']={'record_eligible':eligible,'quarantine_reason':'UNANCHORED_PREFIX_IDENTITY'if unanchored else 'CENSORED'if offline['censored_horizons']else 'UTILITY_TIE'if not offline['informative']else None,
                           'global_fitting_gate_passed':False,'original_source_label_sha256':sha(root/'labels'/f'{r["event_id"]}.json')}
            records.append(r)
            if 'JOINT_REASSOCIATE'in r['branches']:
                o=offline['outcomes'];interactions={str(h):o['JOINT_REASSOCIATE'][str(h)]['utility']-o['REASSOCIATE'][str(h)]['utility']-o['OTHER_REASSOCIATE'][str(h)]['utility']+o['CONTROL'][str(h)]['utility']for h in (8,16,32)}
                total_joint.append({'event_id':r['event_id'],'interactions':interactions,'scope':'same target utility; a zero interaction here does not prove global edge additivity'})
        path=REPORTS/'miniset_v1'/f'{role}.jsonl.gz';path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('wb')as file,gzip.GzipFile(fileobj=file,mode='wb',mtime=0,filename='')as packed:
            for record in records:packed.write((json.dumps(record,sort_keys=True,allow_nan=False)+'\n').encode())
        loaded=NativeCausalMiniSet(path,role);assert len(loaded)==len(records)
        informative=[r for r in records if r['data_gate']['record_eligible']];winners=Counter(a for r in informative for a in r['offline_only']['best_actions'])
        margins=[r['offline_only']['utility_margin']for r in informative];opportunities=sum(r['offline_only']['feasible_correct_alternative_factual_wrong']for r in records)
        diverged=sum(any(b['commits']!=r['branches']['CONTROL']['commits']for a,b in r['branches'].items()if a in ACTIONS)for r in records)
        state_changed=sum(any(b['fingerprints']!=r['branches']['CONTROL']['fingerprints']for a,b in r['branches'].items()if a in ACTIONS)for r in records)
        gap_majority=sum(r['offline_only']['oracle_utility']-r['offline_only']['utility_vector']['ACCEPT_CURRENT']for r in informative)
        horizon_counts={str(h):sum(r['data_gate']['record_eligible']and len({round(r['offline_only']['outcomes'][a][str(h)]['utility'],8)for a in ACTIONS})>1 for r in records)for h in (8,16,32)}
        evaluator=IdentityEvaluator(video);factual=evaluator.align(json.loads(Path(result['factual']['predictions']).read_text()))
        with gzip.open(root/'population.jsonl.gz','rt')as stream:population=[json.loads(line)for line in stream]
        populations[str(video)]=population_audit(video,factual,population)
        roles[role]={'video':video,'records':len(records),'raw_unique_best':sum(r['offline_only']['informative']for r in records),
            'informative_eligible':len(informative),'unknown_target':sum(r['offline_only']['unknown_target']for r in records),
            'unanchored_prefix_quarantine':sum(r['data_gate']['quarantine_reason']=='UNANCHORED_PREFIX_IDENTITY'for r in records),
            'tie_count':sum(len(r['offline_only']['best_actions'])>1 for r in records),'tie_rate':sum(len(r['offline_only']['best_actions'])>1 for r in records)/max(1,len(records)),
            'winning_action_support':dict(winners),'effective_sample_size':float(len(informative)),
            'eligible_utility_margin_quantiles':quantiles(margins),'feasible_correct_alternative_factual_wrong':opportunities,
            'prediction_commit_divergence_events':diverged,'complete_mutable_state_divergence_events':state_changed,
            'H8_H16_H32_eligible_action_difference_counts':horizon_counts,'majority_ACCEPT_oracle_regret_sum_eligible':gap_majority,
            'all_control_factual_complete_state_and_commit_parity':True,'collection_manifest_sha256':sha(root/'COLLECTION_MANIFEST.json'),
            'record_sha256':sha(path),'population_events':collection['population_events'],'sampling_scope':'deterministic first four eligible payloads per exogenous bin; no population weighting from enrichment'}
        manifest_files[role]={'path':str(path.relative_to(ROOT)),'sha256':sha(path),'records':len(records),'sequence':video}
    enough_info=roles['train']['informative_eligible']>=12 and roles['validation']['informative_eligible']>=6
    train_supported={a for a,c in roles['train']['winning_action_support'].items()if c>=3}
    val_supported={a for a,c in roles['validation']['winning_action_support'].items()if c>=2}
    diverse=len(train_supported&val_supported)>=2
    opportunities=roles['train']['feasible_correct_alternative_factual_wrong']>=3 and roles['validation']['feasible_correct_alternative_factual_wrong']>=2
    passed=enough_info and diverse and opportunities
    assert not passed,'Gate changed unexpectedly; inspect before architecture fitting'
    audit={'status':'BLOCKED_MATCH_CORRECTIVE_CANDIDATE_AND_ACTION_SUPPORT','construction_status':'COMPLETE_VALIDATED_BOUNDED_MINISET',
        'roles':roles,'unscreened_native_population':populations,'joint_conflict_audit':total_joint,
        'gate':{'pass':passed,'minimum_informative_pass':enough_info,'two_supported_winning_actions_pass':diverse,
                'corrective_alternative_opportunities_pass':opportunities},
        'candidate_model_training_authorized_by_data_gate':False,'GT_or_future_in_online_inputs':False,'branch_independence_and_local_RNG_replay':True,
        'source_sha256':{str((OUT/'miniset_v1'/f'video{v:02d}/MINISET_RESULT.json')):sha(OUT/'miniset_v1'/f'video{v:02d}/MINISET_RESULT.json')for v in (7,6)},
        'WHAT_DID_WE_LEARN':'forced bad actions cause native divergence, but eligible labels only support ACCEPT and contain no feasible corrective alternative; early unanchored identity utilities are quarantined',
        'binding':binding(),'official_test_read':False}
    manifest={'status':'COMPLETE_DATASET_BLOCKED_FOR_FITTING','dataset':'Native Causal Lifecycle MiniSet v1','files':manifest_files,
        'candidate_fields':CANDIDATE_FIELDS,'schema_sha256':sha(ROOT/'docs/NATIVE_CAUSAL_MINISET_V1_SCHEMA.md'),
        'loader_sha256':sha(ROOT/'reproduction_tools/jev_phase7_miniset.py'),'audit_status':audit['status'],'binding':binding(),
        'MEMORY':'SEPARATE_IDENTIFIABILITY_AUDIT','REACTIVATION':'BLOCKED_NATIVE_RELATIVE_REACT_HOOK','official_test_read':False}
    save(REPORTS/'NATIVE_CAUSAL_MINISET_V1_MANIFEST.json',manifest);save(REPORTS/'NATIVE_CAUSAL_MINISET_V1_AUDIT.json',audit)
    save(REPORTS/'CAUSAL_DATASET_AUDIT.json',audit)
    protect();print(json.dumps({'status':audit['status'],'roles':roles,'gate':audit['gate']}))

if __name__=='__main__':main()
