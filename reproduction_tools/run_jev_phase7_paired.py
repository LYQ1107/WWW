"""Identical-prefix paired native interventions; common live B2 future policy."""
import argparse,copy,gzip,json,time
import torch
from jev_phase7_common import *

CONDITIONS=['CONTROL','GMT','FIXED','DYNAMIC','MLP','MLP_LN','BINARY','B2',
            'B2_VALIDATION_ONLY','B2_LEGACY_VALIDATOR','B2_ORIGINAL_LEGACY',
            'FIXED_EQUAL_CURRENT_BUDGET','DYNAMIC_EQUAL_CURRENT_BUDGET','MLP_EQUAL_CURRENT_BUDGET',
            'MLP_LN_EQUAL_CURRENT_BUDGET','BINARY_EQUAL_CURRENT_BUDGET','B2_EQUAL_CURRENT_BUDGET']

def main(video):
    rollouts=isolate_phase6_imports()
    from jev_phase7_policies import MatchActor,A,R,N
    from jev_phase7_native import AttributionResolver
    from run_jev_phase7_normalized import NormalizedActor
    lab=rollouts.NativeReplayLab(video,'cuda:0',compact_context=True)
    original_resolve=lab.engine.resolve_actions;continuation=MatchActor('B2')
    snapshots=sorted((OUT/'paired_snapshots'/f'video{video:02d}').glob('*.pth'))
    assert snapshots,'no factual B2 REASSOCIATE snapshots; do not substitute favourable captures'
    dest=OUT/'paired'/f'video{video:02d}'
    if dest.exists():raise RuntimeError('refusing reused paired experiment')
    dest.mkdir(parents=True);bind=binding();results=[]
    # Immediate proposals for every available capture; live horizons only first four.
    for index,path in enumerate(snapshots):
        capture=torch.load(path,map_location='cpu');key=tuple(capture['key']);prefix=capture['state'];metadata=capture['metadata']
        assert lab.state_fingerprint(prefix,metadata)==capture['fingerprint']
        for condition in CONDITIONS:
            name='B2'if condition.startswith('B2')or condition=='CONTROL'else condition.split('_EQUAL')[0]
            first=NormalizedActor('MLP')if name=='MLP_LN'else MatchActor(name)
            current=first;lab.engine.resolve_actions=original_resolve
            global_on=condition not in ('B2_VALIDATION_ONLY','B2_ORIGINAL_LEGACY')
            validator='LEGACY'if condition in ('B2_LEGACY_VALIDATOR','B2_ORIGINAL_LEGACY')else 'OWN'
            branch_dest=dest/f'event{index:02d}'/condition;branch_dest.parent.mkdir(parents=True,exist_ok=True)
            mechanism_path=branch_dest.parent/f'{condition}_mechanisms.jsonl.gz';writer=gzip.open(mechanism_path,'wt')
            resolver=AttributionResolver(lab,current,validator,global_on,writer=writer)
            lab.engine.resolve_actions=resolver.resolve;lab.native_resolver=resolver
            first_record=None;fingerprints={};commits=[];first_complete=False
            def choose(policy,feature,question,legal,off_action,context=None):
                context={k:v for k,v in (context or {}).items()if k!='tracker_state_before'}
                d=current.decide(feature,question,legal,off_action=off_action,context=context)if question=='MATCH_DECISION'else None
                action=d.committed_action if d else off_action
                lab.decision_rows.append({'question':question,'action':action,'original_action':action,'off_action':off_action,
                    'legal':list(legal),'feature_vector':feature.detach().cpu().tolist(),'context':context,
                    'probabilities':d.probabilities if d else {a:float(a==action)for a in legal}})
                return action
            lab.pilot.choose=choose
            if condition.endswith('_EQUAL_CURRENT_BUDGET'):
                native_resolve=resolver.resolve
                def equal_resolve(payload,state,*,actions=None,threshold=None,proposal=None):
                    actual=dict(actions)
                    if not first_complete:
                        target=sum(a==R for a in capture['mechanism']['first_actions'].values())
                        records={int(r['context']['detection_index']):r for r in lab.decision_rows if r['question']=='MATCH_DECISION'}
                        candidates=[r for r in proposal.pairs if len(proposal.track_ids)>1]
                        selected=set(sorted(candidates,key=lambda r:(-records[r]['probabilities'].get(R,0),r))[:target])
                        assert len(selected)==target
                        for row in candidates:
                            if row in selected:actual[row]=R
                            elif actual[row]==R:
                                f=torch.tensor(records[row]['feature_vector'])
                                actual[row]=current.decide(f,'MATCH_DECISION',[A,N],off_action=records[row]['off_action']).committed_action
                    return native_resolve(payload,state,actions=actual,threshold=threshold,proposal=proposal)
                lab.engine.resolve_actions=equal_resolve
            def after(step_key,payload,state,result,kwargs,decisions):
                nonlocal current,first_complete,first_record
                commits.append({'key':list(step_key),'committed_ids':result['committed_track_ids']})
                elapsed=step_key[1]-key[1]
                if not first_complete:
                    first_record=copy.deepcopy(resolver.last)
                    assert first_record['score_sha256']==capture['mechanism']['score_sha256']
                    current=continuation;resolver.actor=continuation;resolver.validator='OWN';resolver.global_reassociation=True
                    first_complete=True
                fingerprints[str(elapsed)]={'state_sha256':lab.state_fingerprint(state,lab.metadata),'key':list(step_key)}
                if index>=4 or elapsed>=32 and step_key[2]==1:raise rollouts.EndEventHorizon()
            stamp=time.perf_counter()
            try:
                result=lab.run(branch_dest,prefix=prefix,start_key=key,end_frame=key[1]+(32 if index<4 else 0),
                               metadata=metadata,after_step=after)
            finally:writer.close()
            record={'binding':bind,'condition':condition,'snapshot':str(path),'snapshot_sha256':sha(path),
                    'initial_state_fingerprint':capture['fingerprint'],'key':list(key),'first_mechanism':first_record,
                    'result':result,'fingerprints':fingerprints,'commits':commits,
                    'live_horizons':[8,16,32]if index<4 else [],'continuation_policy':'common live frozen B2',
                    'predictions_sha256':sha(result['predictions']),'mechanisms':str(mechanism_path),
                    'mechanisms_sha256':sha(mechanism_path),'wall_seconds':time.perf_counter()-stamp}
            save(branch_dest/'PAIRED_RESULT.json',record);results.append(record)
        save(dest/'progress.json',{'completed_events':index+1,'total_events':len(snapshots),'completed_branches':len(results)})
    # GT starts here, after native inference; actor module never sees it.
    from jev_phase7_offline import IdentityEvaluator
    from jev_phase6_offline_utility import consequence,prefix_identity
    evaluator=IdentityEvaluator(video)
    factual_predictions=json.loads((OUT/'closed_loop'/f'video{video:02d}/B2/tracking_predictions/jev.json').read_text())
    factual_rows=evaluator.align(factual_predictions);audited=[]
    for r in results:
        capture=torch.load(r['snapshot'],map_location='cpu');key=tuple(r['key']);prefix=capture['state']
        predictions=json.loads(Path(r['result']['predictions']).read_text());rows=evaluator.align(predictions)
        same_observations=all(k in factual_rows and value['bbox']==factual_rows[k]['bbox']and value['score']==factual_rows[k]['score']for k,value in rows.items())
        factual_ids_equal=all(value['id']==factual_rows[k]['id']for k,value in rows.items())
        if r['condition']in ('CONTROL','B2'):assert factual_ids_equal and same_observations
        target_row=next(int(row)for row,a in capture['mechanism']['first_actions'].items()if a==R)
        target=factual_rows.get((key[1],key[2],target_row),{}).get('gt')
        outcomes={str(h):consequence(rows,factual_rows,key,target,h,prefix.next_id)for h in r['live_horizons']}
        mapping,_=prefix_identity(factual_rows,key)
        candidates=capture['mechanism']['track_ids'];correct=[t for t in candidates if target is not None and mapping.get(t)==target]
        audited.append(dict(r,offline_GT=target,offline_correct_candidates=correct,GT_or_future_policy_inputs=False,
                            outcomes=outcomes,factual_ids_equal=factual_ids_equal,perception_geometry_equal=same_observations))
    for index in range(len(snapshots)):
        event=[r for r in audited if r['snapshot']==str(snapshots[index])]
        control=next(r for r in event if r['condition']=='CONTROL');b2=next(r for r in event if r['condition']=='B2')
        assert control['fingerprints']==b2['fingerprints'] and control['commits']==b2['commits']
    save(dest/'PAIRED_AUDIT.json',{'status':'COMPLETE','video':video,'binding':bind,'records':audited,
        'events':len(snapshots),'live_horizon_events':min(4,len(snapshots)),
        'exact_factual_prediction_replay':True,'CONTROL_B2_complete_mutable_state_parity':True,
        'all_branches_same_start':True,'official_test_read':False})
    protect();print(json.dumps({'video':video,'status':'PAIRED_COMPLETE','events':len(snapshots),'branches':len(results)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',required=True,type=int);main(p.parse_args().video)
