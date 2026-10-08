"""Predeclared shared H64 MEMORY follow-up, preserving Phase VI sources."""
import argparse,copy,hashlib,json
import torch
from jev_phase7_common import *

def vector_hash(vector):
    return hashlib.sha256(vector.detach().cpu().numpy().tobytes()).hexdigest()if vector is not None else None

def run(video):
    assert json.loads((REPORTS/'PHASE7_MATCH_MECHANISM_GO_NO_GO.json').read_text())['attribution_status']=='COMPLETE'
    rollouts=isolate_phase6_imports()
    from jev_phase7_policies import MatchActor
    from jev_phase7_native import AttributionResolver
    original=Path('/home/liuyeqiang/WWW_jev_phase6_runtime/20261008/lifecycle_native')/f'video{video:02d}'
    source=[]
    for p in sorted((original/'labels').glob('MEMORY_*.json')):
        r=json.loads(p.read_text())
        if r['read_expected']and r['offline_GT']is not None:source.append((tuple(r['key']),p,r))
    selected=sorted(source)[:4]
    dest=OUT/'memory_followup'/f'video{video:02d}'
    if dest.exists():raise RuntimeError('refusing reused memory follow-up')
    dest.mkdir(parents=True);lab=rollouts.NativeReplayLab(video,'cuda:0',compact_context=True);actor=MatchActor('B2')
    original_resolve=lab.engine.resolve_actions;bind=binding();results=[]
    from jev_phase7_state import CompleteStateDigest
    complete_digest=CompleteStateDigest()
    from jev_phase7_offline import IdentityEvaluator
    from jev_phase6_offline_utility import consequence
    evaluator=IdentityEvaluator(video);baseline=json.loads((original/'baseline/result.json').read_text())
    factual=evaluator.align(json.loads(Path(baseline['predictions']).read_text()))
    for key,label_path,label in selected:
        path=original/'snapshots'/f'{label_path.stem}.pth';capture=torch.load(path,map_location='cpu');r=capture['record'];track=int(r['assigned_track_id'])
        prefix=capture['state'];target=label['offline_GT'];branches={}
        for action in ('CONTROL','WRITE_MEMORY','SKIP_MEMORY'):
            lab.engine.resolve_actions=original_resolve;resolver=AttributionResolver(lab,actor)
            lab.engine.resolve_actions=resolver.resolve;lab.native_resolver=resolver;reads=[];states=[];commits=[]
            def choose(policy,feature,question,legal,off_action,context=None):
                context={k:v for k,v in (context or {}).items()if k!='tracker_state_before'}
                d=actor.decide(feature,question,legal,off_action=off_action,context=context)if question=='MATCH_DECISION'else None
                original_action=d.committed_action if d else off_action
                current_key=(video,context['frame'],context['view'],question,context['detection_index'])
                actual=action if current_key==key and action!='CONTROL'else original_action
                assert actual in legal
                lab.decision_rows.append({'question':question,'action':actual,'original_action':original_action,'off_action':off_action,
                    'legal':list(legal),'feature_vector':feature.detach().cpu().tolist(),'context':context,
                    'probabilities':d.probabilities if d else {a:float(a==actual)for a in legal}})
                return actual
            lab.pilot.choose=choose
            def before(k,payload,state,kwargs,decisions):
                p=kwargs.get('reactivation_proposal')
                if p is not None and track in p.track_ids and track in state.stale_ids and track not in state.active_ids:
                    col=list(p.track_ids).index(track);resolution=resolver.resolve(payload,state,actions=kwargs['actions'],proposal=kwargs['proposal'])
                    query_rows=[row for row in range(len(payload['pred_boxes']))if resolution['existing_track_ids'].get(row)is None]
                    assert len(query_rows)==len(p.scores)
                    reads.append({'key':list(k),'candidate_ids':list(p.track_ids),'query_rows':query_rows,
                        'target_column_scores':p.scores[:,col].tolist(),'full_score_sha256':vector_hash(p.scores),
                        'prototype_sha256':vector_hash(state.reactivation_bank.get(track)),
                        'gallery_sha256':[vector_hash(x)for x in state.memory.get(track,[])]})
            def after(k,payload,state,result,kwargs,decisions):
                states.append({'key':list(k),'state_sha256':lab.state_fingerprint(state,lab.metadata),'complete_state_sha256':complete_digest(state,lab.metadata),
                    'prototype_sha256':vector_hash(state.reactivation_bank.get(track)),
                    'gallery_sha256':[vector_hash(x)for x in state.memory.get(track,[])]})
                commits.append({'key':list(k),'committed_ids':result['committed_track_ids'],'reactivation_assignments':kwargs.get('reactivation_assignments',{})})
                if k[1]>=key[1]+64 and k[2]==1:raise rollouts.EndEventHorizon()
            result=lab.run(dest/label_path.stem/action,prefix=prefix,start_key=key[:3],end_frame=key[1]+64,
                 metadata=capture['metadata'],before_step=before,after_step=after)
            rows=evaluator.align(json.loads(Path(result['predictions']).read_text()))
            if action=='CONTROL':assert all(v==factual[k]for k,v in rows.items())
            outcomes={str(h):consequence({k:v for k,v in rows.items()if k[0]<=key[1]+h},factual,key[1:3],target,h,prefix.next_id)for h in (8,16,32,64)}
            value={'result':result,'reads':reads,'states':states,'commits':commits,'outcomes':outcomes,'prediction_sha256':sha(result['predictions']),
                  'truncated':key[1]+64>max(k[1]for k in lab.keys),'first_read_frame':reads[0]['key'][1]if reads else None}
            branches[action]=value;save(dest/label_path.stem/action/'MEMORY_BRANCH_AUDIT.json',value)
        assert branches['CONTROL']['states']==branches['WRITE_MEMORY']['states']
        assert branches['CONTROL']['commits']==branches['WRITE_MEMORY']['commits']
        horizon_ties={str(h):abs(branches['WRITE_MEMORY']['outcomes'][str(h)]['utility']-branches['SKIP_MEMORY']['outcomes'][str(h)]['utility'])<1e-8 for h in (8,16,32,64)}
        record={'key':list(key),'question':'MEMORY_DECISION','offline_target_GT':target,'GT_or_future_policy_inputs':False,
            'source_label_sha256':sha(label_path),'source_snapshot_sha256':sha(path),'initial_state_fingerprint':lab.state_fingerprint(prefix,capture['metadata']),
            'branches':branches,'utility_ties':horizon_ties,'control_native_factual_replay':True,
            'sampling_scope':'first four known-target READ-enriched canonical sources by key; not population prevalence','binding':bind}
        results.append(record);save(dest/label_path.stem/'MEMORY_PAIRED_RESULT.json',record)
    save(dest/'MEMORY_FOLLOWUP_AUDIT.json',{'status':'COMPLETE','records':results,'binding':bind,'video':video,'shared_windows':[8,16,32,64]})
    protect();print(json.dumps({'video':video,'events':len(results),'H64_ties':sum(r['utility_ties']['64']for r in results)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=int,choices=[23,24],required=True);run(p.parse_args().video)
