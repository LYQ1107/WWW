"""Bounded full-video native closed loops with segmented, auditable commits."""
import argparse,copy,gzip,json,time
import torch
from jev_phase7_common import *

def run(condition,video,device,tag=None,validator='OWN',global_reassociation=True,budget=None,max_payloads=None,capture=False):
    rollouts=isolate_phase6_imports()
    from jev_phase7_policies import MatchActor
    from jev_phase7_native import AttributionResolver
    actor=MatchActor(condition);tag=tag or condition
    dest=OUT/'closed_loop'/f'video{video:02d}'/tag
    if dest.exists():raise RuntimeError('refusing reused closed-loop output')
    lab=rollouts.NativeReplayLab(video,device,compact_context=True)
    journal=dest.parent/f'{tag}_mechanisms.jsonl.gz';journal.parent.mkdir(parents=True,exist_ok=True)
    writer=gzip.open(journal,'wt');resolver=AttributionResolver(lab,actor,validator,global_reassociation,budget,writer)
    lab.native_resolver=resolver;lab.engine.resolve_actions=resolver.resolve
    controller_seconds=0.;captures=[];commits=[];first_fp=lab.state_fingerprint(lab.seed_state,lab.seed_metadata)
    def choose(policy,feature,question,legal,off_action,context=None):
        nonlocal controller_seconds
        context={k:v for k,v in (context or {}).items()if k!='tracker_state_before'}
        stamp=time.perf_counter()
        if question=='MATCH_DECISION':
            result=actor.decide(feature,question,legal,off_action=off_action,context=context)
            action=result.committed_action;probabilities=result.probabilities
        else:action=off_action;probabilities={a:float(a==action)for a in legal}
        controller_seconds+=time.perf_counter()-stamp
        lab.decision_rows.append({'question':question,'action':action,'original_action':action,'off_action':off_action,
            'legal':list(legal),'feature_vector':feature.detach().cpu().tolist(),'context':context,'probabilities':probabilities})
        return action
    lab.pilot.choose=choose
    def before(key,payload,state,kwargs,decisions):
        if capture and len(captures)<12 and 'REASSOCIATE' in resolver.last['first_actions'].values():
            snap={'key':key,'state':state.clone(),'metadata':copy.deepcopy(lab.metadata),
                  'fingerprint':lab.state_fingerprint(state,lab.metadata),'mechanism':resolver.last}
            snapdir=OUT/'paired_snapshots'/f'video{video:02d}';snapdir.mkdir(parents=True,exist_ok=True)
            path=snapdir/f'{len(captures):02d}.pth';torch.save(snap,path)
            captures.append({'path':str(path),'sha256':sha(path),'key':list(key),'fingerprint':snap['fingerprint']})
    def after(key,payload,state,result,kwargs,decisions):
        commits.append({'key':list(key),'committed_ids':result['committed_track_ids'],
                        'memory_actions':kwargs.get('memory_actions',{}),'reactivation_assignments':kwargs.get('reactivation_assignments',{})})
        if len(commits)%64==0:
            segment=dest/'committed_segments'/f'{len(commits)//64:04d}.json'
            save(segment,commits[-64:]);writer.flush()
        if max_payloads is not None and lab.steps_completed>=max_payloads:raise rollouts.EndEventHorizon()
    start=time.perf_counter()
    try:
        result=lab.run(dest,before_step=before,after_step=after)
    finally:writer.close()
    wall=time.perf_counter()-start
    save(dest/'committed_tail.json',commits[(len(commits)//64)*64:])
    metrics=None;prepared=evaluated=None
    if max_payloads is None:
        dataset=lab.pilot.prepare_eval_dataset(lab.subset)
        prepared,evaluated=lab.pilot.run_eval('jev',Path(result['predictions']),dataset)
        metrics=lab.pilot.extract_metrics(evaluated)
    summary={'status':'COMPLETE'if max_payloads is None else 'SMOKE_ONLY','binding':binding(),'condition':condition,'tag':tag,'video':video,
             'metrics':metrics,'result':result,'mechanism_counts':dict(resolver.stats),'controller_seconds':controller_seconds,
             'wall_seconds_excluding_eval':wall,'predictions_sha256':sha(result['predictions']),
             'mechanisms':str(journal),'mechanisms_sha256':sha(journal),'captures':captures,
             'initial_state_fingerprint':first_fp,'budget_target':budget,'budget_achieved':resolver.trigger_rows,
             'budget_exact':budget is None or budget==resolver.trigger_rows,
             'prepared':str(prepared)if prepared else None,'evaluation':str(evaluated)if evaluated else None,
             'validator':validator,'global_reassociation':global_reassociation,
             'complete_video':max_payloads is None,'official_test_read':False}
    save(dest/'result.json',summary);protect()
    print(json.dumps({'tag':tag,'video':video,'status':summary['status'],'metrics':metrics,'reassociate':resolver.trigger_rows}))
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--condition',required=True,choices=['GMT','FIXED','DYNAMIC','MLP','BINARY','B2'])
    p.add_argument('--video',required=True,type=int);p.add_argument('--device',default='cuda:0');p.add_argument('--tag')
    p.add_argument('--validator',choices=['OWN','LEGACY'],default='OWN');p.add_argument('--no-global',action='store_true')
    p.add_argument('--budget',type=int);p.add_argument('--max-payloads',type=int);p.add_argument('--capture',action='store_true')
    a=p.parse_args();run(a.condition,a.video,a.device,a.tag,a.validator,not a.no_global,a.budget,a.max_payloads,a.capture)
