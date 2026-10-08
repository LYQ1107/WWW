"""Bounded Oracle3/4 paired interventions from sealed real native snapshots."""
import argparse,copy,gzip,json,time,random
from pathlib import Path
import numpy as np
import torch
from jev_phase8_common import *
from jev_phase8_utility import effects

def run(video,limit=None,diagnostic=False):
    protect();random.seed(20261008);np.random.seed(20261008);torch.manual_seed(20261008)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    cfg=json.loads(PREREG.read_text());assert video in cfg['train_videos']+cfg['validation_videos']or(diagnostic and video==7)
    scan=OUT/('diagnostic_opportunities_v1'if diagnostic else'opportunity_scan_v1')/f'video{video:02d}'
    manifest=json.loads((scan/'SCAN_RESULT.json').read_text());chosen=manifest['bounded_snapshots'][:limit]
    out=OUT/('diagnostic_forks_v1'if diagnostic else'corrective_forks_v1')/f'video{video:02d}'
    assert not out.exists(),'fork outputs are immutable';out.mkdir(parents=True)
    source=binding();assert not source['worktree_dirty'];save(out/'START_MANIFEST.json',{'binding':source,'argv':sys.argv,'snapshot_selection':'all preregistered chronological representative snapshots; no future filtering','snapshots':chosen})
    from jev_phase7_offline import IdentityEvaluator
    from jev_phase7_native import AttributionResolver
    from jev_phase7_state import CompleteStateDigest
    from jev_phase8_candidate_submit import DirectCandidateResolver
    lab=native_lab(video);base_resolve=lab.engine.resolve_actions;base_propose=lab.original_propose
    evaluator=IdentityEvaluator(video);digest=CompleteStateDigest();factual={}
    with gzip.open(scan/'natural_event_index.jsonl.gz','rt')as f:
        for line in f:
            d=json.loads(line);factual[(d['key'][1],d['key'][2],d['row'])]=int(d['actual_committed_id'])
    results=[];raw=[]
    for si,record in enumerate(chosen):
        assert sha(record['path'])==record['sha256'];s=torch.load(record['path'],map_location='cpu');key=tuple(s['key'])
        assert digest(s['state'],s['metadata'])==s['complete_state_fingerprint']
        for row in record['selected_rows']:
            d=s['descriptors'][row];correct=sorted(d['correct_candidate_ids'],key=lambda t:(d['correct_candidate_ranks'][t],t))
            branches=[('CONTROL',None),('KEEP_ACCEPT',None),('REASSOCIATE',None),('START_NEW',None)]
            branches += [(f'CANDIDATE_{i}',dict(d['feasible_pairs'][t]))for i,t in enumerate(correct[:2])]
            # Multiple simultaneous targets in one real state, when present.
            if len(record['selected_rows'])>1:
                joint=dict(d['feasible_pairs'][correct[0]])
                other=[r for r in record['selected_rows']if r!=row]
                if all(r in joint and int(s['proposal'].track_ids[joint[r]])in s['descriptors'][r]['correct_candidate_ids']for r in other):branches.append(('JOINT_FEASIBLE',joint))
            event={'key':list(key),'row':row,'group':d['event_group'],'snapshot_sha256':record['sha256'],'offline_GT':d['offline_GT'],
                'correct_candidate_ids':correct,'branches':{},'binding':source};post={}
            for tag,pairs in branches:
                lab.engine.resolve_actions=base_resolve
                resolver=DirectCandidateResolver(lab,key,pairs)if pairs is not None else AttributionResolver(lab,FrozenOFFActor())
                lab.native_resolver=resolver;lab.engine.resolve_actions=resolver.resolve
                first=[True];native=[None];trace=[];begin=time.monotonic()
                def propose(payload,state,**kw):
                    p=base_propose(payload,state,**kw)
                    if not state.reactivation_mode and first[0]:
                        assert tuple(int(payload[k])for k in('video_id','frame','view'))==key
                        assert digest(state,lab.metadata)==s['complete_state_fingerprint'],'prefix field mismatch'
                        assert p.track_ids==s['proposal'].track_ids and p.pairs==s['proposal'].pairs
                        assert torch.equal(p.scores,s['proposal'].scores),'current score changed across identical-state forks'
                        first[0]=False
                    return p
                lab.original_propose=propose
                def choose(policy,feature,question,legal,off_action,context=None):
                    context=context or{};a=off_action;here=tuple(int(context.get(k,-1))for k in('video_id','frame','view'))==key and int(context.get('detection_index',-1))==row
                    if here and question=='MATCH_DECISION':
                        if tag=='KEEP_ACCEPT':a='ACCEPT_CURRENT'
                        elif tag=='REASSOCIATE':a='REASSOCIATE'
                        elif tag=='START_NEW':a='START_NEW'
                    if here and tag=='START_NEW'and question=='REACTIVATION_DECISION':a='START_NEW'
                    assert a in legal
                    lab.decision_rows.append({'question':question,'action':a,'original_action':off_action,'off_action':off_action,'legal':list(legal),
                        'feature_vector':feature.detach().cpu().tolist(),'context':{k:v for k,v in context.items()if k!='tracker_state_before'},'probabilities':{v:float(v==a)for v in legal}})
                    return a
                lab.pilot.choose=choose
                def after(k,payload,state,result,kwargs,decisions):
                    trace.append({'key':list(k),'ids':dict(result['committed_track_ids']),'fingerprint':digest(state,lab.metadata),
                        'rng_calls':state.trajectory_rng_calls,'next_id':state.next_id})
                    if k==key:native[0]=copy.deepcopy(resolver.last)
                output=out/f'snapshot{si:03d}_row{row:03d}'/tag
                result=lab.run(output,prefix=s['state'],start_key=key,end_frame=min(key[1]+31,manifest['max_frame']),metadata=s['metadata'],after_step=after)
                aligned=evaluator.align(json.loads(Path(result['predictions']).read_text()));post[tag]=trace
                immediate=aligned[(key[1],key[2],row)];mapping={int(t):int(g)for t,g in s['offline_prefix_identity']['reliable'].items()}
                value={'actual_committed_id':immediate['id'],'immediate_anchored_correct':mapping.get(immediate['id'])==d['offline_GT'],
                    'desired_candidate_committed':immediate['id']in correct,'H32_complete':manifest['max_frame']>=key[1]+31,
                    'new_birth_semantics_vetoed_same_key_bank_reactivation':tag=='START_NEW','native':native[0],
                    'horizons':{str(h):effects(aligned,s['offline_prefix_identity'],key[1],h,{d['offline_GT']})for h in(8,16,32)},
                    'elapsed_seconds':time.monotonic()-begin,'prediction_sha256':sha(result['predictions']),'post_state_trace_sha256':None}
                save(output/'COMPLETE_STATE_TRACE.json',trace);value['post_state_trace_sha256']=sha(output/'COMPLETE_STATE_TRACE.json')
                save(output/'EFFECTS.json',value);event['branches'][tag]=value;raw.append({'path':str(output/'EFFECTS.json'),'sha256':sha(output/'EFFECTS.json')})
            control=event['branches']['CONTROL'];keep=event['branches']['KEEP_ACCEPT']
            parity=post['CONTROL']==post['KEEP_ACCEPT'];assert parity,'CONTROL vs explicit factual ACCEPT complete-state/RNG mismatch'
            factual_parity=all(factual[(t['key'][1],t['key'][2],int(r))]==int(i)for t in post['CONTROL']for r,i in t['ids'].items())
            assert factual_parity,'CONTROL differs from the full factual native stream'
            event['CONTROL_KEEP_full_field_parity']=parity;event['CONTROL_factual_all_committed_ids_parity']=factual_parity
            event['verified_corrective_branches']=[]
            for tag,b in event['branches'].items():
                if tag in('CONTROL','KEEP_ACCEPT'):continue
                for h in(8,16,32):
                    e=b['horizons'][str(h)];c=control['horizons'][str(h)]
                    e['delta_utility']=e['utility']-c['utility'];e['delta_utility_birth_zero']=e['utility_birth_zero']-c['utility_birth_zero']
                    e['delta_target_wrong']=e['target_counts'].get('wrong',0)-c['target_counts'].get('wrong',0)
                    e['delta_wrong_duration_camera_frames']=e['wrong_identity_duration_camera_frames']-c['wrong_identity_duration_camera_frames']
                if b['immediate_anchored_correct']and b['H32_complete']and b['horizons']['32']['delta_utility']>0 and b['horizons']['32']['delta_utility_birth_zero']>0:event['verified_corrective_branches'].append(tag)
            results.append(event);save(out/'FORK_PROGRESS.json',{'status':'RUNNING','completed_events':len(results),'snapshots':len(chosen),'last_key':list(key),'verified_events':sum(bool(e['verified_corrective_branches'])for e in results)})
            save(out/'FORK_RESULT.partial.json',{'events':results});print(json.dumps({'video':video,'events':len(results),'key':key,'row':row,'verified':event['verified_corrective_branches']}),flush=True)
    save(out/'FORK_RESULT.json',{'status':'COMPLETE','video':video,'diagnostic_only':diagnostic,'binding':source,'events':results,'raw_effects':raw,
        'native_scope':'actual production MATCH helper on real proposals; full-state mutated replay uses existing native engine; deployed sliding-inference end-to-end parity NOT_VERIFIED',
        'selection_limit':limit,'H32_not_expanded_for_all_events':True,'no_gt_in_live_actor':True})
    save(out/'FORK_PROGRESS.json',{'status':'COMPLETE','completed_events':len(results),'verified_events':sum(bool(e['verified_corrective_branches'])for e in results)})
    protect()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);p.add_argument('--limit',type=int);p.add_argument('--diagnostic',action='store_true');a=p.parse_args();run(a.video,a.limit,a.diagnostic)
