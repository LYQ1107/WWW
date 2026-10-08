"""Full native proposal scan; bounded sparse error snapshots, no H32 expansion."""
import argparse,copy,gzip,json,time
from collections import Counter
from types import SimpleNamespace
import torch
from jev_phase8_common import *
from jev_phase8_opportunity import PrefixIdentityAnchors,classify_row

def scan(video,diagnostic=False):
    import random,numpy as np
    random.seed(20261008);np.random.seed(20261008);torch.manual_seed(20261008)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    config=json.loads(PREREG.read_text());allowed=config['train_videos']+config['validation_videos']
    assert video in allowed or(diagnostic and video==7),'heldout/official TEST not released'
    run=OUT/('diagnostic_opportunities'if diagnostic else'opportunity_scan')/f'video{video:02d}'
    if run.exists():raise RuntimeError('refusing reused scan output')
    run.mkdir(parents=True);start=binding();assert not start['worktree_dirty'],'commit executing source before new scientific run'
    save(run/'START_MANIFEST.json',{'binding':start,'argv':sys.argv,'role':'diagnostic_only'if diagnostic else'train'if video in config['train_videos']else'validation'})
    lab=native_lab(video);actor=FrozenOFFActor()
    save(run/'NATIVE_RNG_PROVENANCE.json',{'global_seed':20261008,'native_trajectory_seed':lab.seed_state.trajectory_rng_seed,'seed_key':list(lab.seed_key),'native_score_trajectory_policy':'unchanged branch-local Python RNG stored in every state clone'})
    from jev_phase7_native import AttributionResolver
    from jev_phase7_offline import IdentityEvaluator
    from jev_phase7_state import CompleteStateDigest
    from gtr.modeling.jev_state import count_track_history
    evaluator=IdentityEvaluator(video);anchors=PrefixIdentityAnchors();digest=CompleteStateDigest()
    resolution=AttributionResolver(lab,actor);lab.native_resolver=resolution;lab.engine.resolve_actions=resolution.resolve
    def choose(policy,feature,question,legal,off_action,context=None):
        c={k:v for k,v in(context or{}).items()if k!='tracker_state_before'}
        lab.decision_rows.append({'question':question,'action':off_action,'original_action':off_action,'off_action':off_action,
            'legal':list(legal),'feature_vector':feature.detach().cpu().tolist(),'context':c,'probabilities':{a:float(a==off_action)for a in legal}})
        return off_action
    lab.pilot.choose=choose
    def targets(payload):
        key=(video,int(payload['frame']),int(payload['view']));image=lab.pilot.image_for(lab.lookup,*key)
        assert int(image['frame_id'])==key[1]+1 and int(image['view_id'])==key[2]+1,'coordinate fallback not allowed'
        predictions=[];lab.original_append(predictions,payload,image,{r:r+1 for r in range(len(payload['pred_boxes']))})
        aligned=evaluator.align(predictions)
        return {r:aligned[(key[1],key[2],r)]['gt']for r in range(len(predictions))}
    seed_targets=targets(lab.seed_payload);anchors.update(lab.seed_state.association_history[-1]['assignments'],seed_targets,lab.seed_key[1])
    stats=Counter();snapshots=[];group_last={};group_index=Counter();hard_groups=[];pending=None;pending_rows=[];max_frame=max(k[1]for k in lab.keys)
    original_propose=lab.original_propose
    def propose(payload,state,**kw):
        nonlocal pending
        proposal=original_propose(payload,state,**kw)
        if state.reactivation_mode:return proposal
        key=(video,int(payload['frame']),int(payload['view']));gt=targets(payload)
        actions=lab.engine._actions_for_proposal(proposal,state)
        descriptors={r:classify_row(proposal.track_ids,proposal.scores.detach().cpu().numpy(),proposal.pairs,r,gt.get(r),actions[r],anchors,proposal.banned_edges)for r in actions}
        selected=[]
        for row,d in descriptors.items():
            if not d['strict_wrong_with_feasible_correct_candidate']:continue
            g=int(d['offline_GT']);last=group_last.get(g)
            if last is None or key[1]-last>32:group_index[g]+=1;hard_groups.append({'GT':g,'group':group_index[g],'start_frame':key[1]})
            first=last is None or key[1]-last>32;group_last[g]=key[1]
            d['event_group']=f'video{video:02d}_GT{g}_group{group_index[g]}'
            if first and len(snapshots)<config['bounded_snapshots']['max_corrective_groups_per_video']:selected.append(row)
        pending={'key':key,'gt':gt,'descriptors':descriptors,'proposal':proposal,'snapshot':None,'selected_rows':selected}
        if selected:
            snapshot={'key':key,'state':state.clone(),'metadata':copy.deepcopy(lab.metadata),'complete_state_fingerprint':digest(state,lab.metadata),
                'proposal':copy.deepcopy(proposal),'offline_prefix_identity':anchors.diagnostics(),'descriptors':copy.deepcopy(descriptors),
                'raw_score_sha256':__import__('hashlib').sha256(proposal.scores.detach().cpu().numpy().tobytes()).hexdigest(),'selection':'first error in chronological corrective group, no future utility used','binding':start}
            pending['snapshot']=snapshot
        return proposal
    lab.original_propose=propose
    natural=gzip.open(run/'natural_event_index.jsonl.gz','wt');hard=gzip.open(run/'hard_event_index.jsonl.gz','wt')
    def before(key,payload,state,kwargs,decisions):
        nonlocal pending_rows
        assert pending['key']==key;pending_rows=[]
        decisions_by_row={int(d['context']['detection_index']):d for d in decisions if d['question']=='MATCH_DECISION'}
        proposal=pending['proposal'];matrix=proposal.scores.detach().cpu().numpy()
        for row,d in pending['descriptors'].items():
            stats['MATCH_events']+=1;stats[d['bucket']]+=1
            stats['known_GT']+=d['offline_GT']is not None;stats['duplicate_GT_alias_events']+=d['duplicate_GT_alias_ambiguity']
            stats['strict_wrong_with_feasible_correct_candidate']+=d['strict_wrong_with_feasible_correct_candidate']
            stats['loose_majority_wrong_with_candidate']+=d['loose_majority_wrong_with_candidate']
            stats['candidate_available_known_rows']+=bool(d['correct_candidate_ids']);stats['global_feasible_known_rows']+=bool(d['feasible_pairs'])
            compact={k:v for k,v in d.items()if k not in ['feasible_pairs','row_candidate_order','unanchored_or_mixed_candidate_ids']}
            compact.update(key=list(key),row=row,unknown_candidate_count=len(d['unanchored_or_mixed_candidate_ids']))
            pending_rows.append(compact)
            if d['strict_wrong_with_feasible_correct_candidate']:
                full={**d,'key':list(key),'row':row,'candidate_track_ids':list(proposal.track_ids),'candidate_scores':matrix[row].tolist(),
                     'global_conflict':any(c==list(proposal.track_ids).index(t)and r!=row for t in d['correct_candidate_ids']for r,c in proposal.pairs.items()),
                     'native_corrective_action':'NOT_RUN','first_association':d['offline_GT']not in anchors.last_GT_frame,
                     'last_known_target_gap_frames':key[1]-anchors.last_GT_frame.get(d['offline_GT'],key[1])}
                hard.write(json.dumps(full,sort_keys=True)+'\n')
        if pending['snapshot']is not None:
            s=pending['snapshot'];s['decision_rows']=copy.deepcopy(decisions);s['factual_step_kwargs']=copy.deepcopy(kwargs)
            evidence={}
            for row in pending['selected_rows']:
                out=[];pc=proposal.pairs.get(row)
                for col,t in enumerate(proposal.track_ids):
                    m=lab.metadata.get(t,{});p=state.track_embeddings.get(t);obs=payload['reid_features'][row]
                    cosine=float(torch.nn.functional.cosine_similarity(obs.reshape(1,-1),p.reshape(1,-1)))if p is not None else 0.
                    out.append([float(matrix[row,col]),float(matrix[row,col]-(matrix[row,pc]if pc is not None else 0.)),float(col==pc),float(sum(c==col for rr,c in proposal.pairs.items()if rr!=row)),float(count_track_history(state.association_history,t)),float(len(state.memory.get(t,[]))),float(max(0,key[1]-m.get('last_seen',key[1]))),len(m.get('views',[]))/2.,float(t in state.active_ids),float(t in state.stale_ids),cosine,float(p is not None)])
                evidence[row]={'state_features':decisions_by_row[row]['feature_vector'],'candidate_evidence':out,'selected_row':row}
            s['online']=evidence;path=run/'snapshots'/f'CORRECTIVE_{len(snapshots):03d}.pth';path.parent.mkdir(exist_ok=True);torch.save(s,path)
            snapshots.append({'path':str(path),'sha256':sha(path),'key':list(key),'selected_rows':pending['selected_rows'],'groups':[pending['descriptors'][r]['event_group']for r in pending['selected_rows']]})
    def after(key,payload,state,result,kwargs,decisions):
        for r in pending_rows:
            r['actual_committed_id']=int(result['committed_track_ids'][r['row']]);natural.write(json.dumps(r,sort_keys=True)+'\n')
        anchors.update(result['committed_track_ids'],pending['gt'],key[1])
        if lab.steps_completed%128==0:save(run/'SCAN_PROGRESS.json',{'status':'RUNNING','key':list(key),'payloads':lab.steps_completed,'counts':dict(stats),'snapshots':len(snapshots)})
    try:result=lab.run(run/'factual',before_step=before,after_step=after)
    finally:natural.close();hard.close()
    save(run/'SCAN_RESULT.json',{'status':'COMPLETE','video':video,'role':'diagnostic_only'if diagnostic else'train'if video in config['train_videos']else'validation',
        'binding':start,'factual_policy':'GMT_OFF','counts':dict(stats),'factual':result,'factual_predictions_sha256':sha(result['predictions']),
        'natural_event_index_sha256':sha(run/'natural_event_index.jsonl.gz'),'hard_event_index_sha256':sha(run/'hard_event_index.jsonl.gz'),
        'bounded_snapshots':snapshots,'independent_error_groups':hard_groups,'final_anchors':anchors.diagnostics(),
        'global_feasibility_scope':'Oracle2 fixed-edge residual assignment; actual native current/future correction still NOT_RUN',
        'native_direct_candidate_interface':'NOT_YET_VERIFIED','H32_forks_started':False,'no_GT_future_in_actor':True,'max_frame':max_frame})
    save(run/'SCAN_PROGRESS.json',{'status':'COMPLETE','counts':dict(stats),'snapshots':len(snapshots)});protect()
    print(json.dumps({'status':'COMPLETE','video':video,'counts':dict(stats),'snapshots':len(snapshots)}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);p.add_argument('--diagnostic',action='store_true');a=p.parse_args();scan(a.video,a.diagnostic)
