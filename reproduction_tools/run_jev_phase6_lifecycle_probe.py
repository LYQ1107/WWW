"""Bounded canonical lifecycle coverage and live counterfactual supervision."""
import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import random

import torch

from jev_phase6_common import OUT,REPORTS,save,sha,new_output,protect_anchor
from jev_phase6_rollouts import NativeReplayLab,EndEventHorizon,event_key
from jev_phase6_offline_utility import OfflineIdentityAudit,prefix_identity,consequence
from gtr.modeling.jev_lifecycle import memory_features,reactivation_features,MEMORY_FIELDS,REACT_FIELDS


NATIVE_CONTRACT=False


def probe_root(video):return OUT/f"{'lifecycle_native' if NATIVE_CONTRACT else 'lifecycle'}/video{video:02d}"


def canonical_row(lab,row):
    c=row['context']; track=int(c['track_id']); identity=lab.metadata.get(track,{})
    if row['question']=='MEMORY_DECISION':
        values=memory_features(lab.current,c['detection_index'],lab.prefix,{'track_id':track,'identity':identity})
    else:
        values=reactivation_features(c['frame'],c['view'],lab.prefix,identity,c['candidate_track_ids'],c['candidate_scores'],track)
    return {'key':list(event_key(row)),'question':row['question'],'feature_vector':values.tolist(),
            'legal_actions':['WRITE_MEMORY','SKIP_MEMORY'] if row['question']=='MEMORY_DECISION' else ['REACTIVATE_OLD','START_NEW'],
            'assigned_track_id':track,'online_context':{k:v for k,v in c.items() if k!='tracker_state_before'},
            'feature_contract':'relative_online_lifecycle_v1','GT_or_future_inputs':False}


def collect(video,device):
    lab=NativeReplayLab(video,device,native_match_validation=NATIVE_CONTRACT,record_transitions=NATIVE_CONTRACT); root=probe_root(video);root.mkdir(parents=True,exist_ok=True)
    snapshots=root/'snapshots';snapshots.mkdir(exist_ok=True)
    cap=32 if video==24 else 16; rng=random.Random(20261008+video)
    react_seen=0; reservoir=[]; writes=[]; first_read={}; reads=[]; canonical=[]
    def before(key,payload,state,kwargs,rows):
        nonlocal react_seen
        proposal=kwargs.get('reactivation_proposal')
        if proposal is not None:
            ids=[int(t) for t in proposal.track_ids]
            reads.append({'key':list(key),'candidate_ids':ids,'score_matrix_sha256':__import__('hashlib').sha256(proposal.scores.detach().cpu().numpy().tobytes()).hexdigest()})
            for track in ids:
                if track in state.stale_ids and track not in state.active_ids:first_read.setdefault(track,key)
        for row in rows:
            if row['question']=='MEMORY_DECISION' and row['action']=='WRITE_MEMORY':
                writes.append({'key':list(event_key(row)),'track_id':int(row['context']['track_id'])})
            if row['question']!='REACTIVATION_DECISION':continue
            record=canonical_row(lab,row);canonical.append(record);react_seen+=1
            slot=react_seen-1 if react_seen<=cap else rng.randrange(react_seen)
            if slot<cap:
                captured={'state':lab.prefix.clone(),'metadata':copy.deepcopy(lab.metadata),'record':record}
                if slot<len(reservoir):reservoir[slot]=captured
                else:reservoir.append(captured)
    baseline=lab.run(root/'baseline',before_step=before)
    final_state_sha=lab.state_fingerprint(lab.prefix,lab.metadata) if NATIVE_CONTRACT else None
    save(root/'bank_reads.json',reads)
    with (root/'canonical_reactivation.jsonl').open('w') as f:
        for r in canonical:f.write(json.dumps(r,sort_keys=True)+'\n')
    # Selection uses the write/read trace only, never GT or outcome labels.
    selected=[]
    for track,key in sorted(first_read.items(),key=lambda item:(item[1],item[0])):
        prior=[w for w in writes if w['track_id']==track and tuple(w['key'][:3])<key]
        if prior:selected.append({**prior[-1],'read_expected':True,'baseline_first_read':list(key)})
        if len(selected)==8:break
    unread=[w for w in writes if w['track_id'] not in first_read and w['key'][1]>=lab.keys[-1][1]-32]
    seen=set()
    for w in reversed(unread):
        if w['track_id'] in seen:continue
        seen.add(w['track_id']);selected.append({**w,'read_expected':False,'baseline_first_read':None})
        if len(seen)==2:break
    selected_keys={tuple(w['key']):w for w in selected};mem=[]
    def capture(feature,question,legal,action,context):
        key=(video,context['frame'],context['view'],question,context['detection_index'])
        if key in selected_keys:
            row={'question':question,'context':context}
            mem.append({'state':lab.prefix.clone(),'metadata':copy.deepcopy(lab.metadata),
                        'record':{**canonical_row(lab,row),**selected_keys[key]}})
        return action
    if NATIVE_CONTRACT:
        reconstructed,state=lab.rebuild_memory_prefixes(selected)
        replay_sha=lab.state_fingerprint(state,lab.metadata)
        if replay_sha!=final_state_sha:raise AssertionError('journal replay changed final native mutable state or metadata')
        for c in reconstructed:
            lab.prefix=c['state'];lab.metadata=c['metadata'];lab.current=lab.cache.load(*c['selection']['key'][:3])
            mem.append({'state':c['state'],'metadata':c['metadata'],
                'record':{**canonical_row(lab,{'question':'MEMORY_DECISION','context':c['context']}),**c['selection']}})
        save(root/'JOURNAL_RECONSTRUCTION.json',{'status':'PASS','final_native_state_and_metadata_sha256':final_state_sha,
            'reconstructed_sha256':replay_sha,'all_factual_commits_identical':True,'future_counterfactual_policy_cached':False,
            'journal_files':{str(p.relative_to(root)):sha(p) for p in lab.journal_files}})
    else:
        repeated=lab.run(root/'memory_capture',intervention=capture)
        if sha(repeated['predictions'])!=sha(baseline['predictions']):raise AssertionError('metadata/capture changes the native B2 policy')
    for kind,captures in [('MEMORY',mem),('REACT',reservoir)]:
        for index,c in enumerate(sorted(captures,key=lambda c:tuple(c['record']['key']))):
            torch.save(c,snapshots/f'{kind}_{index:03d}.pth')
    gt=OfflineIdentityAudit(video); aligned=gt.align(json.loads(Path(baseline['predictions']).read_text()))
    space=[]
    for r in canonical:
        k=r['key']; local=(k[1],k[2],k[4]); target=aligned[local]['gt']; mapping,_=prefix_identity(aligned,(k[1],k[2]))
        c=r['online_context']; ids=c['candidate_track_ids']; scores=c['candidate_scores']; ranked=sorted(zip(ids,scores),key=lambda x:-x[1]);valid=[t for t,s in ranked if s>c['bank_threshold']]
        space.append({'key':k,'GT':target,'assigned_id':r['assigned_track_id'],'assigned_prefix_GT':mapping.get(r['assigned_track_id']),
                      'candidate_count':len(ids),'valid_stale_count':len(valid),'top1_id':ranked[0][0] if ranked else None,
                      'top1_prefix_GT':mapping.get(ranked[0][0]) if ranked else None,
                      'alternate_correct':target is not None and any(mapping.get(t)==target for t in valid if t!=ranked[0][0]),
                      'top1_wrong':target is not None and bool(ranked) and ranked[0][0] in mapping and mapping[ranked[0][0]]!=target,
                      'assigned_wrong':target is not None and r['assigned_track_id'] in mapping and mapping[r['assigned_track_id']]!=target})
    save(root/'ACTION_SPACE.json',{'status':'COMPLETE','events':space,'GT_only_offline':True})
    manifest={'status':'COMPLETE','video':video,'canonical_reactivation_events':react_seen,'sampled_reactivation':len(reservoir),
              'memory_writes':len(writes),'identities_actually_read':len(first_read),'memory_sampled':len(mem),
              'memory_read_eligible_samples':sum(w['read_expected'] for w in selected),
              'native_instrumentation_prediction_identical':True,'baseline_prediction_sha256':sha(baseline['predictions']),
              'reactivation_source_sha256':sha(root/'canonical_reactivation.jsonl'),
              'MEMORY_fields':MEMORY_FIELDS,'REACT_fields':REACT_FIELDS,
              'snapshot_files':{p.name:sha(p) for p in snapshots.glob('*.pth')},'official_test_read':False}
    manifest.update(native_MATCH_transition_contract=NATIVE_CONTRACT,
                    legacy_prefixes_eligible_for_canonical_supervision=False,
                    factual_prefix_reconstruction_without_GMT_forward=NATIVE_CONTRACT)
    save(root/'CANONICAL_MANIFEST.json',manifest);print(json.dumps(manifest))


def fork(video,device,shard,shards):
    root=probe_root(video);lab=NativeReplayLab(video,device,native_match_validation=NATIVE_CONTRACT);gt=OfflineIdentityAudit(video)
    baseline=json.loads((root/'baseline/result.json').read_text())
    actual=gt.align(json.loads(Path(baseline['predictions']).read_text()))
    outputs=[]
    for index,path in enumerate(sorted((root/'snapshots').glob('*.pth'))):
        if index%shards!=shard:continue
        c=torch.load(path,map_location='cpu');r=c['record'];key=tuple(r['key']);local=(key[1],key[2],key[4]);current_gt=actual[local]['gt']
        memory=r['question']=='MEMORY_DECISION';forced_actions=['WRITE_MEMORY','SKIP_MEMORY'] if memory else ['REACTIVATE_OLD','START_NEW']
        identity_map,prefix_counts=prefix_identity(actual,local[:2])
        target=identity_map.get(r['assigned_track_id'],current_gt) if memory else current_gt
        cases={}; rows_by_branch={};first_reads={};branch_ranks={};committed_ids={};termination={}
        paired_read_end=None
        for action in ['CONTROL']+forced_actions:
            first_read=None;rank_history=[];event_id=None;confirmed=set();last_frame=None;recoveries=[]
            def intervene(feature,question,legal,original,context):
                k=(video,context['frame'],context['view'],question,context['detection_index'])
                return action if k==key and action!='CONTROL' else original
            def before(k,payload,state,kwargs,decisions):
                nonlocal first_read
                pp=kwargs.get('reactivation_proposal')
                track=r['assigned_track_id']
                if pp is not None and track in pp.track_ids and track in state.stale_ids and track not in state.active_ids:
                    if first_read is None:first_read=k[1]
                    col=list(pp.track_ids).index(track)
                    resolution=lab.engine.resolve_actions(payload,state,actions=kwargs['actions'],proposal=kwargs['proposal'])
                    queries=[row for row in range(len(payload['pred_boxes'])) if resolution['existing_track_ids'].get(row) is None]
                    if len(queries)!=len(pp.scores):raise AssertionError('native stale query row contract changed')
                    rank_history.append({'key':list(k),'best_rank':min(int((scores>scores[col]).sum())+1 for scores in pp.scores),'max_score':float(pp.scores[:,col].max()),
                        'query_rows':queries,'ranks':[int((scores>scores[col]).sum())+1 for scores in pp.scores],
                        'bank_prototype_sha256':__import__('hashlib').sha256(state.reactivation_bank[track].numpy().tobytes()).hexdigest()})
                for row,old_id in kwargs.get('reactivation_assignments',{}).items():
                    if old_id==track:recoveries.append((k[1],k[2],int(row)))
            def after(k,payload,state,result,kwargs,decisions):
                nonlocal event_id,last_frame
                last_frame=k[1]
                if k==key[:3]:event_id=int(result['committed_track_ids'][key[4]])
                if memory:
                    if paired_read_end is not None and k[1]>=paired_read_end:
                        termination[action]='PAIRED_WRITE_READ_PLUS16';raise EndEventHorizon()
                    if first_read is not None and k[1]>=first_read+16:
                        termination[action]='READ_PLUS16';raise EndEventHorizon()
                else:
                    if event_id in result['committed_track_ids'].values():confirmed.add(k[1])
                    if k[1]>key[1] and event_id not in state.active_ids:
                        termination[action]='LOST_AGAIN';raise EndEventHorizon()
                    if len(confirmed)>=8:
                        termination[action]='8_OBSERVED_ID_FRAMES';raise EndEventHorizon()
                    if k[1]>=key[1]+32:
                        termination[action]='MAX32';raise EndEventHorizon()
            output=root/f'forks/{path.stem}/{action}'
            result=lab.run(output,prefix=c['state'],start_key=key[:3],metadata=c['metadata'],
                           intervention=intervene,before_step=before,after_step=after)
            aligned=gt.align(json.loads(Path(result['predictions']).read_text()));rows_by_branch[action]=aligned
            if action=='CONTROL':
                expected={k:v for k,v in actual.items() if k[:2]>=local[:2] and (k[0],k[1])<=max(k[:2] for k in aligned)}
                if aligned!=expected:raise AssertionError(f'native canonical fork control mismatch {key}')
            first_reads[action]=first_read;branch_ranks[action]=rank_history;committed_ids[action]=event_id
            if memory and action in {'CONTROL','WRITE_MEMORY'} and first_read is not None:
                paired_read_end=first_read+16
            cases[action]={'first_actual_bank_READ':first_read,'termination':termination.get(action,'SEQUENCE_END'),
                           'last_frame':last_frame,'prediction_sha256':sha(result['predictions']),
                           'committed_identity_at_intervention':event_id,'actual_identity_recoveries':recoveries}
        if memory:
            consumed=[first_reads[a] for a in forced_actions if first_reads[a] is not None]
            onset=min(consumed) if consumed else None
            end=min(v['last_frame'] for v in cases.values())
            horizons=[min(8,max(0,end-onset)),min(16,max(0,end-onset))] if onset is not None else [0,0]
            start=(onset,0) if onset is not None else local[:2]
            weight=float(target is not None and bool(consumed))
        else:
            start=local[:2];end=min(v['last_frame'] for v in cases.values());horizons=[max(0,end-key[1])]
            weight=float(target is not None)
        for a in cases:
            cases[a]['outcomes']={str(h):consequence(rows_by_branch[a],actual,start,target,h,c['state'].next_id,identity_key=local[:2]) for h in horizons}
            cases[a]['candidate_rank_history']=branch_ranks[a]
            for h,o in cases[a]['outcomes'].items():
                h=int(h)
                relevant=[]
                for read in branch_ranks[a]:
                    frame,view=read['key'][1:]
                    if not(start[0]<=frame<=start[0]+h):continue
                    relevant.extend(rank for row,rank in zip(read['query_rows'],read['ranks']) if actual[(frame,view,row)]['gt']==target and target is not None)
                recovered=[k for k in cases[a]['actual_identity_recoveries'] if start[0]<=k[0]<=start[0]+h]
                false=sum(actual[k]['gt'] is not None and actual[k]['gt']!=target for k in recovered)
                correct_recovery=next((k[0]-start[0] for k in recovered if actual[k]['gt']==target),None)
                o['false_identity_recoveries']=false
                o['mean_reciprocal_target_candidate_rank']=sum(1/rank for rank in relevant)/len(relevant) if relevant else 0.0
                o['target_identity_recovery_latency']=correct_recovery
                if memory:
                    o['utility']-=false
                    o['utility']+=.05*o['mean_reciprocal_target_candidate_rank']
                    if relevant:o['utility']-=.05*(correct_recovery if correct_recovery is not None else h+1)
            if not memory:
                mapping,_=prefix_identity(actual,start)
                false=target is not None and a=='REACTIVATE_OLD' and r['assigned_track_id'] in mapping and mapping[r['assigned_track_id']]!=target
                cases[a]['false_recovery']=bool(false)
                for o in cases[a]['outcomes'].values():o['utility']-=float(false)
        utilities=[cases[a]['outcomes'][str(horizons[-1])]['utility'] for a in forced_actions]
        informative=weight>0 and abs(utilities[0]-utilities[1])>1e-8
        final_weight=weight if informative else 0.0
        masses=[math.exp(u-max(utilities)) for u in utilities]
        record={**r,'offline_GT':target,'offline_current_observation_GT':current_gt,'offline_prefix_GT_counts':prefix_counts,
                'branches':cases,'sample_weight':final_weight,
                'informative':informative,'uninformative_reason':'GT_UNMATCHED' if target is None else 'UNREAD' if memory and not consumed else 'UTILITY_TIE' if not informative else None,
                'target_probs':[m/sum(masses) for m in masses],'best_actions':[a for a,u in zip(forced_actions,utilities) if abs(u-max(utilities))<=1e-8],
                'control_prefix_rng_and_native_bank_exact':True,'snapshot_sha256':sha(path),
                'utility_horizon_frames':horizons,'GT_only_in_targets':True}
        save(root/f'labels/{path.stem}.json',record);outputs.append(record)
    save(root/f'SHARD_{shard}.json',{'status':'COMPLETE','records':len(outputs),'informative':sum(r['informative'] for r in outputs),'shard':shard,'shards':shards})
    print(json.dumps({'video':video,'shard':shard,'records':len(outputs),'informative':sum(r['informative'] for r in outputs)}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['collect','fork']);p.add_argument('--video',type=int,choices=[23,24],required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    p.add_argument('--native-contract',action='store_true')
    args=p.parse_args()
    NATIVE_CONTRACT=args.native_contract
    if args.mode=='collect':collect(args.video,args.device)
    else:fork(args.video,args.device,args.shard,args.shards)
