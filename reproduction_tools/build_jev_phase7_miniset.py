"""Bounded native MATCH MiniSet; refuses execution until P0.5 completion."""
import argparse,copy,gzip,json,math
from collections import Counter
import torch
from jev_phase7_common import *


def collect_and_fork(video):
    gate=json.loads((REPORTS/'PHASE7_MATCH_MECHANISM_GO_NO_GO.json').read_text())
    assert gate['attribution_status']=='COMPLETE', 'P0.5 must finish first'
    assert video in (6,7)
    rollouts=isolate_phase6_imports()
    from jev_phase7_policies import MatchActor,A,R,N
    from jev_phase7_native import AttributionResolver
    lab=rollouts.NativeReplayLab(video,'cuda:0',compact_context=True);actor=MatchActor('B2')
    from jev_phase7_state import CompleteStateDigest
    complete_digest=CompleteStateDigest()
    root=OUT/'miniset_v1'/f'video{video:02d}'
    if root.exists():raise RuntimeError('refusing reused MiniSet output')
    root.mkdir(parents=True);start_binding=binding();population=[];captures=[];per_bin=Counter();joint_groups=[]
    final_frame=max(k[1]for k in lab.keys);original_resolve=lab.engine.resolve_actions
    writer=gzip.open(root/'factual_mechanisms.jsonl.gz','wt')
    resolver=AttributionResolver(lab,actor,writer=writer);lab.native_resolver=resolver;lab.engine.resolve_actions=resolver.resolve
    forced=None
    def choose(policy,feature,question,legal,off_action,context=None):
        context={k:v for k,v in (context or {}).items()if k!='tracker_state_before'}
        decision=actor.decide(feature,question,legal,off_action=off_action,context=context)if question=='MATCH_DECISION'else None
        original=decision.committed_action if decision else off_action;action=original
        key=(video,int(context['frame']),int(context['view']),int(context['detection_index']))
        if forced is not None and question=='MATCH_DECISION'and key[:3]==forced[0]:
            if key[3]in forced[1]:action=forced[2]
        assert action in legal
        lab.decision_rows.append({'question':question,'action':action,'original_action':original,'off_action':off_action,
            'legal':list(legal),'feature_vector':feature.detach().cpu().tolist(),'context':context,
            'probabilities':decision.probabilities if decision else {a:float(a==action)for a in legal}})
        return action
    lab.pilot.choose=choose
    def before(key,payload,state,kwargs,decisions):
        mechanism=resolver.last;rows=[];eligible=[]
        for d in decisions:
            if d['question']!='MATCH_DECISION':continue
            row=int(d['context']['detection_index']);scores=mechanism['scores'][row];order=sorted(range(len(scores)),key=lambda c:(-scores[c],c))
            margin=scores[order[0]]-scores[order[1]]if len(order)>1 else None
            descriptor={'key':list(key),'row':row,'legal':d['legal'],'candidate_ids':mechanism['track_ids'],
                'candidate_scores':scores,'proposed_column':mechanism['initial_pairs'].get(row),'factual_action':d['action'],
                'top1_top2_margin':margin}
            rows.append(descriptor);population.append(descriptor)
            if R in d['legal']:eligible.append((margin,row,d))
        bin_id=min(3,int(4*key[1]/(final_frame+1)))
        if eligible and per_bin[bin_id]<4:
            margin,row,decision=min(eligible,key=lambda x:(x[0],x[1]));index=len(captures)
            snapshot={'key':key,'row':row,'state':state.clone(),'metadata':copy.deepcopy(lab.metadata),
                'fingerprint':lab.state_fingerprint(state,lab.metadata),'complete_state_fingerprint':complete_digest(state,lab.metadata),'mechanism':copy.deepcopy(mechanism),
                'decision':copy.deepcopy(decision),'bin':bin_id,'population_rows':rows,
                'detection_feature':payload['reid_features'][row].tolist(),'detection_score':float(payload['detection_scores'][row]),
                'detection_box':payload['pred_boxes'][row].tolist()}
            path=root/'snapshots'/f'MATCH_{index:03d}.pth';path.parent.mkdir(exist_ok=True);torch.save(snapshot,path)
            captures.append(path);per_bin[bin_id]+=1
            # Two genuine overlapping row competitions, deterministic and GT-free.
            competitors=[];target_cols=set(sorted(range(len(mechanism['track_ids'])),key=lambda c:-mechanism['scores'][row][c])[:2])
            for _,other,_ in sorted(eligible):
                if other==row:continue
                top=set(sorted(range(len(mechanism['track_ids'])),key=lambda c:-mechanism['scores'][other][c])[:2])
                if target_cols&top:competitors.append(other)
            if competitors and len(joint_groups)<2:joint_groups.append({'snapshot':str(path),'rows':[row,competitors[0]]})
    try:factual=lab.run(root/'factual',before_step=before)
    finally:writer.close()
    with gzip.open(root/'population.jsonl.gz','wt')as stream:
        for r in population:stream.write(json.dumps(r,sort_keys=True)+'\n')
    # Deterministic first-in-bin sampling has inclusion 1 for chosen prefix events,
    # 0 for all others; selected/eligible fraction is descriptive, not a propensity.
    populations=Counter(min(3,int(4*r['key'][1]/(final_frame+1)))for r in population)
    eligible_payloads=Counter()
    for key in {tuple(r['key'])for r in population if R in r['legal']}:
        eligible_payloads[min(3,int(4*key[1]/(final_frame+1)))]+=1
    save(root/'COLLECTION_MANIFEST.json',{'binding':start_binding,'video':video,'role':'train'if video==7 else 'validation',
        'population_events':len(population),'population_bin_counts':dict(populations),'eligible_payload_bin_counts':dict(eligible_payloads),
        'sampled_bin_counts':dict(per_bin),'selection':'first four eligible payloads per exogenous frame bin; smallest score margin row',
        'sampling_probability':'deterministic inclusion 1 selected / 0 otherwise; no unbiased population-rate claim',
        'snapshots':{str(p):sha(p)for p in captures},'joint_groups':joint_groups,
        'factual_predictions_sha256':sha(factual['predictions']),'P05_gate_sha256':sha(REPORTS/'PHASE7_MATCH_MECHANISM_GO_NO_GO.json')})
    labels=[]
    for index,path in enumerate(captures):
        snapshot=torch.load(path,map_location='cpu');key=tuple(snapshot['key']);row=snapshot['row'];branches={}
        interventions=[('CONTROL',[],None),(A,[row],A),(R,[row],R),(N,[row],N)]
        group=next((g for g in joint_groups if g['snapshot']==str(path)),None)
        if group:
            other=group['rows'][1]
            interventions.extend([('OTHER_REASSOCIATE',[other],R),('JOINT_REASSOCIATE',group['rows'],R)])
        for name,rows,action in interventions:
            forced=(key,rows,action)if rows else None
            lab.engine.resolve_actions=original_resolve
            journal=root/'forks'/path.stem/f'{name}_mechanisms.jsonl.gz';journal.parent.mkdir(parents=True,exist_ok=True)
            writer=gzip.open(journal,'wt');resolver=AttributionResolver(lab,actor,writer=writer)
            lab.engine.resolve_actions=resolver.resolve;lab.native_resolver=resolver
            fingerprints={};commits=[];first=None
            def after(k,payload,state,result,kwargs,decisions):
                nonlocal first
                if first is None:
                    first=copy.deepcopy(resolver.last)
                    assert first['score_sha256']==snapshot['mechanism']['score_sha256']
                elapsed=k[1]-key[1]
                fingerprints[str(elapsed)]={'state_sha256':lab.state_fingerprint(state,lab.metadata),'complete_state_sha256':complete_digest(state,lab.metadata),'key':list(k)}
                commits.append({'key':list(k),'committed_ids':result['committed_track_ids']})
                if elapsed>=32 and k[2]==1:raise rollouts.EndEventHorizon()
            try:result=lab.run(root/'forks'/path.stem/name,prefix=snapshot['state'],start_key=key,end_frame=key[1]+32,
                               metadata=snapshot['metadata'],after_step=after)
            finally:writer.close()
            branches[name]={'result':result,'fingerprints':fingerprints,'first_mechanism':first,'commits':commits,
                            'prediction_sha256':sha(result['predictions']),'mechanisms_sha256':sha(journal)}
            save(root/'forks'/path.stem/name/'BRANCH_AUDIT.json',branches[name])
        forced=None
        # Only completed inference enters this offline label block.
        from jev_phase7_offline import IdentityEvaluator
        from jev_phase6_offline_utility import consequence,prefix_identity
        if index==0:
            evaluator=IdentityEvaluator(video);factual_rows=evaluator.align(json.loads(Path(factual['predictions']).read_text()))
        target=factual_rows.get((key[1],key[2],row),{}).get('gt');mapping,_=prefix_identity(factual_rows,key[1:])
        candidates=snapshot['mechanism']['track_ids'];correct=[t for t in candidates if target is not None and mapping.get(t)==target]
        proposed=snapshot['mechanism']['initial_pairs'].get(row);proposed_id=candidates[proposed]if proposed is not None else None
        outcomes={};aligned={}
        for name,b in branches.items():
            aligned[name]=evaluator.align(json.loads(Path(b['result']['predictions']).read_text()))
            if name=='CONTROL':assert all(v==factual_rows[k]for k,v in aligned[name].items())
            outcomes[name]={}
            for h in (8,16,32):
                window={k:v for k,v in aligned[name].items()if k[0]<=key[1]+h}
                outcomes[name][str(h)]=consequence(window,factual_rows,key[1:],target,h,snapshot['state'].next_id)
        factual_action=snapshot['decision']['action']
        assert branches['CONTROL']['fingerprints']==branches[factual_action]['fingerprints']
        assert branches['CONTROL']['commits']==branches[factual_action]['commits']
        utilities={a:outcomes[a]['32']['utility']for a in (A,R,N)};maximum=max(utilities.values());best=[a for a,u in utilities.items()if abs(u-maximum)<1e-8]
        margin=sorted(utilities.values(),reverse=True)[0]-sorted(utilities.values(),reverse=True)[1]
        informative=target is not None and margin>1e-8;mass={a:math.exp(u-maximum)for a,u in utilities.items()};z=sum(mass.values())
        label={'event_id':f'MATCH_video{video:02d}_{index:03d}','sequence':video,'role':'train'if video==7 else 'validation',
            'frame':key[1],'camera':key[2],'detection_index':row,'question':'MATCH_DECISION','state_fingerprint':snapshot['fingerprint'],'complete_state_fingerprint':snapshot['complete_state_fingerprint'],
            'snapshot_sha256':sha(path),'online':{'state_features':snapshot['decision']['feature_vector'],
            'detection_feature':snapshot['detection_feature'],'detection_score':snapshot['detection_score'],'detection_box':snapshot['detection_box'],
            'candidate_ids':candidates,'candidate_scores':snapshot['mechanism']['scores'][row],
            'proposed_column':proposed,'full_score_matrix':snapshot['mechanism']['scores'],
            'proposal_pairs':snapshot['mechanism']['initial_pairs'],'legal_actions':[A,R,N],'factual_action':factual_action},
            'offline_only':{'GT':target,'correct_candidate_ids':correct,'candidate_miss':target is not None and not correct and all(t in mapping for t in candidates),
                'candidate_identity_unknown_count':sum(t not in mapping for t in candidates),
                'candidate_oracle_uncertain':target is not None and not correct and any(t not in mapping for t in candidates),
                'proposal_correct':mapping.get(proposed_id)==target if target is not None and proposed_id in mapping else None,
                'feasible_correct_alternative_factual_wrong':target is not None and bool(correct)and proposed_id in mapping and mapping.get(proposed_id)!=target,
                'outcomes':outcomes,'utility_vector':utilities,'oracle_utility':maximum,'best_actions':best,
                'target_probabilities':{a:m/z for a,m in mass.items()},'utility_margin':margin,'informative':informative,
                'weight':float(informative),'unknown_target':target is None,'censored_horizons':[h for h in (8,16,32)if key[1]+h>final_frame]},
            'branches':branches,'control_factual_state_parity':True,'identical_prefix_RNG_and_exogenous_stream':True,
            'GT_or_future_policy_inputs':False,'coordinate_contract':'cache0_annotation1','binding':start_binding}
        labels.append(label);save(root/'labels'/f'{label["event_id"]}.json',label)
        save(root/'progress.json',{'completed_events':index+1,'sampled_events':len(captures)})
    save(root/'MINISET_RESULT.json',{'status':'COMPLETE','records':labels,'binding':start_binding,'video':video,'factual':factual})
    protect();print(json.dumps({'video':video,'records':len(labels),'informative':sum(r['offline_only']['informative']for r in labels)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=int,choices=[6,7],required=True);collect_and_fork(p.parse_args().video)
