"""Supplementary exhaustive-margin scope on the identical complete original DEV states."""
import argparse,csv,json,time
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
from cv_pil_dev12_shards import NativeShardAssets
from cv_pil_v2_model import tensorize
from cv_pil_native_handoff import safe_torch_load
from cv_pil_v3_call_scoring import call_scores
from pirga_candidate_state import candidate_state
from pirga_native_context import opportunity
from pirga_decision_head import DecisionMLP,decision_input,ARMS as NEW_ARMS
from pirga_replay_reporting import method_summary,mechanism_audit
from run_pirga_phase1_replay import load_epoch_heads,primary_solve,certified_outcomes
from pirga_indexed_store import sha

ARMS=('C0','C1','C2','C3','C4',*NEW_ARMS)


def execute(pins,out):
    if out.exists():raise FileExistsError(out)
    start=time.monotonic();torch.set_num_threads(1);spec=pins['protocol'];epoch=pins['epoch'];repo=Path(__file__).resolve().parents[1]
    if sha(spec['path'])!=spec['sha256'] or epoch not in (1,3,5):raise ValueError('Unregistered protocol/endpoint')
    p=json.loads(Path(spec['path']).read_text());heads={}
    for name,digest in p['source_code_sha256'].items():
        if sha(repo/name)!=digest:raise ValueError('Frozen ordinary training source changed')
    for arm in NEW_ARMS:
        checkpoint=pins['heads'][arm]
        if sha(checkpoint['path'])!=checkpoint['sha256']:raise ValueError('Ordinary head changed')
        state=safe_torch_load(checkpoint['path'])
        if state['arm']!=arm or state['epoch']!=epoch or state['step']!=epoch*113453 or state['protocol_sha256']!=spec['sha256'] or state['admission_sha256']!=p['admission']['sha256'] or state['seed']!=p['seed']:raise ValueError('Head source or coverage changed')
        model=DecisionMLP();model.load_state_dict(state['state_dict'],strict=True)
        if sum(x.numel() for x in model.parameters())!=38204 or any(not torch.isfinite(x).all() for x in model.parameters()):raise ValueError('Ordinary head parameter contract')
        heads[arm]=model.eval()
    oldpins=pins['phase1_pins']
    if sha(oldpins['path'])!=oldpins['sha256']:raise ValueError('Reference changed')
    old=json.loads(Path(oldpins['path']).read_text());oldspec=old['training_protocol']
    if old['epoch']!=epoch or old['release']!=pins['release'] or sha(oldspec['path'])!=oldspec['sha256']:raise ValueError('Reference source/endpoint differs')
    oldheads=load_epoch_heads(old,json.loads(Path(oldspec['path']).read_text()));assets=NativeShardAssets(**pins['release'])
    if assets.engineering_only:raise ValueError('No prefix evaluation')
    scored={}
    with torch.no_grad():
        for cid,c in assets.calls.items():
            if c['stage']=='run_memory_tracker' or not c['queries']:continue
            if len({(assets.observations[tuple(q)]['frame'],assets.observations[tuple(q)]['camera'],assets.observations[tuple(q)]['image_id']) for q in c['queries']})!=1:raise ValueError('Heterogeneous native call')
            actor=assets.actor((12,*c['queries'][0]));b=tensorize(actor);lags=torch.tensor([r['lag'] for rows in actor['past_histories'] for r in rows],dtype=torch.float32)
            st=candidate_state(b.histories,b.owners,lags,b.cross_view,b.destination_camera,len(b.native_scores),p['horizon_boundaries'])
            qs=torch.stack([F.normalize(assets.feature(key).detach().float().clone(),dim=0) for key in c['query_feature_keys']])
            joint_base=c['track_scores']+2*(qs@st.mean.T);scores=call_scores(assets,c,oldheads);extra={k:[] for k in NEW_ARMS};thresholds=torch.tensor(c['thresholds'],dtype=torch.float32)
            for row,source in enumerate(c['queries']):
                q=qs[row];native=c['track_scores'][row].clone();opp=opportunity(joint_base.numpy(),row)
                identity=c['decision']['accepted_track_ids'][row];accepted=c['track_ids'].index(identity) if identity>=0 else -1
                x=decision_input(st,q,native,thresholds,opp,accepted)
                for arm,head in heads.items():extra[arm].append(head(x,st,q,native)[0])
                if not torch.equal(native+2*(st.mean@q),scores['C1'][row]):raise ValueError('C1 candidate-state drift')
            scores.update({k:torch.stack(v) for k,v in extra.items()});actions={k:primary_solve(v,c['track_ids'],c['thresholds']) for k,v in scores.items()}
            if actions['C0']!=[None if v<0 else v for v in c['decision']['accepted_track_ids']]:raise ValueError('C0 joint action drift')
            for row,source in enumerate(c['queries']):scored[(12,*source)]=dict(scores={k:v[row] for k,v in scores.items()},actions={k:v[row] for k,v in actions.items()})
            if len(scored)%1000<len(c['queries']):print(json.dumps(dict(epoch=epoch,scored_queries=len(scored),GT_semantics_loaded=False)),flush=True)
    labels=assets.supervision();records=[]
    for key in assets.query_index:
        label=labels[key];c=assets.calls[assets.query_index[key][0]];s=scored[key];meta=assets.observations[key[1:]]
        r=dict(key=list(key),person=label['GT_person'],camera=meta['camera'],frame=meta['frame'],positive_covered=bool(label['certified_target_indices']),source_primary_state=label['actual_primary_state'])
        for arm in ARMS:
            pred,rr,rank,primary=certified_outcomes(s['scores'][arm],c['track_ids'],s['actions'][arm],label)
            r.update({arm+'_rank_id':pred,arm+'_rr':rr,arm+'_rank':rank,arm+'_accepted_id':s['actions'][arm],arm+'_primary':primary})
        records.append(r)
    reference=pins['phase1_reference_csv']
    if sha(reference['path'])!=reference['sha256']:raise ValueError('Frozen reference changed')
    with Path(reference['path']).open() as f:oldrows={tuple(json.loads(r['key'])):r for r in csv.DictReader(f)}
    if set(oldrows)!={tuple(r['key']) for r in records}:raise ValueError('DEV scope drift')
    for r in records:
        for arm in ('C0','C1','C2','C3','C4'):
            for field in ('rank_id','rr','rank','accepted_id','primary'):
                value=r[arm+'_'+field]
                if oldrows[tuple(r['key'])][arm+'_'+field]!=('' if value is None else str(value)):raise ValueError('Exact reference output drift')
    audit=mechanism_audit(records,ARMS,forecast='D2_MLP_native_margin',controls=('C1','D1_MLP','C2','C3'))
    audit['status']='ORDINARY_NATIVE_DECISION_CONTROL_ONLY_NOT_PREDICTIVE_SUCCESS';audit['online_admitted']=False
    groups=defaultdict(list)
    for r in records:
        if r['person'] is not None:groups[str((12,r['person']))].append(r)
    result=dict(status='SAME_STATE_NATIVE_REPLAY_ONLY',epoch=epoch,methods=method_summary(records,ARMS),mechanism=audit,
        by_scene={'12':method_summary(records,ARMS)},by_person_scene={k:method_summary(v,ARMS) for k,v in groups.items()},
        all_scores_and_actions_preceded_GT_semantics=True,C0_C1_C2_C3_C4_exact_same_epoch=True,protocol_sha256=spec['sha256'],
        wall_seconds=time.monotonic()-start,GPU_wall_hours=0,scientific_success=False,online_state_mutated=False,
        online_metrics={k:None for k in ['HOTA','AssA','DetA','IDF1','CVIDF1','CVMA','MOTA','IDSW','Frag','FPS']},
        qualification='Supplementary113453-query margin scope; Phase1 reference113415queries/epoch. Same epoch number, not identical training/update budget (38extra visits per epoch). Full learned scores jointly assigned; context uses fixedC1 other rows. No future state claim.')
    out.mkdir(parents=True)
    with (out/'QUERY_RESULTS_OFFLINE_ONLY.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');(out/'INPUT_PINS.json').write_text(json.dumps(pins,indent=2)+'\n');print(json.dumps(dict(epoch=epoch,methods=audit['methods'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pins',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();execute(json.loads(a.pins.read_text()),a.out)
