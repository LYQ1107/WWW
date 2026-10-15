"""Complete native DEV replay for separately labelled canonical300/legacy epochs."""
import argparse,csv,json,time
from collections import defaultdict
from pathlib import Path
import torch
from cv_pil_dev12_shards import NativeShardAssets
from cv_pil_v2_model import CandidatePredictor
from cv_pil_native_handoff import safe_torch_load
from cv_pil_v3_call_scoring import call_scores
from run_pirga_phase1_replay import primary_solve,certified_outcomes
from pirga_replay_reporting import method_summary,mechanism_audit
from pirga_indexed_store import sha

ARMS=('C0','C1','C2','C3','C4')


def execute(pins,out):
    if out.exists():raise FileExistsError(out)
    proto=pins['protocol']
    if sha(proto['path'])!=proto['sha256']:raise ValueError('Coverage control protocol changed')
    p=json.loads(Path(proto['path']).read_text());epoch=pins['epoch'];mode=pins['mode'];models={};repo=Path(__file__).resolve().parents[1]
    if mode not in ('CANONICAL_MASKED300','LEGACY_FULL_CANONICAL') or (mode=='CANONICAL_MASKED300' and epoch is not None) or (mode=='LEGACY_FULL_CANONICAL' and epoch not in [1,3,5]):raise ValueError('Unregistered control endpoint')
    for name,digest in p['source_code_sha256'].items():
        if sha(repo/name)!=digest:raise ValueError('Frozen control source changed '+name)
    expected_step=300 if mode=='CANONICAL_MASKED300' else p['queries_per_epoch']*epoch
    for arm in ['C2','C3','C4']:
        spec=pins['heads'][arm]
        if sha(spec['path'])!=spec['sha256']:raise ValueError('Head changed')
        state=safe_torch_load(spec['path'])
        if state['arm']!=arm or state['step']!=expected_step or state['epoch']!=epoch or state['mode']!=mode or state['protocol_sha256']!=proto['sha256'] or state['admission_sha256']!=p['full_admission']['sha256'] or state['seed']!=p['seed']:raise ValueError('Wrong control source,coverage or mode')
        model=CandidatePredictor(use_camera=arm!='C4');model.load_state_dict(state['state_dict'],strict=True)
        if sum(x.numel() for x in model.parameters())!=38080 or any(not torch.isfinite(x).all() for x in model.parameters()):raise ValueError('Legacy architecture differs')
        models[arm]=model.eval()
    torch.set_num_threads(1);start=time.monotonic();assets=NativeShardAssets(**pins['release'])
    if assets.engineering_only:raise ValueError('No prefix evaluation')
    scored={}
    with torch.no_grad():
        for cid,c in assets.calls.items():
            if c['stage']=='run_memory_tracker' or not c['queries']:continue
            matrices=call_scores(assets,c,models);actions={k:primary_solve(v,c['track_ids'],c['thresholds']) for k,v in matrices.items()}
            if actions['C0']!=[None if v<0 else v for v in c['decision']['accepted_track_ids']]:raise ValueError('C0 joint native drift')
            for row,key in enumerate(c['queries']):scored[(12,*key)]=dict(scores={k:v[row] for k,v in matrices.items()},actions={k:v[row] for k,v in actions.items()})
            if len(scored)%1000<len(c['queries']):print(json.dumps(dict(mode=mode,epoch=epoch,scored_queries=len(scored),GT_semantics_loaded=False)),flush=True)
    labels=assets.supervision();records=[]
    for key in assets.query_index:
        label=labels[key];cid,_=assets.query_index[key];c=assets.calls[cid];s=scored[key]
        r=dict(key=list(key),person=label['GT_person'],positive_covered=bool(label['certified_target_indices']),source_primary_state=label['actual_primary_state'])
        for arm in ARMS:
            pred,rr,rank,primary=certified_outcomes(s['scores'][arm],c['track_ids'],s['actions'][arm],label)
            r.update({arm+'_rank_id':pred,arm+'_rr':rr,arm+'_rank':rank,arm+'_accepted_id':s['actions'][arm],arm+'_primary':primary})
        records.append(r)
    reference=pins['reference_csv']
    if sha(reference['path'])!=reference['sha256']:raise ValueError('Frozen C0/C1 same-range reference changed')
    with Path(reference['path']).open() as f:old={tuple(json.loads(r['key'])):r for r in csv.DictReader(f)}
    if set(old)!={tuple(r['key']) for r in records}:raise ValueError('DEV scope changed')
    for r in records:
        for arm in ['C0','C1']:
            for field in ['rank_id','rr','rank','accepted_id','primary']:
                value=r[arm+'_'+field]
                if old[tuple(r['key'])][arm+'_'+field]!=('' if value is None else str(value)):raise ValueError('C0/C1 exact control drift')
    audit=mechanism_audit(records,ARMS,forecast='C3',controls=('C1','C2','C4'));audit['online_admitted']=False;groups=defaultdict(list)
    for r in records:
        if r['person'] is not None:groups[str((12,r['person']))].append(r)
    result=dict(status='SAME_STATE_NATIVE_REPLAY_ONLY',family=mode,epoch=epoch,step=expected_step,queries=len(records),methods=method_summary(records,ARMS),mechanism=audit,
        by_scene={'12':method_summary(records,ARMS)},by_person_scene={k:method_summary(v,ARMS) for k,v in groups.items()},
        training_mask_policy='Conservative canonical policy identical between refit300 and full legacy control; original old300 retained separately',
        all_actions_before_GT_semantics=True,C0_C1_exact_old_same_range=True,protocol_sha256=proto['sha256'],GPU_wall_hours=0,wall_seconds=time.monotonic()-start,
        online_metrics={k:None for k in ['HOTA','AssA','DetA','IDF1','CVIDF1','CVMA','MOTA','IDSW','Frag','FPS']},scientific_success=False,online_state_mutated=False,
        qualification='Legacy Top8/latest4 used only for TRAIN, as historical experiment; unchanged complete native candidate/history context for fixedDEV12 evaluation. Small300 schedule and full106800-query epoch visits are separately labelled; no favorable endpoint selection or online claim.')
    out.mkdir(parents=True)
    with (out/'QUERY_RESULTS_OFFLINE_ONLY.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');(out/'INPUT_PINS.json').write_text(json.dumps(pins,indent=2)+'\n');print(json.dumps(dict(mode=mode,epoch=epoch,methods=audit['methods'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pins',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();execute(json.loads(a.pins.read_text()),a.out)
