"""New full-epoch heads on unchanged sealed DEV12: same-state primary replay only."""
import argparse
from collections import Counter,defaultdict
import csv,json,subprocess,time
from pathlib import Path
import torch
from cv_pil_dev12_shards import NativeShardAssets
from cv_pil_v2_model import CandidatePredictor,parameter_digest
from cv_pil_native_handoff import safe_torch_load
from pirga_indexed_store import sha
from cv_pil_v3_call_scoring import call_scores
from run_cv_pil_v3_old_heads import primary_solve,classify,retrieval,summarize,ARMS
from audit_cv_pil_v3_outcomes import targeted_audit


def execute(pins,out):
    if out.exists():raise FileExistsError(out)
    started=time.monotonic();torch.set_num_threads(1);epoch=pins['epoch']
    if epoch not in (1,3,5):raise ValueError('No unregistered DEV endpoint')
    protocol=pins['training_protocol']
    if sha(protocol['path'])!=protocol['sha256']:raise ValueError('Training protocol changed')
    p=json.loads(Path(protocol['path']).read_text());assets=NativeShardAssets(**pins['release']);models={}
    for arm in ('C2','C3','C4'):
        spec=pins['heads'][arm]
        if sha(spec['path'])!=spec['sha256']:raise ValueError('Head checkpoint changed')
        state=safe_torch_load(spec['path'])
        if state['arm']!=arm or state['epoch']!=epoch or state['step']!=113415*epoch or state['seed']!=20261015 or state['protocol_sha256']!=protocol['sha256'] or state['admission_sha256']!=p['admission']['sha256']:raise ValueError('Full-epoch head source/budget differs')
        model=CandidatePredictor(use_camera=arm!='C4');model.load_state_dict(state['state_dict'],strict=True)
        if sum(v.numel() for v in model.parameters())!=38080 or any(not torch.isfinite(v).all() for v in model.parameters()):raise ValueError('Head parameter contract failed')
        models[arm]=model.eval()
    scored={}
    with torch.no_grad():
        for cid,c in assets.calls.items():
            if c['stage']=='run_memory_tracker' or not c['queries']:continue
            matrices=call_scores(assets,c,models);actions={arm:primary_solve(m,c['track_ids'],c['thresholds']) for arm,m in matrices.items()}
            if actions['C0']!=[None if v<0 else v for v in c['decision']['accepted_track_ids']]:raise ValueError('Original complete joint primary action differs')
            for row,source in enumerate(c['queries']):scored[(12,*source)]=dict(scores={arm:m[row] for arm,m in matrices.items()},actions={arm:v[row] for arm,v in actions.items()})
            if len(scored)%1000<len(c['queries']):print(json.dumps(dict(epoch=epoch,scored_queries=len(scored),GT_semantics_loaded=False)),flush=True)
    # Only now parse physically separate GT supervision. No oracle Gallery or
    # label-derived candidate is ever provided to the heads or native solver.
    labels=assets.supervision();records=[]
    for key in assets.query_index:
        label=labels[key];meta=assets.observations[key[1:]];cid,_=assets.query_index[key];ids=assets.calls[cid]['track_ids'];s=scored[key]
        r=dict(key=list(key),person=label['GT_person'],camera=meta['camera'],frame=meta['frame'],time_bin50=(meta['frame']-1)//50,
            positive_covered=bool(label['certified_target_indices']),source_primary_state=label['actual_primary_state'],
            original_bank_restored=label['source_label']['bank_restored'],original_newborn=label['source_label']['newborn'])
        for arm in ARMS:
            pred,rr,outcome=retrieval(s['scores'][arm],ids,label)
            primary=classify(s['actions'][arm],label,ids)
            if label['GT_person'] is None:outcome=primary=None
            r.update({arm+'_rank_id':pred,arm+'_rr':rr,arm+'_rank':outcome,arm+'_accepted_id':s['actions'][arm],arm+'_primary':primary})
        records.append(r)
    # Strong untrained controls must remain exactly the prior full same-range
    # results, independent of the new trained checkpoint generation.
    reference=pins['frozen_control_csv']
    if sha(reference['path'])!=reference['sha256']:raise ValueError('Old exact-range control changed')
    old={tuple(json.loads(r['key'])):r for r in csv.DictReader(Path(reference['path']).open())}
    if set(old)!=set(map(lambda r:tuple(r['key']),records)):raise ValueError('Evaluation scope drift')
    for r in records:
        for arm in ('C0','C1'):
            for field in ('rank_id','rr','rank','accepted_id','primary'):
                value=r[arm+'_'+field];expected='' if value is None else str(value)
                if old[tuple(r['key'])][arm+'_'+field]!=expected:raise ValueError('C0/C1 original result drift '+str((r['key'],arm,field)))
    result=summarize(records);mechanism=targeted_audit(records);result['by_scene']={'12':summarize(records)}
    groups=defaultdict(list)
    for r in records:
        if r['person'] is not None:groups[str(r['person'])].append(r)
    result['by_person']={k:summarize(v) for k,v in groups.items()}
    result.update(status='SAME_STATE_NATIVE_REPLAY_ONLY',epoch=epoch,training_protocol_sha256=protocol['sha256'],checkpoints=pins['heads'],
        C0_C1_exact_prior_same_range_control=True,all_scores_and_actions_preceded_GT_semantics=True,queries=len(records),
        source_primary_states=dict(Counter(r['source_primary_state'] for r in records)),query_UNKNOWN=sum(r['person'] is None for r in records),
        coverage_all_queries=sum(r['positive_covered'] for r in records)/len(records),
        original_error_mechanism=mechanism,full_native_closed_loop=False,scientific_success=False,
        online_metrics={k:None for k in ['HOTA','AssA','DetA','IDF1','CVIDF1','CVMA','MOTA','IDSW','Frag','FPS']},
        counterfactual_Bank_Gallery_birth='NOT_RUN_ORIGINAL_ACTIONS_ARE_SOURCE_ONLY',wall_seconds=time.monotonic()-started,GPU_wall_hours=0,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),
        qualification='Fixed TRAIN12 exposed development. Epoch1 engineering,3 formal,5 sufficiency: never select a favorable endpoint or claim online gain. New Phase2 hypotheses remain preregistered even if original C3 is negative.')
    out.mkdir(parents=True)
    with open(out/'QUERY_RESULTS_OFFLINE_ONLY.csv','x') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(out/'INPUT_PINS.json').write_text(json.dumps(pins,indent=2)+'\n')
    (out/'ORIGINAL_ERROR_MECHANISM.json').write_text(json.dumps(mechanism,indent=2)+'\n')
    print(json.dumps(dict(epoch=epoch,mechanism_status=mechanism['status'],methods=mechanism['methods'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pins',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();execute(json.loads(a.pins.read_text()),a.out)
