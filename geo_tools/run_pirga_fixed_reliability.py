"""Low-cost preregistered reliability controls, complete original-state DEV replay."""
import argparse,csv,json,time
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F
from cv_pil_dev12_shards import NativeShardAssets
from cv_pil_v2_model import tensorize
from pirga_fixed_reliability import prototypes,METHODS
from pirga_indexed_store import sha
from run_pirga_phase1_replay import certified_outcomes,primary_solve
from pirga_replay_reporting import method_summary,mechanism_audit


def run(protocol_path,protocol_sha,out):
    if out.exists():raise FileExistsError(out)
    if sha(protocol_path)!=protocol_sha:raise ValueError('Unpinned fixed-control protocol')
    p=json.loads(protocol_path.read_text());repo=Path(__file__).resolve().parents[1]
    for name,digest in p['source_sha256'].items():
        if sha(repo/name)!=digest:raise ValueError('Fixed-control code changed')
    torch.set_num_threads(1);start=time.monotonic();assets=NativeShardAssets(**p['release']);arms=('C0','C1',*METHODS);scored={}
    if assets.engineering_only:raise ValueError('Prefix not evaluation')
    with torch.no_grad():
        for cid,c in assets.calls.items():
            if c['stage']=='run_memory_tracker' or not c['queries']:continue
            contexts={(assets.observations[tuple(q)]['frame'],assets.observations[tuple(q)]['camera'],assets.observations[tuple(q)]['image_id']) for q in c['queries']}
            if len(contexts)!=1:raise ValueError('Mixed native query context')
            actor=assets.actor((12,*c['queries'][0]));batch=tensorize(actor)
            lags=torch.tensor([r['lag'] for rows in actor['past_histories'] for r in rows],dtype=torch.float32);proto=prototypes(batch,lags,p['decay_tau_frames']);scores={arm:[] for arm in arms}
            for row,source in enumerate(c['queries']):
                q=F.normalize(assets.feature(c['query_feature_keys'][row]).float().clone(),dim=0);native=c['track_scores'][row].float().clone()
                scores['C0'].append(native);scores['C1'].append(native+2*(proto['R_mean']@q))
                for method in METHODS:scores[method].append(native+2*(proto[method]@q))
            scores={k:torch.stack(v) for k,v in scores.items()};actions={k:primary_solve(v,c['track_ids'],c['thresholds']) for k,v in scores.items()}
            if actions['C0']!=[None if x<0 else x for x in c['decision']['accepted_track_ids']]:raise ValueError('C0 action drift')
            if not torch.equal(scores['C1'],scores['R_mean']):raise ValueError('C1 mean drift')
            for row,source in enumerate(c['queries']):scored[(12,*source)]=dict(scores={k:v[row] for k,v in scores.items()},actions={k:v[row] for k,v in actions.items()})
            if len(scored)%1000<len(c['queries']):print(json.dumps(dict(scored_queries=len(scored),GT_semantics_loaded=False)),flush=True)
    labels=assets.supervision();records=[]
    for key in assets.query_index:
        s=scored[key];label=labels[key];c=assets.calls[assets.query_index[key][0]];r=dict(key=list(key),person=label['GT_person'],positive_covered=bool(label['certified_target_indices']),source_primary_state=label['actual_primary_state'])
        for arm in arms:
            pred,rr,rank,primary=certified_outcomes(s['scores'][arm],c['track_ids'],s['actions'][arm],label)
            r.update({arm+'_rank_id':pred,arm+'_rr':rr,arm+'_rank':rank,arm+'_accepted_id':s['actions'][arm],arm+'_primary':primary})
        records.append(r)
    ref=p['reference']
    if sha(ref['path'])!=ref['sha256']:raise ValueError('Exact-range control changed')
    with Path(ref['path']).open() as f:reference={tuple(json.loads(r['key'])):r for r in csv.DictReader(f)}
    if set(reference)!={tuple(r['key']) for r in records}:raise ValueError('Scope drift')
    for r in records:
        for arm in ('C0','C1'):
            for field in ('rank_id','rr','rank','accepted_id','primary'):
                value=r[arm+'_'+field]
                if reference[tuple(r['key'])][arm+'_'+field]!=('' if value is None else str(value)):raise ValueError('C0/C1 exact control drift')
    comparisons={};mechanism=None
    for method in METHODS:
        mechanism=mechanism_audit(records,arms,forecast=method,controls=('C1',))
        comparisons[method]=mechanism['comparisons'][method+'_vs_C1']
    groups=defaultdict(list)
    for r in records:
        if r['person'] is not None:groups[str((12,r['person']))].append(r)
    result=dict(status='SAME_STATE_NATIVE_REPLAY_ONLY',research_family='NONLEARNED_HISTORY_RELIABILITY_CONTROLS',
        queries=len(records),methods=method_summary(records,arms),original_error_methods=mechanism['methods'],comparisons_to_C1=comparisons,
        by_scene={'12':method_summary(records,arms)},by_person_scene={k:method_summary(v,arms) for k,v in groups.items()},
        protocol_sha256=protocol_sha,all_actions_before_GT_semantics=True,C0_C1_exact_same_range=True,
        learnable_parameters=0,GPU_wall_hours=0,wall_seconds=time.monotonic()-start,scientific_success=False,online_admitted=False,
        qualification='Complete exposed DEV12, original frozen states; no Gallery WRITE change, no predictive mechanism attribution, no final online propagation measured.')
    out.mkdir(parents=True)
    with (out/'QUERY_RESULTS_OFFLINE_ONLY.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');(out/'INPUT_PINS.json').write_text(json.dumps(p,indent=2)+'\n')
    print(json.dumps(dict(methods=result['original_error_methods'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.protocol,a.protocol_sha,a.out)
