"""Complete sealed DEV replay for fixed Phase2 checkpoints; no mutated state."""
import argparse,csv,json,subprocess,time
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F
from cv_pil_dev12_shards import NativeShardAssets
from cv_pil_v2_model import tensorize
from cv_pil_native_handoff import safe_torch_load
from cv_pil_v3_call_scoring import call_scores
from pirga_candidate_state import candidate_state
from pirga_innovation import InnovationHead,ARMS as NEW_ARMS
from pirga_indexed_store import sha
from run_pirga_phase1_replay import load_epoch_heads,primary_solve,certified_outcomes
from pirga_replay_reporting import method_summary,mechanism_audit,uncertainty_report

ARMS=('C0','C1','C2','C3','C4',*NEW_ARMS,'A8_no_innovation')


def execute(pins,out):
    if out.exists():raise FileExistsError(out)
    start=time.monotonic();torch.set_num_threads(1);protocol=pins['protocol']
    if sha(protocol['path'])!=protocol['sha256']:raise ValueError('Phase2 protocol changed')
    p=json.loads(Path(protocol['path']).read_text());epoch=pins['epoch'];heads={}
    if epoch not in (1,3,5) or p['status']!='PREREGISTERED_PHASE2_BEFORE_OPTIMIZATION':raise ValueError('Unregistered endpoint')
    repo=Path(__file__).resolve().parents[1]
    for name,digest in p['source_code_sha256'].items():
        if sha(repo/name)!=digest:raise ValueError('Frozen training source changed')
    for arm in NEW_ARMS:
        spec=pins['heads'][arm]
        if sha(spec['path'])!=spec['sha256']:raise ValueError('Head changed')
        state=safe_torch_load(spec['path'])
        if state['arm']!=arm or state['epoch']!=epoch or state['step']!=113415*epoch or state['protocol_sha256']!=protocol['sha256'] or state['admission_sha256']!=p['admission']['sha256'] or state['horizon_boundaries']!=p['horizon_boundaries'] or state['seed']!=p['seed']:raise ValueError('Invalid complete endpoint source')
        model=InnovationHead(arm);model.load_state_dict(state['state_dict'],strict=True)
        if sum(v.numel() for v in model.parameters())!=38225 or any(not torch.isfinite(v).all() for v in model.parameters()):raise ValueError('Invalid head parameters')
        heads[arm]=model.eval()
    p1pins=pins['phase1_pins']
    if sha(p1pins['path'])!=p1pins['sha256']:raise ValueError('Phase1 reference pins changed')
    p1=json.loads(Path(p1pins['path']).read_text())
    if p1['epoch']!=epoch or p1['release']!=pins['release']:raise ValueError('Reference scope/epoch drift')
    pp=p1['training_protocol']
    if sha(pp['path'])!=pp['sha256']:raise ValueError('Phase1 reference protocol changed')
    old_heads=load_epoch_heads(p1,json.loads(Path(pp['path']).read_text()))
    assets=NativeShardAssets(**pins['release'])
    if assets.engineering_only:raise ValueError('Engineering prefix not evaluation')
    scored={};diagnostics={}
    with torch.no_grad():
        for cid,c in assets.calls.items():
            if c['stage']=='run_memory_tracker' or not c['queries']:continue
            contexts={(assets.observations[tuple(q)]['frame'],assets.observations[tuple(q)]['camera'],assets.observations[tuple(q)]['image_id']) for q in c['queries']}
            if len(contexts)!=1:raise ValueError('Mixed native query context requires per-row state adapter')
            actor=assets.actor((12,*c['queries'][0]));b=tensorize(actor)
            lags=torch.tensor([r['lag'] for rows in actor['past_histories'] for r in rows],dtype=torch.float32)
            st=candidate_state(b.histories,b.owners,lags,b.cross_view,b.destination_camera,len(b.native_scores),p['horizon_boundaries'])
            scores=call_scores(assets,c,old_heads)
            extra={arm:[] for arm in NEW_ARMS};stats={}
            for row,source in enumerate(c['queries']):
                q=F.normalize(assets.feature(c['query_feature_keys'][row]).detach().float().clone(),dim=0);native=c['track_scores'][row].detach().float().clone();stats[row]={}
                for arm,head in heads.items():
                    sc,mu,lv=head(st,q,native);extra[arm].append(sc)
                    stats[row][arm]=dict(error=(1-mu@q).tolist(),variance=lv.exp().tolist(),horizon=st.conditions[:,7:].argmax(1).tolist())
                base=native+2*(st.mean@q)
                if not torch.equal(base,scores['C1'][row]):raise ValueError('New candidate aggregation changed C1 control')
            scores.update({arm:torch.stack(v) for arm,v in extra.items()});scores['A8_no_innovation']=scores['C1'].clone()
            actions={arm:primary_solve(v,c['track_ids'],c['thresholds']) for arm,v in scores.items()}
            if actions['C0']!=[None if v<0 else v for v in c['decision']['accepted_track_ids']]:raise ValueError('Original joint primary parity failed')
            for row,source in enumerate(c['queries']):
                key=(12,*source);scored[key]=dict(scores={arm:v[row] for arm,v in scores.items()},actions={arm:v[row] for arm,v in actions.items()});diagnostics[key]=stats[row]
            if len(scored)%1000<len(c['queries']):print(json.dumps(dict(epoch=epoch,scored_queries=len(scored),GT_semantics_loaded=False)),flush=True)
    labels=assets.supervision();records=[];uncertainty=defaultdict(list)
    for key in assets.query_index:
        label=labels[key];c=assets.calls[assets.query_index[key][0]];ids=c['track_ids'];meta=assets.observations[key[1:]];s=scored[key]
        r=dict(key=list(key),person=label['GT_person'],camera=meta['camera'],frame=meta['frame'],positive_covered=bool(label['certified_target_indices']),source_primary_state=label['actual_primary_state'])
        for arm in ARMS:
            pred,rr,rank,primary=certified_outcomes(s['scores'][arm],ids,s['actions'][arm],label)
            r.update({arm+'_rank_id':pred,arm+'_rr':rr,arm+'_rank':rank,arm+'_accepted_id':s['actions'][arm],arm+'_primary':primary})
        for arm in NEW_ARMS:
            d=diagnostics[key][arm];pos=label['certified_target_indices'] if label['GT_person'] is not None else []
            uncertainty[arm].append(dict(key=list(key),person=label['GT_person'],primary_correct=r[arm+'_primary'],
                positive_variance=sum(d['variance'][i] for i in pos)/len(pos) if pos else None,
                positive_cosine_error=sum(d['error'][i] for i in pos)/len(pos) if pos else None,
                positive_horizons=sorted({d['horizon'][i] for i in pos})))
        records.append(r)
    # Reference heads and controls are checked against their exact same epoch
    # raw CSV, rather than silently trusting a new scorer implementation.
    reference=pins['phase1_reference_csv']
    if sha(reference['path'])!=reference['sha256']:raise ValueError('Reference changed')
    with Path(reference['path']).open() as f:old={tuple(json.loads(r['key'])):r for r in csv.DictReader(f)}
    if set(old)!={tuple(r['key']) for r in records}:raise ValueError('Scope drift')
    for r in records:
        for arm in ('C0','C1','C2','C3','C4'):
            for field in ('rank_id','rr','rank','accepted_id','primary'):
                value=r[arm+'_'+field];expected='' if value is None else str(value)
                if old[tuple(r['key'])][arm+'_'+field]!=expected:raise ValueError('Exact Phase1 control drift')
    report=mechanism_audit(records,ARMS);groups=defaultdict(list);cameras=defaultdict(list)
    for r in records:
        if r['person'] is not None:groups[str((12,r['person']))].append(r)
        cameras[str(r['camera'])].append(r)
    result=dict(status='SAME_STATE_NATIVE_REPLAY_ONLY',epoch=epoch,methods=method_summary(records,ARMS),mechanism=report,
        by_scene={'12':method_summary(records,ARMS)},by_person_scene={k:method_summary(v,ARMS) for k,v in groups.items()},
        by_camera={k:method_summary(v,ARMS) for k,v in cameras.items()},
        uncertainty={arm:uncertainty_report(v) for arm,v in uncertainty.items()},
        uncertainty_by_person={arm:{group:uncertainty_report([r for r in v if str((12,r['person']))==group]) for group in groups} for arm,v in uncertainty.items()},
        uncertainty_by_horizon={arm:{str(h):uncertainty_report([r for r in v if h in r['positive_horizons']]) for h in range(4)} for arm,v in uncertainty.items()},
        horizon_group_overlap='Positive aliases can occupy multiple horizon groups; no pooled independence claim',
        all_scores_and_actions_preceded_GT_semantics=True,C0_C1_C2_C3_C4_exact_prior_same_epoch=True,A8_exact_C1=True,
        queries=len(records),GT_unknown_queries=sum(r['person'] is None for r in records),positive_candidate_coverage=sum(r['positive_covered'] for r in records)/len(records),
        wall_seconds=time.monotonic()-start,GPU_wall_hours=0,protocol_sha256=protocol['sha256'],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        online_metrics={k:None for k in ['HOTA','AssA','DetA','IDF1','CVIDF1','CVMA','MOTA','IDSW','Frag','FPS']},online_state_mutated=False,scientific_success=False)
    out.mkdir(parents=True)
    with (out/'QUERY_RESULTS_OFFLINE_ONLY.csv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator='\n');w.writeheader();w.writerows(records)
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(out/'INPUT_PINS.json').write_text(json.dumps(pins,indent=2)+'\n')
    (out/'UNCERTAINTY_QUERIES.json').write_text(json.dumps(dict(uncertainty),separators=(',',':'))+'\n')
    print(json.dumps(dict(epoch=epoch,status=report['status'],methods=report['methods'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pins',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();execute(json.loads(a.pins.read_text()),a.out)
