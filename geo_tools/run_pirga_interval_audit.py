"""Preregistered bounded-action diagnostic, complete DEV scores before GT."""
import argparse,csv,json,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn.functional as F
from cv_pil_dev12_shards import NativeShardAssets
from cv_pil_v2_model import tensorize
from pirga_candidate_state import candidate_state
from pirga_interval_opportunity import interval_opportunity
from pirga_indexed_store import sha
from run_pirga_phase1_replay import primary_solve


def run(protocol_path,protocol_sha,out):
    if out.exists():raise FileExistsError(out)
    if sha(protocol_path)!=protocol_sha:raise ValueError('Unpinned action diagnostic')
    p=json.loads(protocol_path.read_text());repo=Path(__file__).resolve().parents[1]
    for name,digest in p['source_sha256'].items():
        if sha(repo/name)!=digest:raise ValueError('Diagnostic source changed')
    torch.set_num_threads(1);start=time.monotonic();assets=NativeShardAssets(**p['release'])
    if assets.engineering_only:raise ValueError('Prefix is not full diagnostic')
    matrices={};caps={};actions={}
    with torch.no_grad():
        for cid,c in assets.calls.items():
            if c['stage']=='run_memory_tracker' or not c['queries']:continue
            actor=assets.actor((12,*c['queries'][0]));b=tensorize(actor);lags=torch.tensor([r['lag'] for rows in actor['past_histories'] for r in rows],dtype=torch.float32)
            st=candidate_state(b.histories,b.owners,lags,b.cross_view,b.destination_camera,len(b.native_scores),p['horizon_boundaries'])
            rows=[]
            for i,key in enumerate(c['query_feature_keys']):
                q=F.normalize(assets.feature(key).float().clone(),dim=0);rows.append(c['track_scores'][i]+2*(st.mean@q))
            base=torch.stack(rows).numpy();bound=(.5*st.counts/(st.counts+1)).numpy()[None,:]
            # Widen intervals conservatively for FP32 final addition and bound
            # arithmetic. Exclusion is not claimed at floating-point ties.
            slack=8*np.finfo(np.float32).eps*(np.abs(base.astype(np.float64))+1)
            matrices[cid]=base;caps[cid]=np.broadcast_to(bound,base.shape).astype(np.float64)+slack
            actions[cid]=primary_solve(torch.from_numpy(base),c['track_ids'],c['thresholds'])
            if len(matrices)%500==0:print(json.dumps(dict(scored_calls=len(matrices),GT_semantics_loaded=False)),flush=True)
    reference=p['reference']
    if sha(reference['path'])!=reference['sha256']:raise ValueError('C1 control changed')
    with Path(reference['path']).open() as f:old={tuple(json.loads(r['key'])):r for r in csv.DictReader(f)}
    if set(old)!=set(assets.query_index):raise ValueError('Scope drift')
    for key,(cid,row) in assets.query_index.items():
        expected=old[key]['C1_accepted_id'];actual=actions[cid][row]
        if expected!=('' if actual is None else str(actual)):raise ValueError('C1 original action drift')
    labels=assets.supervision();events=[]
    for key,(cid,row) in assets.query_index.items():
        label=labels[key]
        if label['actual_primary_state'] not in ('wrong_link','missed_known_positive'):continue
        c=assets.calls[cid];positive=label['certified_target_indices'] if label['GT_person'] is not None else []
        c1fixed=actions[cid][row] is not None and c['track_ids'].index(actions[cid][row]) in positive
        diagnostic=interval_opportunity(matrices[cid],caps[cid],c['thresholds'],row,positive)
        events.append(dict(key=list(key),person=label['GT_person'],original_state=label['actual_primary_state'],pure_positive_candidates=len(positive),C1_original_error_repaired=c1fixed,**diagnostic))
    remaining=[v for v in events if not v['C1_original_error_repaired'] and v['pure_positive_candidates']]
    result=dict(status='BOUNDED_NATIVE_ACTION_OPPORTUNITY_DIAGNOSTIC_ONLY',original_errors=len(events),original_errors_with_legal_pure_positive=sum(bool(v['pure_positive_candidates']) for v in events),
        C1_unrepaired_pure_positive_errors=len(remaining),C1_unrepaired_interval_certified_impossible=sum(v['certified_impossible'] for v in remaining),
        C1_unrepaired_arbitrary_interval_witnesses=sum(v['interval_witness_found'] for v in remaining),
        C1_unrepaired_interval_witness_person_groups=len({v['person'] for v in remaining if v['interval_witness_found']}),
        reasons=dict(Counter(v['reason'] for v in remaining)),events=events,full_scope_C1_actions_exact=True,all_scores_and_bounds_before_GT=True,
        bounds='Frozen new head absolute edge residual<=0.5*history_count/(history_count+1), widened by8 FP32 eps*(abs(C1score)+1) for numerical conservatism',
        models_in_scope='New Phase2 bounded innovations and ordinary D1/D2; original Phase1 C3 and unbounded aggregation replacements not subject to this residual contract',
        interpretation='Interval witnesses use GT-assisted arbitrary edge changes in a superset of model expressivity, not learned repairs or online scores. Necessary exclusion failure does not prove reachability or learnability. Never retune bounds from these results.',
        protocol_sha256=protocol_sha,wall_seconds=time.monotonic()-start,GPU_wall_hours=0,scientific_success=False,online_admitted=False)
    out.mkdir(parents=True);(out/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');(out/'INPUT_PINS.json').write_text(json.dumps(p,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='events'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.protocol,a.protocol_sha,a.out)
