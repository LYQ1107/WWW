"""Bounded diagnostic rollouts with full future traces and frozen OFF snapshots."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import datetime as dt
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
for p in (ROOT,ROOT/'reproduction_tools',ROOT/'third_party/CenterNet2'):sys.path.insert(0,str(p))
from audit_jev_phase5 import BASE, OUT, CACHE, TRAIN, records, key, sha
from run_segmented_small_gate import atomic, trace_path


def event_key(e):
    return int(e['_frame']),int(e['_view']),str(e['question']),int(e['context'].get('detection_index',-1))


def prepare(task):
    from gtr.modeling.jev_perception_cache import FrozenPerceptionCache
    cache=FrozenPerceptionCache(CACHE)
    source=json.loads((BASE/'plan.json').read_text())
    selected=[]
    if task=='memory':
        last=max(k[1] for k in cache.keys() if k[0]==6)
        eligible=sorted([r for r in records(6) if r['question_type']=='MEMORY_DECISION' and key(r)[0]+32<=last],key=key)
        indices=np.round(np.linspace(0,len(eligible)-1,64)).astype(int)
        assert len(set(indices))==64
        selected=[(6,key(eligible[i])) for i in indices]
    elif task=='match_relabel':
        selected=[(v,key(r)) for v in (6,7) for r in records(v) if r['question_type']=='MATCH_DECISION']
    from build_jev_counterfactual_v2 import ordered_production_keys
    indices_by_video={}
    for v in sorted({v for v,k in selected}):
        ordered,_=ordered_production_keys([k for k in cache.keys() if k[0]==v],lambda k:cache.load(*k))
        indices_by_video[v]={k:i for i,k in enumerate(ordered)}
    grouped=defaultdict(list)
    for v,k in selected:
        index=indices_by_video[v][(v,k[0],k[1])]
        chunk=next(c for c in source['chunks'] if c['video_id']==v and c['key_start']<=index<c['key_end'])
        grouped[chunk['name']].append(list(k))
    chunks=[]
    for name,keys in sorted(grouped.items()):
        c=next(c for c in source['chunks'] if c['name']==name)
        chunks.append({**c,'selected_events':keys,'trace':str(trace_path(c['video_id']))})
    plan={'status':'PREDECLARED','task':task,'source_commit':__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'source_plan_sha256':sha(BASE/'plan.json'),'horizons':[8,16,32] if task=='memory' else [8],
          'selected_records':len(selected),'selected_keys_sha256':__import__('hashlib').sha256(json.dumps(selected).encode()).hexdigest(),
          'chunks':chunks,'source_snapshot_commit':source['binding']['source_commit'],'full_trace_and_future_maps_retained':True,
          'max_events_used':False,'official_test_read':False,'selection':'predeclared evenly spaced 64 complete-future-32 video06 MEMORY events' if task=='memory' else 'all frozen complete-video06/video07 MATCH events; GT frame correction only',
          'label_protocol':'legacy frozen fixed-OFF continuation; aligned GT coordinate outcomes also recorded separately',
          'created_utc':dt.datetime.now(dt.timezone.utc).isoformat()}
    atomic(OUT/task/'plan.json',plan)
    print(json.dumps({'task':task,'selected_records':len(selected),'chunks':len(chunks)}))


def worker(task,slot,slots):
    from build_jev_counterfactual_dataset import load_gt
    from build_jev_counterfactual_v2 import build_v2_records,score_rollout
    from gtr.modeling.jev_perception_cache import FrozenPerceptionCache
    from jev_intra_video_chunking import load_state_snapshot
    from run_jev_full_h8_fast_worker import PayloadLRU
    from run_segmented_small_gate import engine
    plan=json.loads((OUT/task/'plan.json').read_text())
    model=engine();cache=FrozenPerceptionCache(CACHE);lru=PayloadLRU(cache,max_entries=512)
    gt=load_gt(TRAIN);aligned_gt=(gt[0],{(v,w,f-1):i for (v,w,f),i in gt[1].items()},gt[2],gt[3])
    horizons=plan['horizons'];started=time.monotonic()
    for idx,chunk in enumerate(plan['chunks']):
        if idx%slots!=slot:continue
        directory=OUT/task/'chunks'/chunk['name'];directory.mkdir(parents=True,exist_ok=True)
        if (directory/'manifest.json').exists():continue
        state,metadata=load_state_snapshot(Path(chunk['snapshot']))
        assert sha(chunk['snapshot'])==chunk['snapshot_sha256'].removeprefix('sha256:')
        assert metadata['source_commit']==plan['source_snapshot_commit']
        wanted={tuple(k) for k in chunk['selected_events']};aligned={};count=0;last=0
        def progress(fields=None,force=False):
            nonlocal last
            stamp=time.monotonic()
            if force or stamp-last>2:
                atomic(directory/'progress.json',{'pid':os.getpid(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'completed':count,'expected':len(wanted),'elapsed_seconds':stamp-started,'updated_utc':dt.datetime.now(dt.timezone.utc).isoformat(),**(fields or {})});last=stamp
        def observe(e,action,steps,hs):
            aligned.setdefault(event_key(e),{})[action]={str(h):score_rollout(steps,images=aligned_gt[1],gt_by_image=gt[2],image_meta=gt[3],start_frame=e['_frame'],max_horizon=h) for h in hs}
        with (directory/'records.jsonl.partial').open('w') as handle:
            def sink(r):
                nonlocal count
                aligned_outcomes=aligned.pop(key(r))
                r['aligned_horizon_outcomes']={str(h):{a:values[str(h)] for a,values in aligned_outcomes.items()} for h in horizons}
                handle.write(json.dumps(r,sort_keys=True,allow_nan=False)+'\n');count+=1;progress({'phase':'record_complete'});handle.flush()
            progress({'phase':'starting'},True)
            _,stats,skipped=build_v2_records(trace=Path(chunk['trace']),cache_root=CACHE,annotations=TRAIN,
                checkpoint_hash='cd72823824d16c86ed27c2dfc8323de610aa9f6c0c0b29249aa3de609deabce8',horizon=max(horizons),derive_horizons=horizons,
                association_backend='formal_gmt_transformer',engine=model,cache_obj=lru,gt_bundle=gt,
                cache_keys_by_video={chunk['video_id']:[k for k in cache.keys() if k[0]==chunk['video_id']]},record_sink=sink,video_ids=[chunk['video_id']],initial_state=state,
                key_start_index=chunk['key_start'],key_end_index=chunk['key_end'],selected_key_range=(chunk['key_start'],chunk['key_end']),
                progress_callback=progress,emit_event_filter=lambda e:event_key(e) in wanted,rollout_observer=observe)
            assert skipped==0 and count==len(wanted),(stats,skipped,count,len(wanted))
        os.replace(directory/'records.jsonl.partial',directory/'records.jsonl')
        atomic(directory/'manifest.json',{'status':'COMPLETE','records':count,'sha256':sha(directory/'records.jsonl'),'source_commit':plan['source_commit'],'snapshot_sha256':chunk['snapshot_sha256']})
        progress({'phase':'COMPLETE'},True)
        print(json.dumps({'chunk':chunk['name'],'records':count}),flush=True)


def aggregate(task):
    plan=json.loads((OUT/task/'plan.json').read_text());all_rows=[]
    for c in plan['chunks']:
        d=OUT/task/'chunks'/c['name'];assert json.loads((d/'manifest.json').read_text())['status']=='COMPLETE'
        all_rows.extend(json.loads(l) for l in (d/'records.jsonl').open())
    assert len(all_rows)==plan['selected_records']
    all_rows.sort(key=lambda r:(r['state']['online_context']['video_id'],*key(r)))
    path=OUT/task/'records.jsonl';path.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in all_rows))
    frozen={(v,key(r)):r for v in (6,7) for r in records(v)}
    mismatches=[(r['state']['online_context']['video_id'],key(r)) for r in all_rows if r['horizon_outcomes']['8']!=frozen[(r['state']['online_context']['video_id'],key(r))]['action_outcomes']]
    assert not mismatches,mismatches[:5]
    summary={'status':'COMPLETE','task':task,'records':len(all_rows),'record_sha256':sha(path),'frozen_H8_outcomes_exactly_reproduced':True,'official_test_read':False,
             'protocol_limitations':['Future action categories are frozen OFF trace actions, with assignments recomputed; this is not adaptive controller continuation.','Window utility starts without prior identity/memory anchors; AssA proxy is not TrackEval AssA.'], 'by_gt_contract':{}}
    for contract,field in [('legacy','horizon_outcomes'),('aligned','aligned_horizon_outcomes')]:
        summary['by_gt_contract'][contract]={}
        for h in plan['horizons']:
            counts=Counter();diffs=[]
            for r in all_rows:
                out=r[field][str(h)];values=[v['utility'] for v in out.values()];counts['ties']+=max(values)-min(values)<=1e-8;counts['different_outcomes']+=len({json.dumps(v,sort_keys=True) for v in out.values()})>1
                if task=='memory':diffs.append(out['WRITE_MEMORY']['utility']-out['SKIP_MEMORY']['utility'])
            summary['by_gt_contract'][contract][str(h)]={**counts,'tie_rate':counts['ties']/len(all_rows),'write_minus_skip_min':min(diffs) if diffs else None,'write_minus_skip_max':max(diffs) if diffs else None}
    atomic(OUT/('MEMORY_HORIZON_AUDIT.json' if task=='memory' else 'MATCH_RELABEL_AUDIT.json'),summary)
    print(json.dumps(summary))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','aggregate']);p.add_argument('--task',choices=['memory','match_relabel'],default='memory');p.add_argument('--slot',type=int,default=0);p.add_argument('--slots',type=int,default=1);a=p.parse_args()
    try:
        if a.mode=='worker':worker(a.task,a.slot,a.slots)
        else:globals()[a.mode](a.task)
    except Exception:
        atomic(OUT/a.task/f'worker_{a.slot}_error.json',{'status':'FAILED','traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
