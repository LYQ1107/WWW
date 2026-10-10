"""Five GT-free admission rules on identical, actually executed raw V3 commits."""
from jev_phase16_common import *
import argparse
import collections
import gzip
import time
import torch
from jev_phase13_runtime import cache_inputs
from jev_phase16_prototype_audit import TraceCursor, quantile
from gtr.modeling.jev_phase16.safe_memory_admission import SafeAdmission, RULES
from gtr.modeling.jev_native_state import fingerprint


def main(video):
    protect();storage_guard();assert video in TRAIN+DEV;torch.set_num_threads(1)
    _,frames,reader=cache_inputs(video);labels=duplicate_masked_labels(video,reader)
    p0=OUT/'P0_r2/full/v3'/f'video{video:02d}/RESULT.json';trace=read(p0)['cases'][0]['queries']
    assert sha(trace['path'])==trace['SHA256'];cursor=TraceCursor(trace['path'])
    out=OUT/'P2_shadow'/f'video{video:02d}';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'RESULT.json').exists()
    rules={r:SafeAdmission(r) for r in RULES};raw=collections.defaultdict(list)
    targets=collections.defaultdict(list);counts=collections.defaultdict(collections.Counter)
    timing=collections.defaultdict(list);owner={};votes=collections.defaultdict(collections.Counter)
    clusters=collections.defaultdict(set);total=0;start=time.monotonic()
    # Native bootstrap is not a scored query/COMMIT trace row.
    payloads=[reader.load(video,0,v) for v in [0,1]]
    bootstrap=max((len(payloads[v]['pred_boxes']),v) for v in [0,1])[1]
    keys=[(0,bootstrap)]+[(f,v) for f in range(frames) for v in [0,1] if (f,v)!=(0,bootstrap)]
    with gzip.open(out/'ADMISSION_EVENTS.jsonl.gz','wt') as stream:
        for frame,view in keys:
            p=reader.load(video,frame,view);gts=labels.current(frame,view)
            records=[] if (frame,view)==(0,bootstrap) else [r for r in cursor.at((video,frame,view)) if r.get('record_type')=='COMMIT']
            ids=list(range(1,len(p['pred_boxes'])+1)) if (frame,view)==(0,bootstrap) else [r['committed_ID'] for r in records]
            assert len(ids)==len(p['pred_boxes'])==len(gts)
            for row,(identity,gt) in enumerate(zip(ids,gts)):
                raw[identity].append(p['reid_features'][row,:1024].clone());targets[identity].append(gt)
                for rule,admission in rules.items():
                    t=time.perf_counter();indices,event=admission.append(identity,p['pred_boxes'][row],p['image_size'],frame,view,
                        lambda j:raw[identity][j])
                    timing[rule].append((time.perf_counter()-t)*1000)
                    c=counts[rule];c['raw_observations']+=1;c['admitted_observations']+=len(indices)
                    c['current_GT_known']+=gt is not None;c['current_provisional']+=event['status']=='PROVISIONAL'
                    c['current_quarantined']+=event['status']=='QUARANTINED'
                    c['pending_peak']=max(c['pending_peak'],event['pending_count']);c['quarantine_peak']=max(c['quarantine_peak'],event['quarantine_count'])
                    for j in indices:
                        target=targets[identity][j];c['admitted_GT_known']+=target is not None
                        wrong=target is not None and identity in owner and owner[identity]!=target
                        c['known_owner_wrong_admitted']+=wrong
                        if wrong:clusters[rule+'_wrong_admitted'].add((video,target,admission.records[identity][j]['view'],admission.records[identity][j]['frame']//64))
                    selected,fallback=admission.matching_indices(identity);c['no_confirmed_current_fallback']+=fallback
                    tv=collections.Counter(targets[identity][j] for j in selected if targets[identity][j] is not None)
                    c['mixed_matching_view_after_write']+=len(tv)>1
                    c['known_owner_wrong_blocked_current']+=gt is not None and identity in owner and owner[identity]!=gt and len(raw[identity])-1 not in indices
                    stream.write(json.dumps(dict(key=[video,frame,view],row=row,rule=rule,GT_OFFLINE_ONLY=gt,**event),allow_nan=False)+'\n')
                if gt is not None:votes[identity][gt]+=1
                if identity not in owner and len(votes[identity])==1 and sum(votes[identity].values())>=3 and sum(votes[identity].values())/len(raw[identity])>=.8:
                    owner[identity]=next(iter(votes[identity]))
                total+=1
            if frame%32==0:
                save(out/'PROGRESS.json',dict(status='SHADOW_ADMISSION',key=[video,frame,view],frames=frames,observations=total,seconds=time.monotonic()-start))
                stream.flush()
    cursor.stream.close()
    result=dict(status='COMPLETE',binding=binding(inputs=[ref(p0),trace,ref(REPORTS/'P2_P3_PROTOCOL.json')],
        evaluator='actual commit replay; GT only after each admission decision',scope='shadow write diagnostics, no tracking causality or deployment FPS'),
        video=video,frames=frames,strict_train_gate_eligible=video in TRAIN and video!=14,
        counts={r:dict(c) for r,c in counts.items()},CPU_admission_ms={r:quantile(v) for r,v in timing.items()},
        clusters={r:[list(x) for x in sorted(v)] for r,v in clusters.items()},
        state_SHA256={r:fingerprint(a.state_dict()) for r,a in rules.items()},events=ref(out/'ADMISSION_EVENTS.jsonl.gz'),
        raw_native_commits_unchanged=True,GT_actor_inputs=False,seconds=time.monotonic()-start)
    save(out/'RESULT.json',result);save(out/'PROGRESS.json',dict(status='COMPLETE',frames=frames,observations=total))
    print('PHASE16_ADMISSION_SHADOW_COMPLETE',video,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);main(p.parse_args().video)
