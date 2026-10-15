"""Native-candidate CV-PIL fast screen using pre-existing TRAIN events.pt.

No online assignment or dataset construction. Offline GT is used only for labels;
all negatives/positives come from the original tracker's actual past candidates.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from quick_cv_pil_probe import (SEED, DEFAULT_RUNTIME, FIT_VIDEOS, DIAGNOSTIC_VIDEO,
    PREFIX, read_cache, tensor_episode, LowRankPredictor, digest, norm)


def load_native(runtime, video):
    data = read_cache(runtime, video)
    folder = runtime / 'history_train' / (PREFIX + str(video))
    manifest = json.loads((folder/'manifest.json').read_text())
    path = folder / 'events.pt'
    if digest(path) != manifest['artifacts']['events.pt']:
        raise ValueError('Native event SHA mismatch')
    # SHA-verified, locally created native tensors; do not load untrusted pickle.
    events = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(events,list):
        raise ValueError('Expected original GMT native event list')
    data['events'] = events
    return data


def native_episodes(data, max_gap=200):
    """Same exact GT-eligible native query/reference proposals as GMT, not an oracle gallery."""
    meta, labels = data['meta'], data['labels']
    result = []
    seen = set()
    for ev in data['events']:
        qids, rids, mat = ev['query'], ev['reference'], ev['logits']
        if tuple(mat.shape) != (len(qids),len(rids)):
            raise ValueError('Native GTA logit layout mismatch')
        for qi,query in enumerate(qids):
            if query < 0 or labels[query] < 0:
                continue
            t,cam=int(meta[query]['frame']),int(meta[query]['camera'])
            by_frame={}
            for col,ref in enumerate(rids):
                if ref < 0 or labels[ref] < 0:continue
                past=meta[ref];age=t-int(past['frame'])
                if int(past['camera'])==cam or not 0<age<=max_gap:continue
                by_frame.setdefault((int(past['frame']),int(past['camera'])),[]).append((col,ref))
            for (t0,cam0),entries in sorted(by_frame.items()):
                # Avoid repeated query/reference cohorts when original observer repeats a window.
                key=(query,t0,cam0,tuple(ref for _,ref in entries))
                if key in seen:continue
                seen.add(key)
                pos=[i for i,(_,ref) in enumerate(entries) if labels[ref]==labels[query]]
                if len(entries)<2 or len(pos)!=1:continue
                refs=[ref for _,ref in entries]
                logits=mat[qi,[col for col,_ in entries]].detach().float().numpy()
                if not np.isfinite(logits).all():continue
                result.append(dict(video=data['video'],query=query,candidate_rows=[[i] for i in refs],
                                   target=pos[0],target_id=int(labels[query]),camera=cam,frame=t,
                                   native_logits=np.asarray(logits,dtype=np.float32),
                                   independent_group=(data['video'],int(labels[query])),
                                   source_camera=cam0,source_frame=t0))
    return result


def scores(data,e,device,model=None,kind='gmt',alpha=2.0):
    q,h,age=tensor_episode(data,e,device,kind)
    if model is not None:h=model(h,e['camera'],age)
    return torch.from_numpy(e['native_logits']).to(device) + alpha*(h@q)


def report(data,episodes,device,model=None,kind='gmt',alpha=2.0,mode='joint'):
    correct=[];rank=[];unique=set()
    with torch.no_grad():
        for e in episodes:
            if mode=='native':s=torch.from_numpy(e['native_logits']).to(device)
            else:s=scores(data,e,device,model,kind,alpha)
            order=torch.argsort(s,descending=True,stable=True).tolist()
            correct.append(order[0]==e['target']);rank.append(1/(order.index(e['target'])+1))
            unique.add((e['video'],e['query']))
    return dict(clauses=len(correct),unique_queries=len(unique),persons=len(set(e['target_id'] for e in episodes)),
                recall1=float(np.mean(correct)) if correct else None,
                mrr=float(np.mean(rank)) if rank else None,correct=correct)


def fit_model(arm,data,episodes,device,steps):
    torch.manual_seed(SEED)
    model=LowRankPredictor(nonlinear=arm!='linear',use_camera=arm!='predictive_no_camera').to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(SEED)
    history=[]
    for step in range(1,steps+1):
        e=episodes[int(rng.integers(len(episodes)))];data0=data[e['video']]
        q,h,age=tensor_episode(data0,e,device)
        p=model(h,e['camera'],age)
        logits=torch.from_numpy(e['native_logits']).to(device)+2*(p@q)
        primary=F.cross_entropy(logits[None],torch.tensor([e['target']],device=device))
        predictive=1-(p[e['target']]*q).sum()
        loss=primary+(.5*predictive if arm in ('predictive','predictive_no_camera','teacher_distill') else 0.)
        if arm=='teacher_distill':
            tq,th,_=tensor_episode(data0,e,device,kind='dinov2')
            teacher=(th@tq)/.07
            kd=F.kl_div(F.log_softmax(logits,dim=0),F.softmax(teacher,dim=0).detach(),reduction='sum')
            loss=loss+.25*kd
        optim.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optim.step()
        if step==1 or step%100==0:history.append(dict(step=step,loss=float(loss.detach())))
    return model.eval(),history


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runtime',type=Path,default=DEFAULT_RUNTIME)
    p.add_argument('--out',type=Path,default=Path('/home/liuyeqiang/WWW_cvpil_native_20261015'))
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--device',choices=['cuda','cpu','auto'],default='auto')
    a=p.parse_args()
    if not 1<=a.steps<=300:raise ValueError('Fixed bounded screen only, max 300 steps')
    if a.out.exists():raise FileExistsError('Refuse to overwrite any existing results')
    if a.device=='auto':a.device='cuda' if torch.cuda.is_available() else 'cpu'
    device=torch.device(a.device);torch.set_num_threads(1)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(a.out.parent).free < 30*2**30:raise RuntimeError('Insufficient storage, stop')
    data={v:load_native(a.runtime,v) for v in (*FIT_VIDEOS,DIAGNOSTIC_VIDEO)}
    eps={v:native_episodes(data[v]) for v in data}
    res=dict(status='PENDING',provenance={str(v):data[v]['provenance'] for v in data},
             native_event_sha={str(v):digest(a.runtime/'history_train'/(PREFIX+str(v))/'events.pt') for v in data},
             training_videos=list(FIT_VIDEOS),diagnostic_video=DIAGNOSTIC_VIDEO,
             native=report(data[12],eps[12],device,mode='native'),
             native_plus_GMT_cosine=report(data[12],eps[12],device),
             counts={str(v):len(eps[v]) for v in eps},methods={},
             limitations=['Existing native tracker candidate/reference set, but offline single-reference ranking only.',
                          'GT used only for offline labels; no GT inputs to trainable models.',
                          'Same queries can occur in multiple historical frames; dependent samples.',
                          'No native Hungarian/Bank/online HOTA, no independent validation, no full-image FPS.'])
    base=res['native_plus_GMT_cosine']['correct'];raw=res['native']['correct']
    fit=[e for v in FIT_VIDEOS for e in eps[v]]
    if min(len(eps[v]) for v in data)<8 or len({e['target_id'] for e in fit})<2:
        res['status']='INSUFFICIENT_REAL_NATIVE_CLAUSES'
    else:
        teacher_gate=False
        if all('dinov2' in data[v]['teacher'] for v in data):
            native_fit=[];teacher_fit=[]
            for v in FIT_VIDEOS:
                native_fit += report(data[v],eps[v],device,mode='native')['correct']
                teacher_fit += report(data[v],eps[v],device,kind='dinov2')['correct']
            teacher_gate=sum(teacher_fit)>sum(native_fit) and sum(not b and a for a,b in zip(teacher_fit,native_fit))>sum(b and not a for a,b in zip(teacher_fit,native_fit))
        res['teacher_gate_training_only']=teacher_gate
        for arm in ['linear','contrastive','predictive','predictive_no_camera']+(['teacher_distill'] if teacher_gate else []):
            m,logs=fit_model(arm,data,fit,device,a.steps)
            r=report(data[12],eps[12],device,model=m)
            r['fixes_vs_native']=sum(not x and y for x,y in zip(raw,r['correct']))
            r['harms_vs_native']=sum(x and not y for x,y in zip(raw,r['correct']))
            r['fixes_vs_native_plus_cosine']=sum(not x and y for x,y in zip(base,r['correct']))
            r['harms_vs_native_plus_cosine']=sum(x and not y for x,y in zip(base,r['correct']))
            r.pop('correct');r['training_log']=logs
            res['methods'][arm]=r
        ref=max(res['native']['recall1'],res['native_plus_GMT_cosine']['recall1'],
                res['methods']['linear']['recall1'],res['methods']['contrastive']['recall1'])
        candidates=[res['methods'][arm] for arm in ('predictive','teacher_distill') if arm in res['methods']]
        res['status']='COMPLETE_OFFLINE_NATIVE_SCREEN'
        res['gate']='WORTH_NATIVE_CLOSED_LOOP_REVIEW' if any(x['recall1']>ref and x['fixes_vs_native']>x['harms_vs_native'] for x in candidates) else 'NO_UNIQUE_PREDICTIVE_RANKING_GAIN'
    res['native'].pop('correct');res['native_plus_GMT_cosine'].pop('correct')
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/'native_screen.json').write_text(json.dumps(res,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:res[k] for k in ['status','counts']},ensure_ascii=False))
    print('native-screen:',a.out/'native_screen.json')

if __name__=='__main__':main()
