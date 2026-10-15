"""Quick, GT-only *offline* screening of camera-conditioned latent identity prediction.

Uses existing original-TRAIN compact crops from the VGGT/GMT study. It does NOT
run native tracking or prove online HOTA, and does not download/build a dataset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = Path(os.environ.get('GEO_RUNTIME', '/data1/liuyeqiang/WWW_vggt_geo_runtime'))
PREFIX = 'history_train_keyframe_v1_v'
FIT_VIDEOS = (5, 8)
DIAGNOSTIC_VIDEO = 12
SEED = 20261015


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_cache(runtime, video):
    folder = runtime / 'history_train' / (PREFIX + str(video))
    manifest_path = folder / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError(f'Existing compact cache missing: {manifest_path}; STOP, do not recollect videos')
    m = json.loads(manifest_path.read_text())
    if not (m['video_id'] == video and m['source_split'] == 'association_train'
            and m['neutral_prefix40_exact'] and not m['encoder_failures']):
        raise ValueError('Cache provenance/neutrality gate failed')
    if m.get('fit_video_ids') != [5, 8] or m.get('validation_video_ids') != [12]:
        raise ValueError('Cache split changed; training/diagnostic split is frozen')
    for name in ['targets.npz', 'observations.json']:
        if digest(folder / name) != m['artifacts'][name]:
            raise ValueError(f'Cache SHA mismatch: {name}')
    meta = json.loads((folder / 'observations.json').read_text())
    with np.load(folder / 'targets.npz', allow_pickle=False) as z:
        labels = np.asarray(z['labels'], dtype=np.int64).copy()
        original = np.asarray(z['gmt'][:, :1024], dtype=np.float32).copy()
        teacher = {key: np.asarray(z[key], dtype=np.float32).copy()
                   for key in ('dinov2', 'vggt') if key in z.files}
    if len(meta) != len(labels) or len(meta) != len(original) or original.shape[1] != 1024:
        raise ValueError('Observation/feature/label alignment or GMT visual dim mismatch')
    for name, a in [('gmt', original), *teacher.items()]:
        if len(a) != len(labels) or not np.isfinite(a).all():
            raise ValueError(f'Invalid feature tensor: {name}')
    if any(int(x['camera']) not in (1, 2) or int(x['frame']) < 0 for x in meta):
        raise ValueError('Unrecognized camera/frame metadata')
    def norm(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    original = norm(original)
    teacher = {name: norm(a) for name, a in teacher.items()}
    return dict(video=video, meta=meta, labels=labels, gmt=original, teacher=teacher,
                provenance=dict(manifest=str(manifest_path), manifest_sha256=digest(manifest_path),
                                targets_sha256=m['artifacts']['targets.npz'],
                                observations_sha256=m['artifacts']['observations.json'],
                                source_split=m['source_split'], observations=len(meta)))


def build_episodes(item, max_gap=200, history_k=4, bin_width=10):
    """Strictly past opposite-view, GT-grouped candidate histories. Offline ONLY."""
    obs, labels = item['meta'], item['labels']
    buckets = defaultdict(list)
    episodes, represented = [], set()
    for index in sorted(range(len(obs)), key=lambda i: (int(obs[i]['frame']), int(obs[i]['camera']), i)):
        row = obs[index]
        t, cam, person = int(row['frame']), int(row['camera']), int(labels[index])
        if person >= 0:
            # Source frames must be strictly earlier: synchronized cross-camera pairs cannot leak.
            candidates = {}
            for identity, indices in buckets.items():
                valid = [i for i in indices if int(obs[i]['camera']) != cam
                         and 0 < t-int(obs[i]['frame']) <= max_gap]
                if valid:
                    candidates[identity] = valid[-history_k:]
            ids = sorted(candidates)
            key = (person, cam, t // bin_width)
            if len(ids) >= 2 and person in candidates and key not in represented:
                represented.add(key)
                episode = dict(video=item['video'], query=index, candidate_ids=ids,
                               candidate_rows=[candidates[g] for g in ids],
                               target=ids.index(person), target_id=person, camera=cam,
                               frame=t, independent_group=[item['video'], person],
                               has_past_only=True)
                episodes.append(episode)
            buckets[person].append(index)
    return episodes


def norm(x):
    return F.normalize(x, dim=-1, eps=1e-8)


def tensor_episode(data, e, device, kind='gmt', max_gap=200):
    vectors = data['gmt'] if kind == 'gmt' else data['teacher'][kind]
    query = torch.from_numpy(vectors[e['query']]).to(device)
    proto = torch.stack([torch.from_numpy(vectors[rows].mean(0)).to(device)
                         for rows in e['candidate_rows']])
    last = [int(data['meta'][rows[-1]]['frame']) for rows in e['candidate_rows']]
    age = torch.tensor([(e['frame']-t)/max_gap for t in last], device=device, dtype=torch.float32)
    return norm(query), norm(proto), age


class LowRankPredictor(nn.Module):
    """Frozen GMT embedding + small trainable camera/temporal residual."""
    def __init__(self, dim=1024, rank=16, nonlinear=True, use_camera=True):
        super().__init__()
        self.use_camera = use_camera
        self.nonlinear = nonlinear
        self.down = nn.Linear(dim + 3, rank)
        self.up = nn.Linear(rank, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, source, camera, age):
        k = source.shape[0]
        if not k:
            return source
        condition = torch.zeros(k, 3, dtype=source.dtype, device=source.device)
        if self.use_camera:
            condition[:, int(camera)-1] = 1.
        condition[:, 2] = age.clamp(0, 1)
        z = self.down(torch.cat([source, condition], -1))
        if self.nonlinear:
            z = F.gelu(z)
        residual = .25*torch.tanh(self.up(z))
        return norm(source+residual)


def evaluate(data, episodes, device, model=None, kind='gmt', max_gap=200, temperature=.07):
    n, reciprocal, good, ids = 0, 0., [], set()
    with torch.no_grad():
        for e in episodes:
            q,h,age = tensor_episode(data,e,device,kind,max_gap)
            if model is not None:
                h = model(h,e['camera'],age)
            scores = (h@q).cpu()
            order = torch.argsort(scores,descending=True,stable=True).tolist()
            good.append(order[0]==e['target'])
            reciprocal += 1/(1+order.index(e['target']))
            n+=1
            ids.add(e['target_id'])
    return dict(n=n,persons=len(ids),recall1=sum(good)/n if n else None,
                mrr=reciprocal/n if n else None, correct=good)


def train_arm(arm, data, episodes, device, steps=300, seed=SEED, max_gap=200):
    torch.manual_seed(seed)
    model = LowRankPredictor(nonlinear=(arm!='linear'), use_camera=(arm!='predictive_no_camera')).to(device)
    optim = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    rng = np.random.RandomState(seed)
    indices = rng.randint(len(episodes),size=steps)
    logs = []
    for step,idx in enumerate(indices,1):
        e = episodes[int(idx)]
        q,h,age=tensor_episode(data[e['video']],e,device,max_gap=max_gap)
        optim.zero_grad(set_to_none=True)
        p=model(h,e['camera'],age)
        logits=(p@q)/.07
        ce=F.cross_entropy(logits[None],torch.tensor([e['target']],device=device))
        pred=1-(p[e['target']]*q).sum()
        loss=ce+(.5*pred if arm in ('predictive','predictive_no_camera','teacher_distill') else 0.)
        if arm=='teacher_distill':
            tq,th,_=tensor_episode(data[e['video']],e,device,kind='dinov2',max_gap=max_gap)
            teacher_scores=(th@tq)/.07
            kd=F.kl_div(F.log_softmax(logits,dim=0), F.softmax(teacher_scores,dim=0).detach(),reduction='sum')
            loss=loss+.25*kd
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optim.step()
        if step==1 or step%100==0:
            logs.append(dict(step=step,loss=float(loss.detach().cpu()),
                             ranking_loss=float(ce.detach().cpu()),prediction_loss=float(pred.detach().cpu())))
    return model.eval(),logs


def run_screen(runtime,output,steps=300,max_gap=200,history_k=4,device='auto'):
    started=time.monotonic()
    torch.set_num_threads(1)
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    if device == 'auto':device='cuda' if torch.cuda.is_available() else 'cpu'
    device=torch.device(device)
    caches={v:read_cache(runtime,v) for v in (*FIT_VIDEOS,DIAGNOSTIC_VIDEO)}
    episodes={v:build_episodes(caches[v],max_gap=max_gap,history_k=history_k) for v in caches}
    stats={str(v):dict(observations=len(caches[v]['meta']),persons=len(set(caches[v]['labels'].tolist())-{-1}),
                       causal_episodes=len(episodes[v]),positive_and_negative=True) for v in caches}
    baseline=evaluate(caches[12],episodes[12],device,max_gap=max_gap)
    teacher={}
    for name in sorted(caches[12]['teacher']):
        if all(name in caches[v]['teacher'] for v in caches):
            teacher[name]=evaluate(caches[12],episodes[12],device,kind=name,max_gap=max_gap)
    enough=min(len(episodes[5]),len(episodes[8]))>=3 and len(episodes[12])>=8
    teacher_fit=None
    if enough and all('dinov2' in caches[v]['teacher'] for v in (*FIT_VIDEOS,DIAGNOSTIC_VIDEO)):
        fit_gmt=[];fit_dino=[]
        for v in FIT_VIDEOS:
            fit_gmt.extend(evaluate(caches[v],episodes[v],device,max_gap=max_gap)['correct'])
            fit_dino.extend(evaluate(caches[v],episodes[v],device,kind='dinov2',max_gap=max_gap)['correct'])
        fixes=sum(not x and y for x,y in zip(fit_gmt,fit_dino))
        harms=sum(x and not y for x,y in zip(fit_gmt,fit_dino))
        teacher_fit=dict(fixes=fixes,harms=harms,raw_correct=sum(fit_gmt),teacher_correct=sum(fit_dino),
                         eligible=(fixes>harms and sum(fit_dino)>sum(fit_gmt)))
    result=dict(status='AUDIT_ONLY_INSUFFICIENT_EPISODES' if not enough else 'COMPLETE_OFFLINE_SCREEN',
                eligible=enough,seed=SEED,steps=steps,fit_videos=list(FIT_VIDEOS),diagnostic_video=12,
                split='original TRAIN only; TRAIN12 has historical exposure and is not independent',
                max_gap=max_gap,history_k=history_k,per_video=stats,
                provenance={str(v):caches[v]['provenance'] for v in caches},
                raw_gmt={k:v for k,v in baseline.items() if k!='correct'},
                teacher_only={k:{z:y for z,y in v.items() if z!='correct'} for k,v in teacher.items()},
                arms={},teacher_fit_only=teacher_fit,total_seconds=None,
                limitations=['GT-grouped candidate histories are an offline oracle-gallery diagnostic, not deployable GMT IDs.',
                             'All reference frames strictly precede the query; same-time pairs excluded.',
                             'TRAIN12 previously exposed; no independent generalization or full-image FPS/HOTA.',
                             'Video/person/time-binned queries are dependent; Recall1 is descriptive only.'])
    if enough:
        fit=[e for v in FIT_VIDEOS for e in episodes[v]]
        for arm in ['linear','contrastive','predictive','predictive_no_camera']+(['teacher_distill'] if teacher_fit and teacher_fit['eligible'] else []):
            model,logs=train_arm(arm,caches,fit,device,steps,max_gap=max_gap)
            values=evaluate(caches[12],episodes[12],device,model,max_gap=max_gap)
            fixes=sum(not a and b for a,b in zip(baseline['correct'],values['correct']))
            harms=sum(a and not b for a,b in zip(baseline['correct'],values['correct']))
            result['arms'][arm]=dict(**{k:v for k,v in values.items() if k!='correct'},
                                    delta_recall1=values['recall1']-baseline['recall1'],fixes=fixes,harms=harms,
                                    train_logs=logs,parameters=sum(p.numel() for p in model.parameters()))
            output.mkdir(parents=True,exist_ok=True)
            torch.save(dict(arm=arm,seed=SEED,steps=steps,state_dict=model.cpu().state_dict(),
                            source_provenance=result['provenance']),output/(arm+'.pth'))
        result['predictive_vs_contrastive_delta']=result['arms']['predictive']['recall1']-result['arms']['contrastive']['recall1']
        result['predictive_vs_no_camera_delta']=result['arms']['predictive']['recall1']-result['arms']['predictive_no_camera']['recall1']
        best_predictive=max((r['recall1'] for name,r in result['arms'].items() if name in ('predictive','teacher_distill')),default=-1.)
        best_plain=max(baseline['recall1'],result['arms']['linear']['recall1'],result['arms']['contrastive']['recall1'])
        best_has_fixes=any(result['arms'][name]['fixes']>result['arms'][name]['harms']
                           for name in ('predictive','teacher_distill') if name in result['arms'])
        result['deployment_gate']='PENDING_FULL_NATIVE_VALIDATION' if best_predictive>best_plain and best_has_fixes else 'NO_PREDICTIVE_SPECIFIC_SCREENING_GAIN'
    else:
        result['deployment_gate']='INSUFFICIENT_CAUSAL_CROSS_VIEW_EPISODES'
    result['total_seconds']=time.monotonic()-started
    output.mkdir(parents=True,exist_ok=True)
    path=output/'screen.json'
    if path.exists():raise FileExistsError(f'Result exists, refusing to overwrite: {path}')
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return result


def self_test():
    rng=np.random.RandomState(19)
    meta=[];labels=[];emb=[]
    for frame in range(1,101,3):
        for cam in (1,2):
            for identity in (7,8,9):
                v=np.eye(3,16)[identity-7]+.03*rng.randn(16)
                meta.append(dict(frame=frame,camera=cam))
                labels.append(identity);emb.append(v)
    arr=np.asarray(emb,dtype=np.float32);arr/=np.maximum(np.linalg.norm(arr,axis=1,keepdims=True),1e-8)
    d=dict(video=5,meta=meta,labels=np.asarray(labels),gmt=arr,teacher={})
    ep=build_episodes(d,max_gap=50,history_k=3,bin_width=10)
    assert ep and all(e['has_past_only'] for e in ep)
    for e in ep:
        assert all(meta[i]['frame']<e['frame'] and meta[i]['camera']!=e['camera']
                   for rows in e['candidate_rows'] for i in rows)
        assert e['target_id'] in e['candidate_ids'] and len(e['candidate_ids'])>=2
    dev=torch.device('cpu')
    _,h,gap=tensor_episode(d,ep[0],dev)
    head=LowRankPredictor(dim=16)
    assert torch.allclose(head(h,ep[0]['camera'],gap),h,atol=1e-6)
    report=evaluate(d,ep,dev)
    assert report['recall1']>.95
    print(json.dumps(dict(self_test='PASS',episodes=len(ep),raw_recall1=report['recall1'],causal=True)))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--runtime',type=Path,default=DEFAULT_RUNTIME)
    p.add_argument('--out',type=Path,default=None)
    p.add_argument('--steps',type=int,default=300)
    p.add_argument('--max-gap',type=int,default=200)
    p.add_argument('--history-k',type=int,default=4)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    a=p.parse_args()
    if a.self_test:self_test();return
    if a.steps<=0 or a.steps>300:raise ValueError('Only the frozen <=300-step quick screen is supported')
    out=a.out or (Path('/home/liuyeqiang/WWW_cvpil_fastprobe_20261015')/f'fixed_seed{SEED}_step{a.steps}')
    if out.exists():raise FileExistsError(f'Refusing to overwrite old results: {out}')
    import shutil
    parent=out.parent
    parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(parent).free < 30*2**30:
        raise RuntimeError('Free-space safety threshold <30 GiB; stop before writing')
    r=run_screen(a.runtime,out,a.steps,a.max_gap,a.history_k,a.device)
    print(json.dumps(dict(status=r['status'],episodes=r['per_video'],baseline=r['raw_gmt'],
                          arms={k:{j:v for j,v in x.items() if j in ('recall1','delta_recall1','fixes','harms')} for k,x in r['arms'].items()},
                          deployment_gate=r['deployment_gate'],report=str(out/'screen.json')),indent=2))


if __name__ == '__main__':
    main()
