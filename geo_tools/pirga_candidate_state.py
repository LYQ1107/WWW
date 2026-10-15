"""GT-free causal candidate states from the admitted physical-vector index.

Only in-memory derived states are memoized. No vector files are duplicated.
Time is measured in native frame indices; physical seconds are not certified.
"""
from dataclasses import dataclass
from collections import OrderedDict
import json,math,sqlite3
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from pirga_training_cache import NativeTrainingCache
from pirga_indexed_store import sha


@dataclass(frozen=True)
class CandidateState:
    mean:torch.Tensor
    conditions:torch.Tensor
    counts:torch.Tensor
    cross_counts:torch.Tensor
    latest_lag:torch.Tensor


def candidate_state(histories,owners,lags,cross,camera,n,horizon_boundaries):
    """All arguments are inference evidence; query features/GT are absent."""
    if camera not in (1,2) or len(histories)!=len(owners) or len(lags)!=len(owners) or (len(lags) and (lags<=0).any()):raise ValueError('Illegal camera or noncausal history')
    h=histories
    counts=torch.bincount(owners,minlength=n).float()
    summed=torch.zeros(n,1152).index_add_(0,owners,h)
    raw_mean=summed/counts.clamp_min(1)[:,None];mean=F.normalize(summed,dim=-1)
    cross_counts=torch.zeros(n).index_add_(0,owners,cross.float())
    mean_lag=torch.zeros(n).index_add_(0,owners,lags)/counts.clamp_min(1)
    latest=torch.zeros(n);cross_latest=torch.zeros(n)
    for i in range(n):
        if counts[i]>0:latest[i]=lags[owners==i].min()
        cross_mask=(owners==i)&cross
        cross_latest[i]=lags[cross_mask].min() if cross_mask.any() else latest[i]
    conditions=torch.zeros(n,11)
    if camera not in (1,2):raise ValueError('Physical camera contract changed')
    conditions[:,camera-1]=1
    conditions[:,2]=cross_counts/counts.clamp_min(1)
    conditions[:,3]=torch.log1p(mean_lag)/math.log(401)
    conditions[:,4]=torch.log1p(cross_latest)/math.log(401)
    conditions[:,5]=torch.log1p(counts)/math.log(65)
    conditions[:,6]=raw_mean.norm(dim=-1)
    bins=torch.bucketize(cross_latest,torch.tensor(horizon_boundaries,dtype=latest.dtype),right=False)
    conditions[:,7:]=F.one_hot(bins,4).float()
    return CandidateState(mean,conditions,counts,cross_counts,latest)


class CandidateStates:
    def __init__(self,cache,horizon_boundaries,max_cached_calls=3000):
        if list(horizon_boundaries)!=sorted(set(horizon_boundaries)) or len(horizon_boundaries)!=3:
            raise ValueError('Three fixed increasing TRAIN horizon boundaries required')
        if cache.device.type!='cpu':raise ValueError('Candidate-state executor is CPU-only')
        self.cache=cache;self.boundaries=list(horizon_boundaries);self.memo=OrderedDict();self.max_cached=max_cached_calls;self.frames={}
        for scene,spec in cache.manifest['stores'].items():
            source=json.loads((Path(spec['root'])/'manifest.json').read_text())
            if sha(Path(spec['root'])/'index.sqlite')!=source['artifacts']['index.sqlite']:raise ValueError('Native metadata database changed')
            db=sqlite3.connect('file:'+str(Path(spec['root'])/'index.sqlite')+'?mode=ro',uri=True)
            rows=list(db.execute('SELECT id,frame FROM features ORDER BY id'));db.close()
            if [r[0] for r in rows]!=list(range(len(rows))):raise ValueError('Non-contiguous physical index')
            self.frames[int(scene)]=torch.tensor([r[1] for r in rows],dtype=torch.long)

    def state(self,ci):
        if ci in self.memo:
            self.memo.move_to_end(ci);return self.memo[ci]
        c=self.cache.calls[ci];frames=self.frames[c['scene']];frame=frames[c['query_features'][0]]
        if any(frames[i]!=frame for i in c['query_features']):raise ValueError('Mixed native query times')
        lags=(frame-frames[c['histories'].cpu()]).float()
        if len(lags) and (lags<=0).any():raise ValueError('Future or simultaneous history in predictor')
        h=F.normalize(self.cache.features[c['scene']].index_select(0,c['histories']).float(),dim=-1)
        state=candidate_state(h,c['owners'],lags,c['cross'],c['camera'],c['native_scores'].shape[1],self.boundaries)
        self.memo[ci]=state
        if len(self.memo)>self.max_cached:self.memo.popitem(last=False)
        return state

    def query(self,key):
        ci,row=self.cache.locations[tuple(key)];c=self.cache.calls[ci]
        q=F.normalize(self.cache.features[c['scene']][c['query_features'][row]].float(),dim=0)
        return self.state(ci),q,c['native_scores'][row].clone(),ci,row


def train_horizon_statistics(cache):
    """Input-only lag statistics: count native references, never independent people."""
    frames={}
    for scene,spec in cache.manifest['stores'].items():
        db=sqlite3.connect('file:'+str(Path(spec['root'])/'index.sqlite')+'?mode=ro',uri=True)
        rows=list(db.execute('SELECT id,frame FROM features ORDER BY id'));db.close()
        frames[int(scene)]=np.array([r[1] for r in rows],dtype=np.int64)
    nearest=[];cross_nearest=[];by_scene={}
    for c in cache.calls:
        f=frames[c['scene']];lags=f[c['query_features'][0]]-f[c['histories'].numpy()]
        if len(lags) and (lags<=0).any():raise ValueError('Noncausal TRAIN time')
        owners=c['owners'].numpy();cross=c['cross'].numpy();local=[]
        for i in np.unique(owners):
            mask=owners==i;v=int(lags[mask].min());nearest.append(v);local.append(v)
            if cross[mask].any():cross_nearest.append(int(lags[mask&cross].min()))
        by_scene.setdefault(str(c['scene']),[]).extend(local)
    values=np.asarray(cross_nearest)
    raw=np.quantile(values,[.25,.5,.75],method='nearest').astype(int).tolist();bounds=[]
    for v in raw:bounds.append(max(v,bounds[-1]+1 if bounds else 1))
    return dict(status='TRAIN_INPUT_ONLY_HORIZONS_FROZEN',GT_used=False,DEV_read=False,time_unit='native frame index; not physical seconds',
        unique_calls=len(cache.calls),candidate_call_states=len(nearest),cross_view_supported_states=len(values),
        quantile_rule='nearest empirical25/50/75 percentiles of latest strict-past cross-camera lag in cross-view-supported legal candidate-call states; ties made increasing by+1frame',
        raw_quantiles=raw,horizon_boundaries=bounds,max_latest_lag=max(nearest),max_cross_camera_latest_lag=int(values.max()),by_scene_candidate_call_states={k:len(v) for k,v in by_scene.items()},
        independence_warning='Native candidate-call states share observations and identities; counts are not independent people.')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--cache-sha',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    torch.set_num_threads(1);cache=NativeTrainingCache(a.cache,a.cache_sha)
    result=train_horizon_statistics(cache);result.update(cache_path=str(a.cache),cache_sha256=a.cache_sha,source_sha256=sha(__file__),GPU_wall_hours=0)
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
