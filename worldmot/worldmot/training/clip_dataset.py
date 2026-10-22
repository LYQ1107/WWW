"""Continuous object-level clips; scene/shard boundaries never reset a clip."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

class WorldQueryClipDataset(Dataset):
    def __init__(self,paths,max_observations=None,clip_length=16,scenes=None):
        self.scenes={};self.clip_length=int(clip_length);self.max_observations=max_observations
        paths=[Path(p) for p in paths]; files=[]
        for p in paths:
            files.extend(sorted(p.glob('scene_*/shard_*.npz')) if p.is_dir() else [p])
        grouped={}
        for p in files:
            sid=int(p.parent.name.split('_')[-1])
            if scenes is not None and sid not in scenes:continue
            grouped.setdefault(sid,[]).append(p)
        for sid,ps in grouped.items():
            frames=[]
            for p in sorted(ps):
                with np.load(p,allow_pickle=False) as d:
                    if {'gt_xyz','ground_truth_xyz','oracle_xyz'}&set(d.files):raise ValueError('GT geometry forbidden')
                    if 'frame_offsets' not in d.files:raise ValueError('legacy per-frame pilot cache is not a full temporal cache')
                    offsets=d['frame_offsets'];arr={k:np.array(d[k]) for k in d.files if k not in ('frame_offsets','timestamps')}
                    for i,timestamp in enumerate(d['timestamps']):
                        frame={k:v[int(offsets[i]):int(offsets[i+1])] for k,v in arr.items()}
                        frame['timestamp']=int(timestamp);frames.append(frame)
            times=[r['timestamp'] for r in frames]
            if times!=sorted(set(times)):raise ValueError(f'noncausal/duplicate cached timestamps in scene {sid}')
            self.scenes[sid]=frames
        self.set_clip_length(self.clip_length)
    def set_clip_length(self,length):
        self.clip_length=int(length)
        self.clips=[(sid,start,min(len(frames),start+self.clip_length)) for sid,frames in sorted(self.scenes.items()) for start in range(0,len(frames),self.clip_length)]
    def __len__(self):return len(self.clips)
    def __getitem__(self,index):
        sid,start,end=self.clips[index];frames=self.scenes[sid][start:end]
        return {'scene':sid,'frames':frames,'start':start,'end':end}

def collate_clips(rows):
    batch=len(rows);steps=max(len(r['frames']) for r in rows);n=max(1,max((len(f['observations']) for r in rows for f in r['frames']),default=0))
    out={'observations':torch.zeros(batch,steps,n,532),'geometry':torch.zeros(batch,steps,n,18),
         'camera_ids':torch.zeros(batch,steps,n,dtype=torch.long),'observation_valid':torch.zeros(batch,steps,n,dtype=torch.bool),
         'identity':torch.full((batch,steps,n),-2,dtype=torch.long),'world_xyz':torch.zeros(batch,steps,n,3),
         'geometry_valid':torch.zeros(batch,steps,n,dtype=torch.bool),
         'world_covariance':torch.zeros(batch,steps,n,3,3),'time_valid':torch.zeros(batch,steps,dtype=torch.bool),
         'timestamp':torch.zeros(batch,steps,dtype=torch.long),'scene':torch.tensor([r['scene'] for r in rows])}
    for b,row in enumerate(rows):
        for t,f in enumerate(row['frames']):
            m=len(f['observations']);out['time_valid'][b,t]=True;out['timestamp'][b,t]=f['timestamp']
            for dest,source in [('observations','observations'),('geometry','geometry'),('camera_ids','camera_ids'),('observation_valid','observation_valid'),('identity','identity'),('world_xyz','xyz_world'),('world_covariance','covariance'),('geometry_valid','geometry_valid')]:
                out[dest][b,t,:m]=torch.from_numpy(np.nan_to_num(f[source]) if np.issubdtype(f[source].dtype,np.floating) else f[source])
    return out
