"""Causal trained PWQT inference with observation-initialized births."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from ..models.persistent_world_queries import PersistentWorldQueryTransformer, QueryBatchState
from ..models.world_observation_encoder import WorldObservationToken, observations_to_tensors
from .query_lifecycle import constrained_assign
from .query_memory_bank import QueryMemoryBank

@dataclass
class RuntimeResult:
    timestamp:int
    status:str
    assignments:dict[int,int]
    global_ids_by_observation:dict[int,int]
    births:list[int]
    diagnostics:dict
    query_to_observations:dict[int,list[int]]

def distribution(x):
    a=x.detach().float().reshape(-1);a=a[torch.isfinite(a)&(abs(a)<1e8)]
    if not a.numel():return {'count':0}
    return {'count':a.numel(),'mean':float(a.mean()),'p05':float(torch.quantile(a,.05)),'p50':float(torch.quantile(a,.5)),'p95':float(torch.quantile(a,.95))}

def birth_groups(indices,pair_logits,cameras,threshold):
    """Learned multi-view newborn grouping; one detection per camera."""
    groups=[{o} for o in indices]
    scores=torch.sigmoid(pair_logits.detach()).cpu()
    edges=sorted([(float((scores[i,j]+scores[j,i])/2),i,j) for i in indices for j in indices if i<j and cameras[i]!=cameras[j]],reverse=True)
    for score,i,j in edges:
        if score<threshold:break
        gi=next((g for g in groups if i in g),None);gj=next((g for g in groups if j in g),None)
        if gi is gj:continue
        if {cameras[o] for o in gi}&{cameras[o] for o in gj}:continue
        gi.update(gj);groups.remove(gj)
    return [sorted(g) for g in groups]

class PersistentQueryRuntime:
    def __init__(self,model:PersistentWorldQueryTransformer,device='cpu',checkpoint=None,allow_untrained=False,max_missed=None,
                 null_margin=None,birth_threshold=None,pair_threshold=None):
        self.model=model.to(device).eval();self.device=str(device);self.checkpoint=checkpoint;self.trained=False;self.manifest={}
        if checkpoint:
            saved=torch.load(checkpoint,map_location=device,weights_only=True)
            self.model.load_state_dict(saved.get('model',saved),strict=True);self.trained=True;self.manifest=saved.get('manifest',{})
        if not self.trained and not allow_untrained:raise RuntimeError('PWQT checkpoint required; random weights are smoke-only')
        thresholds=self.manifest.get('thresholds',{})
        self.null_margin=float(thresholds.get('null_margin',0.5) if null_margin is None else null_margin)
        self.birth_threshold=float(thresholds.get('birth_threshold',0.5) if birth_threshold is None else birth_threshold)
        self.pair_threshold=float(thresholds.get('pair_threshold',0.7) if pair_threshold is None else pair_threshold)
        if not 0<self.birth_threshold<1:raise ValueError('birth probability threshold must be strictly between zero and one')
        self.max_missed=int(self.manifest.get('max_missed',64) if max_missed is None else max_missed);self.coordinates_calibrated=False;self.memory=QueryMemoryBank();self.state=None;self.scene_id=None;self.last_timestamp=None
    def reset(self,scene_id=None):
        self.scene_id=scene_id;self.memory.reset();self.state=None;self.last_timestamp=None
    @torch.inference_mode()
    def step(self,tokens:list[WorldObservationToken],timestamp:int)->RuntimeResult:
        if self.last_timestamp is not None and timestamp<=self.last_timestamp:raise ValueError('timestamps must increase inside a scene')
        if tokens:
            obs,geo,cameras,valid=observations_to_tensors(tokens,device=self.device)
        else:
            obs=torch.empty(1,0,self.model.observation_dim,device=self.device);geo=torch.empty(1,0,18,device=self.device);cameras=[];valid=torch.empty(1,0,dtype=torch.bool,device=self.device)
        if obs.shape[-1]!=self.model.observation_dim:raise ValueError('checkpoint/cache feature schema mismatch')
        ct=torch.tensor([cameras],device=self.device,dtype=torch.long)
        if self.state is None:self.state=self.model.init_state(1,obs.device,obs.dtype)
        output=self.model(obs,geo,self.state,valid,camera_ids=ct)
        logits=output['assignment_logits'].clone();birth=output['birth_logits'].clone()
        if valid.numel():
            logits[...,:-1]=logits[...,:-1].masked_fill(~valid[:,None,:],-1e9);birth=birth.masked_fill(~valid,-1e9)
        active=self.state.global_ids>=0
        assigned=constrained_assign(logits,birth,cameras,torch.ones_like(output['state'].existence),self.null_margin,self.birth_threshold,active)
        mapping={q:list(os) for q,os in assigned.query_to_observations.items()};groups=birth_groups(assigned.births,output['pair_logits'][0],cameras,self.pair_threshold)
        free=[q for q in range(self.model.max_world_queries) if int(self.state.global_ids[0,q])<0]
        overflow=max(0,len(groups)-len(free));new_slots=[]
        for group,q in zip(groups,free):mapping[q]=group;new_slots.append(q)
        xyz=torch.tensor(np.asarray([t.xyz_world for t in tokens],np.float32).reshape(1,-1,3),device=self.device)
        cov=torch.tensor(np.asarray([t.covariance for t in tokens],np.float32).reshape(1,-1,3,3),device=self.device)
        gv=torch.tensor([[t.diagnostics.get('geometry_coordinate_valid',bool(t.geometry_reliability)) for t in tokens]],device=self.device,dtype=torch.bool)
        state=self.model.accept_observations(output['state'],output,[mapping],xyz,cov,gv)
        global_by_obs={};birth_ids=[]
        for q,indices in mapping.items():
            gid=int(state.global_ids[0,q])
            if gid<0:
                gid=self.memory.allocate(timestamp,indices[0]);state.global_ids[0,q]=gid;birth_ids.append(gid)
            state.last_seen[0,q]=timestamp
            for o in indices:global_by_obs[o]=gid
        closed=[]
        for q in range(self.model.max_world_queries):
            if int(state.global_ids[0,q])>=0 and int(state.missed[0,q])>self.max_missed:
                gid=int(state.global_ids[0,q]);self.memory.close(gid,timestamp);closed.append(gid)
                fresh=self.model.init_state(1,obs.device,obs.dtype)
                for name,value in state.__dict__.items():
                    if torch.is_tensor(value) and name!='slot_ids':value[0,q]=getattr(fresh,name)[0,q]
        self.state=state.detach();self.last_timestamp=timestamp
        rejected=[]
        for o in range(len(tokens)):
            if o in global_by_obs:continue
            if not bool(valid[0,o]):reason='INVALID_OBSERVATION'
            elif float(torch.sigmoid(birth[0,o]))<self.birth_threshold:reason='NULL_AND_BIRTH_BELOW_THRESHOLD'
            else:reason='QUERY_CAPACITY'
            rejected.append({'observation':o,'reason':reason,'birth_probability':float(torch.sigmoid(birth[0,o]))})
        return RuntimeResult(timestamp,'TRAINED' if self.trained else 'SMOKE_UNTRAINED',
            {q:os[0] for q,os in mapping.items()},global_by_obs,birth_ids,{
            'scene_id':self.scene_id,'slot_ids_are_not_global_ids':True,'query_coordinates':'persistent_aligned_world_observation_xyz',
            'coordinates_calibrated':False,'observation_count':len(tokens),'valid_observation_count':int(valid.sum()),
            'valid_observation_ratio':float(valid.float().mean()) if valid.numel() else 0,
            'assignment_logits':distribution(logits[...,:-1]),'NULL_logits':distribution(logits[...,-1]),
            'assignment_minus_NULL':distribution(logits[...,:-1]-logits[...,-1:]),'birth_probability':distribution(torch.sigmoid(birth)),
            'existence_probability':distribution(output['state'].existence),'created_queries':len(birth_ids),'closed_ids':closed,
            'duplicate_rejections':assigned.duplicate_rejections,'rejections':rejected,'query_capacity':self.model.max_world_queries,
            'query_capacity_overflow':overflow,'checkpoint':self.checkpoint,'random_weights_forbidden_formal':not self.trained,
            'thresholds':{'null_margin':self.null_margin,'birth_threshold':self.birth_threshold,'pair_threshold':self.pair_threshold}},mapping)
