"""Bounded, causal local-detection matching for optional identity-vector reuse."""
import torch
from torch.nn import functional as F


class TargetFeatureCache:
    def __init__(self,max_age=20,max_targets=128,min_cosine=.9,min_margin=.05,min_iou=.5,max_cameras=2):
        if max_age<1 or max_targets<1 or max_cameras<1:raise ValueError('positive cache bounds required')
        self.max_age=max_age;self.max_targets=max_targets;self.min_cosine=min_cosine;self.min_margin=min_margin;self.min_iou=min_iou
        self.max_cameras=max_cameras
        self.entries={};self.stats=dict(stored_targets=0,reused_targets=0,queried_targets=0,stale_camera_snapshots=0);self.peak_payload_bytes=0

    def clear(self):self.entries.clear()

    def payload_bytes(self):
        return sum(value.numel()*value.element_size() for entry in self.entries.values() for value in entry.values() if isinstance(value,torch.Tensor))

    def store(self,camera,frame,native,boxes,projected,valid,confidence):
        if int(camera) not in self.entries and len(self.entries)>=self.max_cameras:raise ValueError('camera cache allocation exceeds fixed bound')
        if int(camera) in self.entries and frame<self.entries[int(camera)]['frame']:raise ValueError('nonmonotonic camera feature update')
        legal=valid & torch.isfinite(native).all(1)&torch.isfinite(boxes).all(1)&torch.isfinite(projected).all(1)&torch.isfinite(confidence)&(native.norm(dim=1)>0)&(projected.norm(dim=1)>0)
        indices=legal.nonzero(as_tuple=True)[0]
        if not len(indices):self.entries.pop(int(camera),None);return
        order=confidence[indices].argsort(descending=True,stable=True)[:self.max_targets];indices=indices[order]
        self.entries[int(camera)]=dict(frame=int(frame),native=F.normalize(native[indices].detach().float(),dim=1).clone(),boxes=boxes[indices].detach().float().clone(),projected=projected[indices].detach().clone())
        self.stats['stored_targets']+=len(indices);self.peak_payload_bytes=max(self.peak_payload_bytes,self.payload_bytes())

    def reuse(self,camera,frame,native,boxes,output_dim):
        n=len(native);vectors=native.new_zeros((n,output_dim));valid=torch.zeros(n,dtype=torch.bool,device=native.device);sources=torch.full((n,),-1,dtype=torch.long,device=native.device)
        self.stats['queried_targets']+=n
        entry=self.entries.get(int(camera))
        if entry is None or not n:return vectors,valid,sources
        age=frame-entry['frame']
        if age<0:raise ValueError('future feature cache state cannot be consumed')
        if age>self.max_age:
            self.stats['stale_camera_snapshots']+=1;self.entries.pop(int(camera),None);return vectors,valid,sources
        if entry['projected'].shape[1]!=output_dim or entry['native'].shape[1]!=native.shape[1]:raise ValueError('cache feature dimensions changed')
        finite=torch.isfinite(native).all(1)&torch.isfinite(boxes).all(1)&(native.norm(dim=1)>0)
        query=F.normalize(torch.nan_to_num(native.float()),dim=1)
        cosine=query@entry['native'].T
        top=cosine.topk(min(2,len(entry['native'])),dim=1)
        candidates=top.indices[:,0]
        margin=(top.values[:,0]-top.values[:,1])/top.values[:,0].abs().clamp_min(1e-8) if top.values.shape[1]>1 else torch.ones(n,device=native.device)
        mutual=cosine.argmax(dim=0)[candidates]==torch.arange(n,device=native.device)
        reference=entry['boxes'][candidates]
        left=torch.maximum(boxes[:,:2],reference[:,:2]);right=torch.minimum(boxes[:,2:],reference[:,2:])
        intersection=(right-left).clamp_min(0).prod(1)
        area=(boxes[:,2:]-boxes[:,:2]).clamp_min(0).prod(1);ref_area=(reference[:,2:]-reference[:,:2]).clamp_min(0).prod(1)
        iou=intersection/(area+ref_area-intersection).clamp_min(1e-8)
        valid=finite&mutual&(top.values[:,0]>=self.min_cosine)&(margin>=self.min_margin)&(iou>=self.min_iou)
        vectors[valid]=entry['projected'][candidates[valid]];sources[valid]=entry['frame']
        self.stats['reused_targets']+=int(valid.sum())
        return vectors,valid,sources
