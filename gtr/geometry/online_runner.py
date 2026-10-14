"""Causal GMT protocol: one synchronized group in, identities out, no future filtering.

Uses original initialization, transformer, GTA, assignment and memory methods.
This is evaluated against a separate causal GMT OFF baseline.
"""
import copy,numpy as np,torch
from detectron2.structures import Instances,Boxes
from ..modeling.meta_arch.gtr_rcnn import poss_ids,old_ids,old_reids
from ..modeling.meta_arch.custom_rcnn import CustomRCNN

def clone_instances(inst):
    result=Instances(inst.image_size)
    for name,value in inst.get_fields().items():
        if isinstance(value,torch.Tensor):result.set(name,value.clone())
        elif isinstance(value,Boxes):result.set(name,Boxes(value.tensor.clone()))
        else:result.set(name,copy.deepcopy(value))
    return result

class CausalGMT:
    def __init__(self,model):
        self.model=model;self.instances=[];self.id_count=0;self.counts={};self.reids={};self.frame=0;self.first_timestamp=None;self.last_timestamp=None;self.views=None
        if model.roi_heads.delay_cls:raise ValueError('causal protocol does not support full-video delayed classification')
        poss_ids.poss_ids=set();old_ids.old_ids=set();old_reids.old_reids=[];model.geo_runtime.reset()

    @torch.inference_mode()
    def step(self,records):
        views=len(records);timestamp=int(records[0]['frame_id'])
        if any(int(r['frame_id'])!=timestamp for r in records):raise ValueError('unsynchronized timestamps')
        if self.last_timestamp is not None and timestamp<=self.last_timestamp:raise ValueError('nonmonotonic stream')
        if self.views is not None and views!=self.views:raise ValueError('camera membership changed')
        self.views=views;self.last_timestamp=timestamp
        if self.first_timestamp is None:self.first_timestamp=timestamp
        current=[]
        for v,r in enumerate(records):current.extend(self.model.inference([r],views,[self.first_timestamp,timestamp,timestamp],v,do_postprocess=False))
        self.model.geo_runtime.observe(current,records,self.frame);self.instances.extend(current)
        if self.frame==0:
            order=np.argsort([len(x) for x in current]).tolist()[::-1];first=order.pop(0)
            initial=self.instances[first];initial.track_ids=torch.arange(1,len(initial)+1,device=initial.reid_features.device)
            self.id_count=len(initial)
            for i in range(1,self.id_count+1):self.counts[i]=1;self.reids[i]=initial[i-1]
            context=[initial]
            for v in sorted(order):
                context.append(Instances.cat([self.instances[v]]));k=len(context)-1
                asso,boxes,_,Np,_=self.model.get_asso(context,k=k)
                context,self.id_count,self.counts,self.reids=self.model.run_first_tracker_plus(context,[asso[0][:,:Np]],boxes[:Np],k,self.id_count,self.counts,self.reids)
                self.instances[v]=context[-1]
        else:
            start=max(0,self.frame+1-self.model.test_len)*views;end=self.frame*views;context=self.instances[start:end];older=self.instances[:start]
            for v in range(views):
                context=context+[self.instances[end+v]];k=len(context)-1
                asso,boxes,_,Np,_=self.model.get_asso(context,k=k)
                context,self.id_count,self.counts=self.model.run_global_tracker_plus(views,context,[asso[0][:,:Np]],boxes[:Np],k,self.id_count,self.counts,self.reids,older,v)
                self.instances[end+v]=context[-1]
        output=self.instances[-views:]
        # Postprocess must not scale mutable boxes that remain in tracker history.
        output=CustomRCNN._postprocess([clone_instances(i) for i in output],records,[(0,0)]*views,not_clamp_box=self.model.not_clamp_box)
        self.frame+=1
        return output
