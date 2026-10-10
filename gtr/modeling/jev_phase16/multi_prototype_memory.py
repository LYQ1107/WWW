"""Bounded four-slot readers. No candidate ID, GT or network capacity changes."""
import copy
import torch
from torch.nn import functional as F
from .history_segment_encoder import SegmentHistory,diverse_four

VARIANTS=['A_original_four','B_recent_four','C_temporal_segments','D_camera_segments','E_bounded_combination']


class MultiPrototypeMemory:
    def __init__(self,variant):
        assert variant in VARIANTS[1:];self.variant=variant;self.identities={}

    def append(self,identity,feature,frame,view,row):
        identity=int(identity);key=(int(frame),int(view),int(row));x=self.identities.setdefault(identity,{})
        if self.variant in [VARIANTS[1],VARIANTS[4]]:
            values=x.setdefault('recent',[]);values.append(dict(sum=feature[:1024].detach().clone(),count=1,keys=[key],first=key,last=key))
            if len(values)>4:del values[0]
        if self.variant in [VARIANTS[2],VARIANTS[4]]:
            h=x.setdefault('temporal',SegmentHistory(16 if self.variant==VARIANTS[2] else 6));h.append(feature,key)
        if self.variant in [VARIANTS[3],VARIANTS[4]]:
            h=x.setdefault('camera',{}).setdefault(int(view),SegmentHistory(8 if self.variant==VARIANTS[3] else 3));h.append(feature,key)
        assert self.stored_vectors(identity)<=16

    def append_instance(self,inst,frame,view):
        for row,identity in enumerate(inst.track_ids.tolist() if inst.has('track_ids') else []):
            self.append(identity,inst.reid_features[row],frame,view,row)

    def stored_vectors(self,identity):
        x=self.identities[int(identity)]
        return len(x.get('recent',[]))+len(x['temporal'].items if 'temporal' in x else [])+sum(len(h.items) for h in x.get('camera',{}).values())

    def values(self,identity,view):
        x=self.identities[int(identity)]
        if self.variant==VARIANTS[1]:return list(reversed(x['recent']))
        if self.variant==VARIANTS[2]:return diverse_four(x['temporal'].items)
        cams=x['camera'];own=cams[int(view)].items if int(view) in cams else []
        other=cams[1-int(view)].items if 1-int(view) in cams else []
        if self.variant==VARIANTS[3]:return [own[-1] if own else None,own[0] if own else None,other[-1] if other else None,other[0] if other else None]
        return [x['recent'][-1],x['temporal'].items[0],own[-1] if own else None,other[-1] if other else None]

    def batch(self,x,refs,view):
        assert len(refs)==x['history_visual'].shape[1]
        result=dict(x);hist=torch.zeros_like(x['history_visual']);mask=torch.zeros_like(x['history_mask']);provenance=[]
        for col,identity in enumerate(refs):
            values=self.values(identity,view);provenance.append(values)
            for slot,item in enumerate(values):
                if item is not None:hist[0,col,slot]=F.normalize(item['sum'],dim=-1);mask[0,col,slot]=True
        result['history_visual']=hist;result['history_mask']=mask
        pair=x['pair_evidence'].clone();pair[...,:4]=torch.einsum('bqd,bkhd->bqkh',x['detection_visual'],hist)
        result['pair_evidence']=pair
        return result,provenance

    def state_dict(self):
        values={}
        for identity,x in self.identities.items():
            item={}
            if 'recent' in x:item['recent']=x['recent']
            if 'temporal' in x:item['temporal']=dict(capacity=x['temporal'].capacity,items=x['temporal'].items)
            if 'camera' in x:item['camera']={v:dict(capacity=h.capacity,items=h.items) for v,h in x['camera'].items()}
            values[identity]=item
        return copy.deepcopy(dict(version=1,variant=self.variant,identities=values))

    def load_state_dict(self,state):
        assert state['version']==1 and state['variant']==self.variant
        self.identities={}
        for identity,x in copy.deepcopy(state['identities']).items():
            item={}
            if 'recent' in x:item['recent']=x['recent']
            if 'temporal' in x:
                h=SegmentHistory(x['temporal']['capacity']);h.items=x['temporal']['items'];item['temporal']=h
            if 'camera' in x:
                item['camera']={}
                for view,value in x['camera'].items():
                    h=SegmentHistory(value['capacity']);h.items=value['items'];item['camera'][view]=h
            self.identities[identity]=item

    @classmethod
    def from_prefix(cls,variant,prefix):
        result=cls(variant);boundary=tuple(prefix['key'][1:]);instances=prefix['instances']
        first=max((len(instances[v]),v) for v in [0,1])[1]
        order=[first,1-first]+list(range(2,len(instances)))
        for index in order:
            frame,view=divmod(index,2);inst=instances[index]
            if (frame,view)>=boundary and not(frame==0 and inst.has('track_ids')):continue
            if inst.has('track_ids'):result.append_instance(inst,frame,view)
        return result
