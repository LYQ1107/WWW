"""GT-free chronological segment sums, with one stored vector per segment."""
import copy
import torch
from torch.nn import functional as F


class SegmentHistory:
    def __init__(self,capacity,chunk=8,gap=8,similarity=.75):
        self.capacity=capacity;self.chunk=chunk;self.gap=gap;self.similarity=similarity;self.items=[]

    def append(self,feature,key):
        feature=feature[:1024].detach();key=tuple(map(int,key))
        old=self.items[-1] if self.items else None
        split=old is None or old['count']>=self.chunk or key[0]-old['last'][0]>self.gap
        if not split:split=float(F.normalize(feature,dim=-1)@F.normalize(old['sum'],dim=-1))<self.similarity
        if split:
            self.items.append(dict(sum=feature.clone(),count=1,first=key,last=key,keys=[key]))
            if len(self.items)>self.capacity:del self.items[1]  # Preserve earliest anchor and newest segments.
        else:
            old['sum']=old['sum']+feature;old['count']+=1;old['last']=key;old['keys'].append(key)

    def state_dict(self):return copy.deepcopy(self.items)


def diverse_four(items):
    if not items:return []
    chosen=[len(items)-1]
    if len(items)>1:chosen.append(0)
    values=F.normalize(torch.stack([s['sum'] for s in items]),dim=-1)
    while len(chosen)<min(4,len(items)):
        distance=(values@values[chosen].T).max(-1).values
        distance[chosen]=2
        chosen.append(int(distance.argmin()))
    return [items[i] for i in chosen]
