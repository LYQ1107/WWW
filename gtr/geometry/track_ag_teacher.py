"""Learned history-conditioned AG alone, with optional subsequent QD ablations."""
import torch
from torch import nn
from .track_query_encoder import TrackQueryEncoder
from .identity_guided_sampler import IdentityGuidedSampler
from .track_qd_teacher import TrackQDTeacher


class TrackRelationTeacher(nn.Module):
    def __init__(self,native_dim,channels,architecture='ag',ag_mode='C',qd_mode='track',k=24,summary='mean',matched_dino=False):
        super().__init__()
        if architecture not in ('ag','qd') or ag_mode not in ('roi','A','B','C'):raise ValueError('unknown prespecified teacher setting')
        self.architecture,self.ag_mode,self.k=architecture,ag_mode,k
        self.query=TrackQueryEncoder(native_dim,summary=summary)
        bottleneck=228 if matched_dino else None
        self.sampler=IdentityGuidedSampler(channels,projection_hidden=bottleneck)
        self.qd=TrackQDTeacher([channels]*4,mode=qd_mode,projection_hidden=bottleneck) if architecture=='qd' else None
        self.score=nn.Sequential(nn.Linear(264,128),nn.GELU(),nn.Linear(128,1)) if architecture=='ag' else None

    def forward(self,features,soft=False,history_permutation=None):
        n=len(features['histories']);device=features['native_scores'].device
        if not n:return features['native_scores'].new_zeros(1),dict(neutral=True,layer_weights=[],selected_tokens=[])
        # A zero-center full-image target gets a clearly recorded neutral native
        # fallback. It is never dropped from evaluation or treated as fake tokens.
        if self.ag_mode!='roi' and not bool(features['inside'].any()):
            return torch.cat((features['native_scores'],features['native_scores'].new_zeros(1))),dict(neutral=True,layer_weights=[],selected_tokens=[])
        values=[];layer_weights=[];selected_counts=[]
        for index in range(n):
            source=index if history_permutation is None else int(history_permutation[index])
            history=features['histories'][source]
            q=self.query(history['features'],history['times'],history['cameras'],features['time'],features['camera'],history['observed_length'])
            if self.ag_mode=='roi':
                selected=[value[None] for value in features['roi']];count=1
            else:
                tokens=features['layers'][2]
                score,_=self.sampler.scores(tokens,features['attention'],q,features['inside'],self.ag_mode)
                if soft and self.ag_mode=='C':
                    weights=score.softmax(0)
                    selected=[(value*weights[:,None]).sum(0,keepdim=True) for value in features['layers']];count=len(score)
                else:
                    _,indices,_=self.sampler(tokens,features['attention'],q,features['xy'],features['inside'],~features['inside'],self.k,self.ag_mode)
                    selected=[value[indices] for value in features['layers']];count=len(indices)
                    if self.ag_mode=='C':
                        # Hard indices are discrete; learned relevance keeps a
                        # differentiable selected-token weighting path after warmup.
                        weights=score[indices].softmax(0)*max(len(indices),1)
                        selected=[value*weights[:,None] for value in selected]
            selected_counts.append(count)
            if not count:raise ValueError('nonempty ROI branch unexpectedly has zero tokens')
            if self.qd is not None:
                value,weights=self.qd(q,selected,features['pair'][index]);layer_weights.append(None if weights is None else weights)
            else:
                embedding=self.sampler.key(selected[2].float()).mean(0)
                # Identical normalized projected inputs for pooled ROI and raw
                # token variants; raw VGGT magnitude cannot create a weak ROI control.
                embedding=torch.nn.functional.normalize(embedding,dim=0)
                projected_query=torch.nn.functional.normalize(self.sampler.query(q),dim=0)
                value=self.score(torch.cat((projected_query,embedding,features['pair'][index]))).squeeze(-1)
            values.append(value)
        # Dynamic track candidates plus explicit NULL, never a global ID table.
        logits=torch.cat((torch.stack(values),torch.zeros(1,device=device)))
        if not torch.isfinite(logits).all():raise ValueError('nonfinite teacher logits')
        return logits,dict(neutral=False,layer_weights=layer_weights,selected_tokens=selected_counts)


class NativeRelationMLP(nn.Module):
    def __init__(self,native_dim,capacity='qd'):
        super().__init__()
        if capacity not in ('ag','qd'):raise ValueError('unknown matched native MLP capacity')
        first,second=(176,64) if capacity=='ag' else (536,512)
        self.mlp=nn.Sequential(nn.Linear(3*native_dim+8,first),nn.GELU(),nn.Linear(first,second),nn.GELU(),nn.Linear(second,1))

    def forward(self,features,soft=False,history_permutation=None):
        values=[]
        for index,history in enumerate(features['histories']):
            source=index if history_permutation is None else int(history_permutation[index]);history=features['histories'][source]
            native=torch.nn.functional.normalize(history['features'].float(),dim=-1)
            current=torch.nn.functional.normalize(features['current'].float(),dim=-1)
            values.append(self.mlp(torch.cat((current,native[-1],native.mean(0),features['pair'][index]))).squeeze(-1))
        if not values:return features['native_scores'].new_zeros(1),dict(neutral=True,layer_weights=[],selected_tokens=[])
        return torch.cat((torch.stack(values),features['native_scores'].new_zeros(1))),dict(neutral=False,layer_weights=[],selected_tokens=[])
