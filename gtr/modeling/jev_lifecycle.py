"""Small typed identity adapters and canonical, strictly online metadata."""
import math
import torch
from torch import nn

from .jev_decision import JEVDecisionController, QUESTION_NAMES

MEMORY_FIELDS=('detector_confidence','reid_consistency','prototype_distance',
               'log_track_maturity','views_fraction','recent_feature_variance',
               'log_track_age','quality_rank')
REACT_FIELDS=('log_time_since_last_seen','log_stale_age','log_memory_count','memory_quality',
              'log_candidate_count','top1_score','top2_score','top1_top2_margin','candidate_entropy',
              'cross_camera_consistency','log_track_lifetime','last_reliable_quality','assigned_candidate_score')


def memory_features(payload,row,state,metadata):
    track=int(metadata['track_id']); m=metadata.get('identity',{})
    feature=torch.as_tensor(payload['reid_features'][row]).float().cpu()
    samples=state.memory.get(track,[])[-10:]
    consistency=variance=0.0
    if samples:
        history=torch.stack([torch.as_tensor(x).float().cpu() for x in samples])
        proto=history.mean(0)
        consistency=float(torch.nn.functional.cosine_similarity(feature[None],proto[None]).item())
        norm=torch.nn.functional.normalize(history,dim=-1)
        variance=float(norm.var(0,unbiased=False).mean().item())
    confidence=float(payload['detection_scores'][row]); qualities=m.get('recent_confidences',[])
    rank=sum(q<=confidence for q in qualities)/max(1,len(qualities))
    age=max(0,int(payload['frame'])-int(m.get('first_seen',payload['frame'])))
    return torch.tensor([confidence,consistency,1-consistency,math.log1p(state.track_hits.get(track,0)),
                         len(m.get('views',[]))/2,variance,math.log1p(age),rank],dtype=torch.float32)


def reactivation_features(frame,view,state,identity,candidate_ids,scores,assigned_id):
    values=torch.as_tensor(scores).float().cpu()
    order=values.sort(descending=True).values
    top1=float(order[0]) if len(order) else 0.0; top2=float(order[1]) if len(order)>1 else 0.0
    probs=values.softmax(0) if len(values) else values
    entropy=float(-(probs*probs.clamp_min(1e-8).log()).sum()) if len(values) else 0.0
    gap=max(0,int(frame)-int(identity.get('last_seen',frame)))
    stale=max(0,int(frame)-int(identity.get('stale_since',frame)))
    qualities=identity.get('memory_quality',[])
    quality=sum(qualities)/len(qualities) if qualities else 0.0
    lifetime=max(0,int(identity.get('last_seen',frame))-int(identity.get('first_seen',frame)))
    score=float(values[list(candidate_ids).index(assigned_id)])
    return torch.tensor([math.log1p(gap),math.log1p(stale),math.log1p(len(state.memory.get(assigned_id,[]))),
        quality,math.log1p(len(candidate_ids)),top1,top2,top1-top2,entropy,
        float(view in identity.get('views',[])),math.log1p(lifetime),float(identity.get('last_confidence',0)),score])


class TypedJEVController(nn.Module):
    """39,712 parameters; existing JEV core and semantic legal-action scorer."""
    def __init__(self):
        super().__init__()
        self.match_adapter=nn.Linear(64,64)
        nn.init.zeros_(self.match_adapter.weight);nn.init.zeros_(self.match_adapter.bias)
        self.memory_adapter=nn.Linear(len(MEMORY_FIELDS),64)
        self.reactivation_adapter=nn.Linear(len(REACT_FIELDS),64)
        self.core=JEVDecisionController(64,hidden_dim=128,question_dim=32,action_dim=32)

    def forward(self,state_features,questions,legal_actions):
        state=state_features.float()
        if state.ndim==1:state=state[None]
        q=self.core._questions_tensor(questions,len(state),state.device)
        adapted=torch.empty_like(state)
        for index in range(3):
            rows=q==index
            if not bool(rows.any()):continue
            if index==0:adapted[rows]=state[rows]+self.match_adapter(state[rows])
            elif index==1:adapted[rows]=self.memory_adapter(state[rows,:len(MEMORY_FIELDS)])
            else:adapted[rows]=self.reactivation_adapter(state[rows,:len(REACT_FIELDS)])
        return self.core(adapted,q,legal_actions)


def pack_typed_state(values):
    state=torch.zeros(64,dtype=torch.float32)
    if len(values)>64:raise ValueError('typed state exceeds shared adapter width')
    state[:len(values)]=torch.as_tensor(values)
    return state
