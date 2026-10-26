"""Dedicated historical membership; no birth, lifecycle, gate or write heads."""
import math
import torch
from torch import nn
from torch.nn import functional as F

MODEL_SCHEMA="worldobject_membership_v5_v1"


class WorldObjectMembershipHead(nn.Module):
    def __init__(self,visual_dim=1024,hidden_dim=256,geometry_dim=32,pair_chunk=4096,dropout=.05):
        super().__init__()
        self.visual_dim=int(visual_dim);self.hidden_dim=int(hidden_dim);self.geometry_dim=int(geometry_dim)
        self.pair_chunk=int(pair_chunk);self.dropout=float(dropout)
        self.roi_projection=nn.Sequential(nn.Linear(visual_dim,hidden_dim),nn.LayerNorm(hidden_dim),nn.GELU())
        self.pair_encoder=nn.Sequential(nn.Linear(3*hidden_dim+geometry_dim+14,hidden_dim),
            nn.LayerNorm(hidden_dim),nn.GELU(),nn.Linear(hidden_dim,hidden_dim),nn.GELU())
        self.membership=nn.Linear(hidden_dim,1)
        self.no_observation=nn.Sequential(nn.Linear(hidden_dim+8,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,2))
        self.log_appearance_scale=nn.Parameter(torch.tensor(math.log(10.)))
        nn.init.zeros_(self.membership.weight);nn.init.zeros_(self.membership.bias)
        nn.init.zeros_(self.no_observation[-1].weight);nn.init.zeros_(self.no_observation[-1].bias)

    def configuration(self):
        return {"schema":MODEL_SCHEMA,"visual_dim":self.visual_dim,"hidden_dim":self.hidden_dim,
                "geometry_dim":self.geometry_dim,"pair_chunk":self.pair_chunk,"dropout":self.dropout,
                "prediction_heads":["membership","per_camera_no_observation"]}

    def forward(self,history_features,observation_features,pair_geometry,observation_context,history_context,camera_slots):
        m,n=history_features.shape[0],observation_features.shape[0]
        if history_features.shape!=(m,2,self.visual_dim) or observation_features.shape!=(n,self.visual_dim):
            raise ValueError("V5_VISUAL_INPUT_SHAPE")
        if pair_geometry.shape!=(m,n,self.geometry_dim) or observation_context.shape!=(n,6) or history_context.shape!=(m,8):
            raise ValueError("V5_CONTEXT_INPUT_SHAPE")
        if camera_slots.shape!=(n,) or (n and not bool(((camera_slots>=0)&(camera_slots<2)).all())):
            raise ValueError("V5_TWO_CAMERA_SLOTS_REQUIRED")
        q=self.roi_projection(history_features);d=self.roi_projection(observation_features)
        q=F.dropout(q,p=self.dropout,training=self.training);d=F.dropout(d,p=self.dropout,training=self.training)
        null=self.no_observation(torch.cat([q.mean(1),history_context],-1))
        raw=torch.stack([history_features[:,v]@observation_features.T for v in range(2)],dim=1)
        cosine=raw[:,camera_slots,torch.arange(n,device=raw.device)] if n else raw.new_empty((m,0))
        parts=[]
        for begin in range(0,m*n,self.pair_chunk):
            idx=torch.arange(begin,min(m*n,begin+self.pair_chunk),device=d.device);i=idx//n;j=idx%n
            a=q[i,camera_slots[j]];b=d[j]
            x=torch.cat([a,b,torch.abs(a-b),pair_geometry[i,j],history_context[i],observation_context[j]],-1)
            parts.append(self.membership(self.pair_encoder(x)).squeeze(-1))
        residual=torch.cat(parts).reshape(m,n) if parts else (q.sum()+d.sum())*0+d.new_zeros((m,n))
        score=residual+self.log_appearance_scale.clamp(0,4).exp()*(cosine-.7)
        return {"membership_logits":score,"no_observation_logits":null,
                "projected_observations":F.normalize(d,dim=-1),"projected_history":F.normalize(q,dim=-1)}


def membership_inputs(history,observations,context,config,device="cpu"):
    import numpy as np
    from .world_object_v5_geometry import historical_geometry
    m,n=len(history),len(observations);d=config.feature_dim
    q=np.zeros((m,2,d),np.float32);o=np.zeros((n,d),np.float32)
    hc=np.zeros((m,8),np.float32);oc=np.zeros((n,6),np.float32)
    for i,x in enumerate(history):
        for camera in (1,2):
            feature=x.view_identity_features.get(camera,x.identity_embedding)
            if feature is not None:q[i,camera-1]=feature
        hc[i]=[min(x.age,1000)/1000,min(x.missed,100)/100,x.identity_confidence,
            float(x.status=="CONFIRMED"),min(x.hits,100)/100,float(1 in x.camera_history),
            float(2 in x.camera_history),float(x.status=="TENTATIVE")]
    for j,x in enumerate(observations):
        o[j]=x.roi_feature;oc[j]=[x.score,float(x.visible),float(x.camera_id==1),float(x.camera_id==2),
                                float(x.score>=config.birth_high_score),float(x.feature_valid)]
    g=historical_geometry(history,observations,context,config)
    arrays=(q,o,g,oc,hc,np.asarray([x.camera_id-1 for x in observations],np.int64))
    if any(not np.isfinite(a).all() for a in arrays):raise ValueError("NONFINITE_MEMBERSHIP_INPUTS")
    return tuple(torch.as_tensor(a,device=device) for a in arrays)
