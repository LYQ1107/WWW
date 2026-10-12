"""Matched small final fusers on one *frozen trained JEV* frontend.

These isolate use of auxiliary predictions and final-reader separation. They
are not pure controllers pretrained from scratch, nor a fourth major revision.
"""
import torch
from torch import nn
from gtr.modeling.jev_stage2.model import Reader
from gtr.modeling.jev_oneweek.plans import selected_tokens


class DiagnosticFusion(nn.Module):
    def __init__(self, backbone, kind):
        super().__init__(); self.backbone = backbone; self.kind = kind
        for p in backbone.parameters(): p.requires_grad_(False)
        self.pool = nn.Sequential(nn.Linear(128,32),nn.GELU(),nn.Linear(32,64))
        self.edge_value = nn.Linear(64,1); self.plan_value = nn.Linear(64,1)
        for layer in (self.edge_value,self.plan_value):
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
        if kind == 'explicit_scores':
            self.gates = nn.Sequential(nn.Linear(4,504),nn.GELU(),nn.Linear(504,128))
            self.project = nn.Linear(128,64)
        elif kind == 'independent_readers':
            self.project = nn.ModuleList([nn.Linear(64,64),nn.Linear(64,64)])
            self.scalar = nn.ModuleList([nn.Linear(4,64),nn.Linear(4,64)])
            self.readers = nn.ModuleList([Reader(64),Reader(64)])
        elif kind == 'mlp':
            self.project = nn.Sequential(nn.Linear(132,386),nn.GELU(),nn.Linear(386,64))
        elif kind == 'attention':
            self.project = nn.Linear(132,64)
            self.readers = nn.ModuleList([Reader(64),Reader(64)])
        else: raise ValueError(kind)

    def train(self, mode=True):
        super().train(mode); self.backbone.eval(); return self

    def details(self, x):
        with torch.no_grad(): old = self.backbone.details(x)
        if not old['plan_scores']: return old
        cross, stable = old['cross_features'], old['stability_features']
        b,q,kp,width = cross.shape
        probabilities = [torch.cat([old[name].sigmoid(),cross.new_zeros(b,q,1)],-1)
                         for name in ('cross_logits','stability_logits')]
        margin = old['edge_logits']-old['edge_logits'][:,:,-1:]
        affinity = torch.einsum('bqd,bkd->bqk',x['detection_visual'],x['anchor_visual'])
        affinity *= x['anchor_mask'][:,None]
        affinity = torch.cat([affinity,cross.new_zeros(b,q,1)],-1)
        scalars = torch.stack([*probabilities,margin,affinity],-1)
        both = torch.cat([cross,stable],-1)
        if self.kind == 'explicit_scores':
            feature = self.project(both*self.gates(scalars).sigmoid())
        elif self.kind == 'independent_readers':
            keys = torch.stack([cross,stable],-2).reshape(b*q*kp,2,width)
            streams = [reader((project(view)+scalar(scalars)).reshape(b*q*kp,1,width),keys)
                       for reader,project,scalar,view in zip(self.readers,self.project,self.scalar,[cross,stable])]
            feature = ((streams[0]+streams[1])/2).reshape(b,q,kp,width)
        else:
            feature = self.project(torch.cat([both,scalars],-1))
            if self.kind == 'attention':
                keys = torch.stack([cross,stable],-2).reshape(b*q*kp,2,width)
                feature = feature.reshape(b*q*kp,1,width)
                for reader in self.readers: feature = reader(feature,keys)
                feature = feature.reshape(b,q,kp,width)
        d = dict(old,features=feature,edge_logits=old['edge_logits']+self.edge_value(feature).squeeze(-1))
        values = []
        for i, pp in enumerate(old['plans']):
            qn,kn = int(x['question_mask'][i].sum()),int(x['identity_mask'][i].sum())
            edge = torch.cat([d['edge_logits'][i,:qn,:kn]-old['edge_logits'][i,:qn,:kn],
                              d['edge_logits'][i,:qn,-1:]-old['edge_logits'][i,:qn,-1:]],-1)
            token = torch.cat([feature[i,:qn,:kn],feature[i,:qn,-1:]],1)
            selected,index = selected_tokens(token,pp)
            additive = edge.double()[torch.arange(qn,device=edge.device)[None],index].sum(-1)
            joint = self.plan_value(self.pool(torch.cat([selected.mean(1),selected.max(1).values],-1))).squeeze(-1)
            values.append(old['plan_scores'][i]+additive+joint.double())
        d['plan_scores'] = values
        return d

    def forward(self,x):
        d = self.details(x); self.runtime_details = d
        self.runtime_choice = list(d['plans'][0][int(d['plan_scores'][0].argmax())]) if d['plan_scores'] else []
        return d['edge_logits']
