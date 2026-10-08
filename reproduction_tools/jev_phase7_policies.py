"""Same-state, legal three-action policies for historical B2 attribution."""
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from gtr.modeling.jev_baselines import _prepare_inputs, _result
from gtr.modeling.jev_decision import ACTION_TO_INDEX
from jev_phase7_common import OUT, B2

A, R, N = 'ACCEPT_CURRENT', 'REASSOCIATE', 'START_NEW'

class StrongMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(64,153),nn.GELU(),nn.Linear(153,153),nn.GELU(),nn.Linear(153,3))
    def forward(self, features, questions, legal_actions):
        state,q,ids,mask = _prepare_inputs(features,questions,legal_actions)
        assert bool((q==0).all()) and bool(((ids<3)|~mask).all())
        logits = self.network(state).gather(1,ids.clamp_min(0))
        return _result(logits,ids,mask)

class DynamicThreshold(nn.Module):
    def __init__(self):
        super().__init__();self.base_threshold=nn.Parameter(torch.tensor(.1))
        self.conditioner=nn.Sequential(nn.Linear(64,154),nn.GELU(),nn.Linear(154,154),nn.GELU(),nn.Linear(154,1))
    def threshold_for(self,x):
        return self.base_threshold+self.conditioner(x.reshape(-1,64)).squeeze(-1)
    def forward(self,features,questions,legal_actions):
        state,q,ids,mask=_prepare_inputs(features,questions,legal_actions)
        assert bool((q==0).all()) and bool(((ids<3)|~mask).all())
        tau=self.threshold_for(state);current=state[:,56]-tau;alternate=state[:,1]-tau
        logs=torch.stack((F.logsigmoid(current),F.logsigmoid(-current)+F.logsigmoid(alternate),
                          F.logsigmoid(-current)+F.logsigmoid(-alternate)),dim=1)
        # Binary second round uses the same threshold, without alternate mass.
        binary=~((ids==1)&mask).any(dim=1)
        binary_logs=torch.stack((F.logsigmoid(current),torch.zeros_like(current),F.logsigmoid(-current)),dim=1)
        logs=torch.where(binary[:,None],binary_logs,logs)
        return _result(logs.gather(1,ids.clamp_min(0)),ids,mask)

class MatchActor:
    """No tracker/GT/future access. Second-round binary call uses this interface."""
    def __init__(self,name):
        self.name=name;self.model=None;self.fast=None;self.temperature=1.
        if name in ('B2','BINARY'):
            from jev_phase6_common import load_controller
            path=B2 if name=='B2' else __import__('pathlib').Path('/home/liuyeqiang/WWW_jev_phase6_runtime/20261008/gating_training/G3/model_calibrated.pth')
            self.model=load_controller(path,'cpu')
            from jev_phase6_fast_match import FastMatchPolicy
            self.fast=FastMatchPolicy(self.model)
        elif name in ('MLP','DYNAMIC'):
            payload=torch.load(OUT/'training'/name/'model_calibrated.pth',map_location='cpu')
            self.model=StrongMLP() if name=='MLP' else DynamicThreshold()
            self.model.load_state_dict(payload['model'],strict=True);self.model.eval()
            self.temperature=float(payload['temperature'])

    @torch.no_grad()
    def probabilities(self,feature,legal):
        if legal==[N]:return {N:1.}
        x=feature.detach().cpu().float().reshape(1,64)
        if self.name=='B2':p=torch.softmax(self.fast.logits(x,legal),-1)[0]
        elif self.name=='BINARY':
            binary=[A,N];p0=torch.softmax(self.fast.logits(x,binary),-1)[0]
            alt=torch.sigmoid((x[0,1]-.1)*10)
            mass={A:p0[0],R:p0[1]*alt,N:p0[1]*(1-alt)} if R in legal else {A:p0[0],N:p0[1]}
            p=torch.stack([mass[a]for a in legal])
        elif self.name in ('MLP','DYNAMIC'):
            out=self.model(x,[0],[legal]);p=torch.softmax(out['logits']/self.temperature,-1)[0]
        else:
            tau=.1;current=x[0,56]-tau;alternate=x[0,1]-tau
            if self.name=='GMT':current=x[0,59]
            pa=torch.sigmoid(current);pr=torch.sigmoid(alternate)
            mass={A:pa,R:(1-pa)*pr,N:(1-pa)*(1-pr)} if R in legal else {A:pa,N:1-pa}
            p=torch.stack([mass[a]for a in legal])
        assert torch.isfinite(p).all() and bool((p>=0).all()) and abs(float(p.sum())-1)<1e-5
        return dict(zip(legal,map(float,p)))

    @torch.no_grad()
    def decide(self,feature,question,legal,*,off_action=None,context=None):
        assert question=='MATCH_DECISION'
        p=self.probabilities(feature,legal)
        if legal==[N]:action=N
        elif self.name=='GMT':action=off_action
        elif self.name in ('FIXED','DYNAMIC'):
            x=feature.detach().cpu().float().reshape(1,64)
            tau=.1 if self.name=='FIXED' else float(self.model.threshold_for(x)[0])
            action=A if float(x[0,56])>tau else R if R in legal and float(x[0,1])>tau else N
        elif self.name=='BINARY':
            binary=self.fast.action(feature,[A,N])
            action=A if binary==A else R if R in legal and float(feature[1])>.1 else N
        else:action=max(p,key=p.get)
        assert action in legal
        return SimpleNamespace(committed_action=action,probabilities=p,state_features=feature.detach().cpu().tolist())
