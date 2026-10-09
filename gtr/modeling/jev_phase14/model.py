"""WHO, availability and observed-history trust share causal visual evidence."""
import torch
from torch import nn
from torch.nn import functional as F
from gtr.modeling.jev_stage2.model import GlobalIdentityJev

class SharedMLP(nn.Module):
    """Capacity-matched ordinary pooling/MLP control without attention readers."""
    def __init__(self, d=256):
        super().__init__(); self.d=d
        self.visual=nn.Linear(1024,d); self.det=nn.Linear(8,d); self.identity=nn.Linear(12,d)
        self.evidence=nn.Linear(16,d); self.kind=nn.Embedding(4,d); self.task=nn.Embedding(3,d)
        self.shared=nn.Sequential(nn.Linear(d,768),nn.GELU(),nn.Linear(768,d))
        self.blocks=nn.ModuleList([nn.Sequential(nn.Linear(d,1536),nn.GELU(),nn.Linear(1536,d),nn.LayerNorm(d)) for _ in range(4)])
        self.option=nn.Sequential(nn.Linear(3*d,512),nn.GELU(),nn.Linear(512,d),nn.LayerNorm(d))
        self.new=nn.Parameter(torch.randn(1,1,1,d)*.02); self.choice=nn.Linear(d,1)
    def features(self,x):
        det=self.visual(x['detection_visual'])+self.det(x['detection_meta'])
        hist=self.visual(x['history_visual'])+self.kind.weight[None,None]+self.identity(x['identity_meta'])[:,:,None]
        mask=x['history_mask'] & x['identity_mask'][:,:,None]
        hist=hist*mask[...,None]; bank=hist.sum(2)/mask.sum(2).clamp_min(1)[...,None]
        tokens=torch.cat([det,bank],1); tm=torch.cat([x['question_mask'],x['identity_mask']],1)
        state=(tokens*tm[...,None]).sum(1)/tm.sum(1).clamp_min(1)[:,None]
        state=self.shared(state)
        for block in self.blocks:state=state+block(state)
        b,q,d=det.shape;k=bank.shape[1]
        identity=bank[:,None].expand(-1,q,-1,-1)+self.evidence(x['pair_evidence'])
        identity=torch.cat([identity,self.new.expand(b,q,1,-1)],2)
        options=self.option(torch.cat([identity,det[:,:,None].expand(-1,-1,k+1,-1),state[:,None,None].expand(-1,q,k+1,-1)],-1))
        queries=[det+state[:,None]+self.task.weight[t] for t in range(3)]
        return self.choice(options).squeeze(-1),options,queries

class ReliableIdentityPolicy(nn.Module):
    def __init__(self,variant='multi_question',objective='availability_joint'):
        super().__init__();self.variant=variant;self.objective=objective
        base='full' if variant=='multi_question' else variant
        self.core=SharedMLP() if base=='shared_mlp' else GlobalIdentityJev(base)
        self.availability_head=nn.Linear(256,1);self.trust_head=nn.Linear(256,1)
    def details(self,x):
        expected={'detection_visual','detection_meta','history_visual','history_mask','identity_meta','pair_evidence','legal','question_mask','identity_mask','question_type'}
        if set(x)!=expected:raise ValueError('Only causal tensors may enter identity policy; refs and GT labels stay outside')
        if isinstance(self.core,SharedMLP):
            who,option,queries=self.core.features(x)
        else:
            state,h,det=self.core.encode_state(x)
            qt,who_query=self.core.encode_questions(x,state,h,det)
            # Capture the exact shared OptionReader representation without changing
            # the historical module or adding another transformer. Hooks are removed
            # even on errors; this wrapper is single-threaded per experiment actor.
            feature=[];hook=self.core.norm.register_forward_hook(lambda m,a,o:feature.append(o))
            try:who=self.core.score_questions(x,state,h,det,qt,who_query)
            finally:hook.remove()
            assert len(feature)==1;option=feature[0]
            if self.variant in ['multi_question','fixed_question','set_transformer','motip']:
                queries=[who_query]
                for t in [1,2]:
                    typed=dict(x);typed['question_type']=torch.full_like(x['question_type'],t)
                    _,query=self.core.encode_questions(typed,state,h,det)
                    # Ordinary controls receive all shared evidence as well.
                    if self.variant in ['set_transformer','motip']:query=query+state.mean(1)[:,None]
                    queries.append(query)
            else:queries=[who_query,who_query,who_query]
        availability=self.availability_head(queries[1]).squeeze(-1)
        trust=self.trust_head(option[:,:,:-1]+queries[2][:,:,None]).squeeze(-1)
        z=who
        if self.objective=='availability_joint':
            existing=who[:,:,:-1]+F.logsigmoid(availability)[:,:,None]+F.logsigmoid(trust)
            terminal=who[:,:,-1]+F.logsigmoid(-availability)
            z=torch.cat([existing,terminal[:,:,None]],-1)
        legal=torch.cat([x['legal'],torch.ones_like(x['question_mask'])[:,:,None]],-1)
        return dict(logits=z.masked_fill(~legal,-1e4),choice_logits=who,availability_logits=availability,trust_logits=trust)
    def forward(self,x):return self.details(x)['logits']
