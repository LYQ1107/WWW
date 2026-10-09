"""Shared state, dynamic multi-token question and action readers; no GTA inputs."""
import torch
from torch import nn

class Reader(nn.Module):
    def __init__(self,d):
        super().__init__();self.attn=nn.MultiheadAttention(d,8,batch_first=True,dropout=0);self.norm1=nn.LayerNorm(d);self.ff=nn.Sequential(nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d));self.norm2=nn.LayerNorm(d)
    def forward(self,q,k,mask=None):
        if k.shape[1]==0:return self.norm2(q+self.ff(q))
        if mask is not None:
            empty=~mask.any(1);k=k.clone();mask=mask.clone();k[empty,0]=0;mask[empty,0]=True
        x=self.norm1(q+self.attn(q,k,k,key_padding_mask=~mask if mask is not None else None,need_weights=False)[0]);return self.norm2(x+self.ff(x))

class GlobalIdentityJev(nn.Module):
    def __init__(self,variant='full',d=256):
        super().__init__();self.variant=variant;self.d=d
        self.visual=nn.Linear(1024,d);self.detection=nn.Linear(8,d);self.identity=nn.Linear(12,d);self.evidence=nn.Linear(16,d);self.task=nn.Embedding(3,d);self.history_kind=nn.Embedding(4,d)
        self.state_slots=nn.Parameter(torch.randn(1,8,d)*.02);self.state_read=Reader(d);self.state_layers=nn.ModuleList([Reader(d),Reader(d)])
        baseline=variant in ['motip','camel','set_transformer']
        self.question_read=Reader(d) if not baseline else None;self.question_evidence=Reader(d) if not baseline else None;self.option_question=Reader(d) if not baseline else None;self.option_state=Reader(d) if not baseline else None
        self.competition=nn.Linear(2*d,d);self.gate=nn.Linear(d,d) if not baseline else None;self.option_projection=nn.Linear(d,d) if not baseline else None;self.norm=nn.LayerNorm(d);self.choice=nn.ModuleList([nn.Linear(d,1),nn.Linear(d,1)]) if variant!='no_typed_head' else None;self.untyped=nn.Linear(d,1) if variant=='no_typed_head' else None;self.memory_head=nn.Linear(d,1);self.terminal=nn.Embedding(2,d);self.decoder_layers=nn.ModuleList([Reader(d) for _ in range(4)]) if baseline else nn.ModuleList()
    def encode_state(self,x):
        b,q,_=x['detection_visual'].shape;k=x['history_visual'].shape[1];device=x['detection_visual'].device
        h=self.visual(x['history_visual'])+self.history_kind.weight[None,None]+self.identity(x['identity_meta'])[:,:,None]
        h=h*x['history_mask'][...,None];det=self.visual(x['detection_visual'])+self.detection(x['detection_meta'])
        tokens=torch.cat([det,h.reshape(b,k*4,self.d)],1);mask=torch.cat([x['question_mask'],(x['history_mask']&x['identity_mask'][:,:,None]).reshape(b,k*4)],1)
        state=self.state_read(self.state_slots.expand(b,-1,-1),tokens,mask)
        for layer in self.state_layers:state=layer(state,state)
        return state,h,det
    def encode_questions(self,x,state,h,det):
        b,q,d=det.shape;k=h.shape[1];typ=x['question_type'];query=det+self.task(typ)
        if k:
            weights=x['identity_mask'][...,None].float();bank=(h.sum(2)*weights).sum(1)/weights.sum(1).clamp_min(1)
            bank0=(h[:,:,2]*weights).sum(1)/weights.sum(1).clamp_min(1);bank1=(h[:,:,3]*weights).sum(1)/weights.sum(1).clamp_min(1)
            evidence=self.evidence(x['pair_evidence']);avg=(evidence*x['legal'][...,None]).sum(2)/x['legal'].sum(2).clamp_min(1)[...,None]
        else:bank=det.new_zeros(b,d);bank0=bank;bank1=bank;avg=det.new_zeros(b,q,d)
        tokens=torch.stack([query,self.detection(x['detection_meta']),bank[:,None].expand(-1,q,-1),bank0[:,None].expand(-1,q,-1),bank1[:,None].expand(-1,q,-1),avg],2)
        if self.variant=='fixed_question':tokens=self.task(typ)[:,:,None].expand(-1,-1,6,-1)
        if self.variant in ['no_question_reader','motip','camel','set_transformer']:return tokens,query
        flat=tokens.reshape(b*q,6,d);flat=self.question_evidence(flat,flat)
        flat=self.question_read(flat,state[:,None].expand(-1,q,-1,-1).reshape(b*q,8,d))
        return flat.reshape(b,q,6,d),flat.mean(1).reshape(b,q,d)
    def score_questions(self,x,state,h,det,question_tokens,question):
        b,q,d=det.shape;k=h.shape[1];option=h.sum(2)/x['history_mask'].sum(2).clamp_min(1)[...,None]
        option=option[:,None].expand(-1,q,-1,-1)+self.evidence(x['pair_evidence'])+det[:,:,None]
        terminal=self.terminal(x['question_type'].clamp_max(1))+det;option=torch.cat([option,terminal[:,:,None]],2)
        if self.variant=='motip':
            for layer in self.decoder_layers[:3]:option=layer(option.reshape(b*q,k+1,d),state[:,None].expand(-1,q,-1,-1).reshape(b*q,8,d)).reshape(b,q,k+1,d)
            option=self.decoder_layers[3](option.reshape(b*q,k+1,d),question_tokens.reshape(b*q,6,d)).reshape(b,q,k+1,d)
        elif self.variant=='camel':
            for layer in self.decoder_layers[:2]:option=layer(option.reshape(b*q,k+1,d),option.reshape(b*q,k+1,d),torch.cat([x['legal'],torch.ones(b,q,1,dtype=torch.bool,device=det.device)],2).reshape(b*q,k+1)).reshape(b,q,k+1,d)
            for layer in self.decoder_layers[2:]:option=layer(option.reshape(b*q,k+1,d),state[:,None].expand(-1,q,-1,-1).reshape(b*q,8,d)).reshape(b,q,k+1,d)
        elif self.variant=='set_transformer':
            for layer in self.decoder_layers:option=layer(option.reshape(b*q,k+1,d),option.reshape(b*q,k+1,d),torch.cat([x['legal'],torch.ones(b,q,1,dtype=torch.bool,device=det.device)],2).reshape(b*q,k+1)).reshape(b,q,k+1,d)
        elif self.variant!='no_option_reader':
            option=self.option_question(option.reshape(b*q,k+1,d),question_tokens.reshape(b*q,6,d)).reshape(b,q,k+1,d)
            option=self.option_state(option.reshape(b*q,k+1,d),state[:,None].expand(-1,q,-1,-1).reshape(b*q,8,d)).reshape(b,q,k+1,d)
        legal=torch.cat([x['legal'],torch.ones(b,q,1,dtype=torch.bool,device=det.device)],2)
        if self.variant!='no_competition':
            mean=(option*legal[...,None]).sum(2)/legal.sum(2).clamp_min(1)[...,None];maximum=option.masked_fill(~legal[...,None],-1e4).max(2).values;option=option+self.competition(torch.cat([mean,maximum],-1))[:,:,None]
        if self.variant not in ['no_gating','set_transformer','motip','camel']:
            option=self.norm(option+self.option_projection(option)*torch.tanh(self.gate(question))[:,:,None])
        else:option=self.norm(option)
        if self.variant=='no_typed_head':logits=self.untyped(option).squeeze(-1)
        else:
            heads=torch.stack([layer(option).squeeze(-1) for layer in self.choice],-1);logits=heads.gather(-1,x['question_type'][:,:,None,None].expand(-1,-1,k+1,1).clamp_max(1)).squeeze(-1)
        return logits.masked_fill(~legal,-1e4)
    def forward(self,x):
        state,h,det=self.encode_state(x)
        if self.variant=='no_shared_state':state=state*0
        qt,question=self.encode_questions(x,state,h,det);return self.score_questions(x,state,h,det,qt,question)
    def memory_probability(self,x):
        state,h,det=self.encode_state(x);_,question=self.encode_questions(x,state,h,det);return torch.sigmoid(self.memory_head(question).squeeze(-1))
