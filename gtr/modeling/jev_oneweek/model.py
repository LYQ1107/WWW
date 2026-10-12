"""Frozen XIV initialization + compact equal-evidence association controllers.

Typed cross-camera compatibility and observed Global-ID continuity are distinct
supervised questions. A joint plan reader is explicitly non-additive. This is
an incremental comparison on a shared pretrained initializer, not a claim that
ordinary controls were pretrained independently for equal total lifetime cost.
"""
import torch
from torch import nn
from gtr.modeling.jev_phase14.model import ReliableIdentityPolicy
from gtr.modeling.jev_stage2.model import Reader
from .plans import plans,selected_tokens

ARMS=['jev','fixed_question','cross_only','stability_only','additive',
      'no_shared','no_question_reader','mlp','deepsets','attention',
      'linear','state_threshold','threshold','binary']


class JevTrack(nn.Module):
    def __init__(self,arm='jev',d=64):
        super().__init__();assert arm in ARMS;self.arm=arm;self.d=d
        self.identity=ReliableIdentityPolicy('multi_question','availability_joint')
        for p in self.identity.parameters():p.requires_grad_(False)
        self.option=nn.Linear(256,d);self.history=nn.Linear(256,d)
        self.current=nn.Linear(256,d);self.shared=nn.Linear(256,d)
        self.evidence=nn.Linear(36,d);self.norm=nn.LayerNorm(d)
        self.tasks=nn.Embedding(2,d) if arm not in ['mlp','deepsets','attention','linear','state_threshold','threshold','binary'] else None
        if arm in ['mlp','linear','state_threshold','threshold','binary']:
            self.ordinary=nn.Sequential(nn.Linear(5*d,160),nn.GELU(),nn.Linear(160,d))
        elif arm=='deepsets':
            self.ordinary=nn.ModuleList([nn.Sequential(nn.Linear(d,256),nn.GELU(),nn.Linear(256,d)) for _ in range(2)])
        else:self.readers=nn.ModuleList([Reader(d),Reader(d)])
        self.cross_head=nn.Linear(d,1);self.continuity_head=nn.Linear(d,1)
        if arm!='linear':
            self.edge_value=nn.Linear(d,1)
            nn.init.zeros_(self.edge_value.weight);nn.init.zeros_(self.edge_value.bias)
        if arm=='linear':
            self.fusion=nn.Linear(7,1)
            nn.init.zeros_(self.fusion.weight);nn.init.zeros_(self.fusion.bias)
        if arm in ['mlp','deepsets']:
            self.plan_pool=nn.Sequential(nn.Linear(2*d,128),nn.GELU(),nn.Linear(128,d),nn.GELU())
        elif arm not in ['linear','state_threshold','threshold','binary','additive']:
            self.plan_reader=Reader(d)
        if arm not in ['linear','state_threshold','threshold','binary','additive']:
            self.plan_value=nn.Linear(d,1)
            nn.init.zeros_(self.plan_value.weight);nn.init.zeros_(self.plan_value.bias)

    def train(self,mode=True):
        super().train(mode)
        self.identity.eval()
        return self

    def encode(self,x):
        base={k:v for k,v in x.items() if k!='commitment_features'}
        cache={};core=self.identity.core;original=core.encode_state
        def encoded(value):
            result=original(value);cache['state']=result;return result
        core.encode_state=encoded
        hook=core.norm.register_forward_hook(lambda m,a,o:cache.update(option=o))
        try:
            with torch.no_grad():result=self.identity.details(base)
        finally:
            core.encode_state=original;hook.remove()
        state,h,det=cache['state'];option=cache['option']
        d=self.d;b,q,_=det.shape;k=h.shape[1]
        shared=self.shared(state);current=self.current(det)
        own=self.history(h[:,:,2]);other=self.history(h[:,:,3])
        own=own*x['history_mask'][:,:,2,None];other=other*x['history_mask'][:,:,3,None]
        zero=own.new_zeros(b,1,d);own=torch.cat([own,zero],1);other=torch.cat([other,zero],1)
        context=torch.cat([x['pair_evidence'],x['commitment_features']],-1)
        evidence=self.evidence(context)
        evidence=torch.cat([evidence,evidence.new_zeros(b,q,1,d)],2)
        opt=self.option(option);det=current[:,:,None].expand(-1,-1,k+1,-1)
        own=own[:,None].expand(-1,q,-1,-1);other=other[:,None].expand(-1,q,-1,-1)
        combined=opt+det+evidence+own+other
        if self.arm in ['mlp','linear','state_threshold','threshold','binary']:
            feature=self.norm(self.ordinary(torch.cat([opt,det,evidence,own,other],-1)))
            cross=feature;stability=feature
        elif self.arm=='deepsets':
            legal=torch.cat([x['legal'],x['question_mask'][:,:,None]],-1)
            feature=combined
            for block in self.ordinary:
                pooled=(feature*legal[...,None]).sum(2)/legal.sum(2).clamp_min(1)[...,None]
                feature=feature+block(feature+pooled[:,:,None])
            cross=self.norm(feature);stability=cross
        elif self.arm=='attention':
            flat=combined.reshape(b,q*(k+1),d)
            for reader in self.readers:flat=reader(flat,shared)
            cross=self.norm(flat.reshape(b,q,k+1,d));stability=cross
        else:
            features=[]
            if self.arm=='no_shared':
                # Actual recomputation, not a zeroed shared state proxy.
                with torch.no_grad():second_state=original(base)[0]
                states=[shared,self.shared(second_state)]
            else:states=[shared,shared]
            for task in range(2):
                view=other if task==0 else own
                option_query=combined+self.tasks.weight[task]
                if self.arm=='fixed_question':option_query=torch.zeros_like(combined)+self.tasks.weight[task]
                if self.arm=='no_question_reader':read=option_query
                else:read=self.readers[task](option_query.reshape(b,q*(k+1),d),states[task]).reshape(b,q,k+1,d)
                features.append(self.norm(opt+view+evidence+read))
            cross,stability=features
        if self.arm=='cross_only':stability=stability*0
        if self.arm=='stability_only':cross=cross*0
        cross_logits=self.cross_head(cross).squeeze(-1)[:,:,:k]
        stability_logits=self.continuity_head(stability).squeeze(-1)[:,:,:k]
        feature=(cross+stability)/2
        if self.arm=='linear':
            p=x['pair_evidence'];c=x['commitment_features']
            numeric=torch.stack([p[:,:,:,:4].max(-1).values,-p[:,:,:,10:14].square().sum(-1).sqrt(),
                                 p[:,:,:,3],c[:,:,:,0],1/(1+p[:,:,:,4].abs()),p[:,:,:,6],p[:,:,:,9]],-1)
            delta=self.fusion(numeric).squeeze(-1)
            delta=torch.cat([delta,delta.new_zeros(b,q,1)],-1)
        elif self.arm=='state_threshold':
            delta=-self.edge_value(shared.mean(1))[:,None].expand(-1,q,k)
            delta=torch.cat([delta,delta.new_zeros(b,q,1)],-1)
        elif self.arm in ['threshold','binary']:
            delta=-self.edge_value(feature[:,:,:k]).squeeze(-1)
            delta=torch.cat([delta,delta.new_zeros(b,q,1)],-1)
        else:delta=self.edge_value(feature).squeeze(-1)
        return dict(base_logits=result['logits'],edge_logits=result['logits']+delta,
                    features=feature,shared=shared,cross_logits=cross_logits,stability_logits=stability_logits)

    def score_plans(self,details,x,index,alternatives):
        q=int(x['question_mask'][index].sum());k=int(x['identity_mask'][index].sum())
        base=torch.cat([details['edge_logits'][index,:q,:k],details['edge_logits'][index,:q,-1:]],-1)
        feature=torch.cat([details['features'][index,:q,:k],details['features'][index,:q,-1:]],1)
        selected,indices=selected_tokens(feature,alternatives)
        rows=torch.arange(q,device=base.device)[None].expand(len(alternatives),-1)
        initial=base.double()[rows,indices].sum(-1)
        if self.arm in ['linear','state_threshold','threshold','binary','additive']:
            residual=initial*0
        elif self.arm in ['mlp','deepsets']:
            residual=self.plan_value(self.plan_pool(torch.cat([selected.mean(1),selected.max(1).values],-1))).squeeze(-1)
        else:
            query=details['shared'][index].mean(0)[None,None].expand(len(alternatives),1,-1)
            pooled=self.plan_reader(query,selected).squeeze(1)
            residual=self.plan_value(pooled).squeeze(-1)
        return initial+residual.double()

    def details(self,x):
        if not x['question_mask'].any():
            k=x['identity_mask'].shape[1]
            empty=x['detection_visual'].new_empty(len(x['question_mask']),0,k+1)
            return dict(base_logits=empty,edge_logits=empty,plans=[[]],plan_scores=[])
        result=self.encode(x);all_plans=[];scores=[]
        for index in range(len(x['question_mask'])):
            q=int(x['question_mask'][index].sum());k=int(x['identity_mask'][index].sum())
            logits=torch.cat([result['base_logits'][index,:q,:k],result['base_logits'][index,:q,-1:]],-1)
            alternatives=plans(logits,x['legal'][index,:q,:k])
            all_plans.append(alternatives);scores.append(self.score_plans(result,x,index,alternatives))
        result.update(plans=all_plans,plan_scores=scores,arm=self.arm)
        return result

    def forward(self,x):
        result=self.details(x)
        if not self.training:
            self.runtime_details=result
            if result['plan_scores']:
                selected=int(result['plan_scores'][0].argmax())
                self.runtime_choice=list(result['plans'][0][selected])
            else:self.runtime_choice=[]
        return result['edge_logits']
