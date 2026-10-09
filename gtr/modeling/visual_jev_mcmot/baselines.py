"""Deployment-only baseline interfaces: no dataset/annotation/evaluator imports."""
import dataclasses
import time
import torch
from .model import VisualJev
from ..jev_candidate_models import build_candidate_model

class FixedNativeController:
    """Ordinary threshold + identical native solver; unmatched means DEFER."""
    def __init__(self):
        self.records=[];self.timings=[]
    def native_context(self,*args,**kwargs):pass
    def match(self,batch,original,galleries,observations):
        from .native_action_adapter import lawful_match
        begin=time.perf_counter();values=batch.scores-batch.thresholds[None]
        assignment,fallback=lawful_match(values,batch)
        end=time.perf_counter()
        self.records.append({'task':'MATCH','abstain_rows':0,'fallback_rows':fallback,'questions':len(observations),'memory_reused_for_questions':0})
        self.timings.append({'task':'MATCH','feature_ms':0.,'policy_assignment_ms':(end-begin)*1000,'total_ms':(end-begin)*1000})
        return assignment,values
    def memory(self,*args,**kwargs):return None
    def reactivation(self,*args,**kwargs):return None

class LegacyNumericAdapter(torch.nn.Module):
    def __init__(self,name):
        super().__init__()
        self.network=build_candidate_model(name)
        self.consequence=torch.nn.Linear(2,3)
        self.variant=name

    def forward_numeric(self,state64,evidence12,legal):
        logits=self.network(state64,evidence12,legal)
        return {'choice_logits':logits[...,0],'consequences':self.consequence(logits)}

    def forward(self,state,questions,options):
        b,q,k=options.mask.shape
        out=self.forward_numeric(questions.context.reshape(b*q,64),options.evidence.reshape(b*q,k,12),options.mask.reshape(b*q,k))
        return {'choice_logits':out['choice_logits'].reshape(b,q,k),'consequences':out['consequences'].reshape(b,q,k,3)}

class NumericalOnlyVisualAdapter(VisualJev):
    def __init__(self):
        super().__init__('full');self.variant='numerical_only'
    def forward(self,state,questions,options):
        state=dataclasses.replace(state,visual=torch.zeros_like(state.visual))
        questions=dataclasses.replace(questions,visual=torch.zeros_like(questions.visual))
        options=dataclasses.replace(options,visual=torch.zeros_like(options.visual))
        return super().forward(state,questions,options)
