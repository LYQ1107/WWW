"""Deployment-only baseline interfaces: no dataset/annotation/evaluator imports."""
import dataclasses
import torch
from .model import VisualJev
from ..jev_candidate_models import build_candidate_model

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
