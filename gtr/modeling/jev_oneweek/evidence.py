"""Frozen-weight interventions remove evidence before the common encoder too.

The opaque shared ID namespace and legal eligibility remain native. Proposals
are recomputed from the intervened base scorer, so this is an evidence ablation,
not an isolated fixed-proposal scorer intervention or a training comparison.
"""
import torch
from torch import nn
from gtr.modeling.jev_stage2.memory import transform_inputs


def remove_evidence(x,kind):
    assert kind in ['no_cross_evidence','no_stability_evidence']
    if kind=='no_cross_evidence':
        out=transform_inputs(x,'no_cross_camera')
        out['commitment_features'][...,[10,11,12]]=0
        # Feedback is disabled in the new JEV runner. Never admit an optional
        # reliability estimate computed from other-camera observations.
        out['commitment_features'][...,16]=.5;out['commitment_features'][...,17]=0
    else:
        out={key:value.clone() for key,value in x.items()}
        h=out['history_visual'];mask=out['history_mask'];meta=out['identity_meta'];p=out['pair_evidence']
        other=h[:,:,3].clone();present=mask[:,:,3].clone()
        h[:]=0;mask[:]=False;h[:,:,0]=other;h[:,:,1]=other;h[:,:,3]=other
        mask[:,:,0]=present;mask[:,:,1]=present;mask[:,:,3]=present
        meta[:,:,[0,1,2,4,6,9,10]]=0
        p[:,:,:,[4,5,7,10,11,12,13,14,15]]=0
        p[:,:,:,:4]=torch.einsum('bqd,bkhd->bqkh',out['detection_visual'],h)
        c=out['commitment_features'];cross=c[...,[10,11,12]].clone();c[:]=0
        c[...,[10,11,12]]=cross;c[...,16]=.5
    assert torch.equal(out['legal'],x['legal']) and torch.equal(out['identity_mask'],x['identity_mask'])
    return out


class EvidenceIntervention(nn.Module):
    def __init__(self,model,kind):
        super().__init__();self.model=model;self.kind=kind
    def forward(self,x):
        result=self.model(remove_evidence(x,self.kind))
        self.runtime_choice=self.model.runtime_choice
        self.runtime_details=self.model.runtime_details
        return result
