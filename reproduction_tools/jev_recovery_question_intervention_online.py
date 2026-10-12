"""Frozen final-fusion interventions in real cold-start mutated-state tracking."""
import argparse
import torch
from jev_recovery_common import *
from jev_recovery_learning import load_model
from jev_recovery_question_association import fused,scores


class FrozenIntervention(torch.nn.Module):
    def __init__(self,actor,mode):
        super().__init__();self.actor=actor;self.mode=mode
    def forward(self,x):
        original=self.actor.details(x)
        if original['plan_scores']:
            d=fused(self.actor,original,x,self.mode)
            assert len(original['plans'])==1,'Native camera call is one batch'
            d['plan_scores']=[scores(self.actor,d,x,original['plans'][0],self.mode)]
            self.runtime_choice=list(d['plans'][0][int(d['plan_scores'][0].argmax())])
        else:d=original;self.runtime_choice=[]
        self.runtime_details=d;return d['edge_logits']


def main(mode):
    protect()
    import jev_oneweek_online as old
    from jev_recovery_online import configure
    configure(old,2,'XIV')
    old.OUT=OUT/'question_intervention_native';old.REPORTS=REPORTS/'question_intervention_native'
    def policy(arm,seed,phase):
        actor,checkpoint,_=load_model('jev',seed,phase,2,'XIV')
        return FrozenIntervention(actor,mode).eval(),checkpoint
    old.policy=policy;old.main(mode,20261009,'pilot',16)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',required=True,choices=['cross_representation_only',
        'persistent_representation_only','mean_representations','no_joint_plan_residual']);a=p.parse_args()
    try:main(a.mode)
    except Exception:
        import traceback
        failure('frozen_question_native_'+a.mode,traceback.format_exc());raise
