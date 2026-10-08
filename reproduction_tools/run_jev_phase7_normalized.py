"""Normalized ordinary MLP supplement; process-local factory, same native runner."""
import argparse
from jev_phase7_common import *
import torch
import jev_phase7_policies as policies
from jev_phase7_mlp_norm import NormalizedMLP
from run_jev_phase7_attribution import run

OriginalActor=policies.MatchActor
class NormalizedActor(OriginalActor):
    def __init__(self,name):
        super().__init__('FIXED');self.name='MLP'
        path=OUT/'normalization_control/training/MLP/model_calibrated.pth'
        payload=torch.load(path,map_location='cpu');self.model=NormalizedMLP()
        self.model.load_state_dict(payload['model'],strict=True);self.model.eval()
        self.temperature=float(payload['temperature'])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--video',required=True,type=int);p.add_argument('--validator',default='OWN',choices=['OWN','LEGACY'])
    p.add_argument('--budget',type=int);a=p.parse_args();policies.MatchActor=NormalizedActor
    tag='MLP_LN_BUDGET'if a.budget is not None else 'MLP_LN_LEGACY_VALIDATOR'if a.validator=='LEGACY'else 'MLP_LN'
    run('MLP',a.video,'cuda:0',tag=tag,validator=a.validator,budget=a.budget)
