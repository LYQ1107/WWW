"""Load only preregistered fixed-endpoint tiny heads; original GMT untouched."""
import torch
from gmt_jev_common import *

def load_actor(kind):
 from gtr.modeling.gmt_jev import IdentityCorrection
 assert kind in ['mlp','attention','cic']
 report=read(REPORTS/'LEARNED_GATE_SMOKE_TRAINING.json')
 assert report['status']=='COMPLETE'
 case=report['cases'][kind];reference=case['checkpoint']
 assert sha(reference['path'])==reference['SHA256']
 checkpoint=torch.load(reference['path'],map_location='cpu')
 assert checkpoint['kind']==kind and checkpoint['steps']==300
 assert checkpoint['risk_mode']=='learned' and checkpoint['cap']==.05 and checkpoint['seed']==SEED
 actor=IdentityCorrection(kind,risk_mode='learned').cuda().eval()
 actor.head.load_state_dict(checkpoint['head'],strict=True)
 for parameter in actor.parameters():parameter.requires_grad_(False)
 return actor,reference
