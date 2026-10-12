"""Reuse exact native tracker/evaluator, overriding only this process's namespace."""
import argparse,traceback
from jev_recovery_common import *
from jev_recovery_learning import load_model,initialize

def configure(module,revision,base):
 module.OUT=OUT/f'R{revision}_{base}';module.REPORTS=REPORTS/f'R{revision}_{base}'
 for name in ('protect','binding','save','ref','read','failure'):setattr(module,name,globals()[name])
 def policy(arm,seed,phase):
  if arm=='parent':
   from jev_oneweek_online import FrozenPlanControl
   model,cp=initialize('jev',seed,revision,base)
   return FrozenPlanControl(model.identity,'parent').eval(),cp
  model,cp,_=load_model(arm,seed,phase,revision,base);return model,cp
 module.policy=policy
 return module

def main(kind,arm,seed,phase,revision,base,videos):
 protect()
 if kind=='online':
  import jev_oneweek_online as old
  configure(old,revision,base)
  for video in videos:old.main(arm,seed,phase,video)
 else:
  import jev_oneweek_matlab as old
  configure(old,revision,base);old.main(arm,seed,phase,videos)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--kind',choices=['online','matlab'],default='online');p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,default=20261009);p.add_argument('--phase',default='pilot');p.add_argument('--revision',type=int,default=1);p.add_argument('--base',default='XIV');p.add_argument('--videos',type=int,nargs='+',required=True);a=p.parse_args()
 try:main(a.kind,a.arm,a.seed,a.phase,a.revision,a.base,a.videos)
 except Exception:failure('native_online',traceback.format_exc());raise
