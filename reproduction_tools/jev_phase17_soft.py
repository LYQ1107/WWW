"""Bounded two-rule native closed loops; every branch starts at frozen real state."""
from jev_phase17_common import *
from jev_phase17_factorial import run_case
from gtr.modeling.jev_phase17.soft_memory import SoftEvidenceMemory
import argparse
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--video',type=int,required=True);p.add_argument('--rule',choices=['E1','E2'],required=True);p.add_argument('--scope',choices=['prefix','cold','contract'],default='prefix');a=p.parse_args()
 assert read(REPORTS/'SOFT_MEMORY_PROTOCOL.json')['status']=='PREREGISTERED'
 factory=lambda prefix:SoftEvidenceMemory(a.rule,prefix)
 if a.scope=='prefix':
  cases=[run_case(a.video,a.rule,'prefix',entry=e,memory_factory=factory,namespace='soft',capture_limit=1) for e in read(XV/'train_commitment_prefixes_v1'/f'video{a.video:02d}/RESULT.json')['counterfactual_prefixes']]
  save(OUT/'soft/prefix'/a.rule/f'video{a.video:02d}/RESULT.json',dict(status='COMPLETE',binding=binding(),cases=cases))
 else:run_case(a.video,a.rule,a.scope,stop=33 if a.scope=='contract' else None,memory_factory=factory,namespace='soft',capture_limit=1)
