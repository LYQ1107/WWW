"""Execute only the two frozen prefix rules; cold expansion requires later safety gate."""
from jev_phase17_common import *
from jev_phase17_queue import pin,queue
import argparse
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--cold-qualified',action='store_true');args=parser.parse_args()
 if args.cold_qualified:
  source=Path(read(OUT/'source_soft_v1.json')['path']);gates=read(REPORTS/'SOFT_MEMORY_CAUSAL_RESULTS.json')['rules'];jobs=[dict(key=f'cold_{rule}_video{v}',script='jev_phase17_soft.py',args=['--video',str(v),'--rule',rule,'--scope','cold'],result=str(OUT/'soft/cold'/rule/f'video{v:02d}/RESULT.json')) for rule in ['E1','E2'] if gates[rule]['full_cold_qualified'] for v in DEV]
  queue(source,jobs,'soft_cold_queue_v1');sys.exit(0)
 source=pin('source_soft_v1');jobs=[]
 for rule in ['E1','E2']:
  jobs.append(dict(key=f'contract_{rule}_video17',script='jev_phase17_soft.py',args=['--video','17','--rule',rule,'--scope','contract'],result=str(OUT/'soft/contract'/rule/'video17/RESULT.json')))
  for v in TRAIN:jobs.append(dict(key=f'prefix_{rule}_video{v}',script='jev_phase17_soft.py',args=['--video',str(v),'--rule',rule,'--scope','prefix'],result=str(OUT/'soft/prefix'/rule/f'video{v:02d}/RESULT.json')))
 queue(source,jobs,'soft_queue_v1')
