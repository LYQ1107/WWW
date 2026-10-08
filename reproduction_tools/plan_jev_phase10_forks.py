"""Recover exactly the frozen 840 original branch specifications, before rerun."""
import json
from collections import defaultdict
from pathlib import Path
import torch
from jev_phase10_common import *

def main():
 protect();cfg=json.loads((REPORTS/'PHASE10_PREREGISTRATION.json').read_text());old=json.loads((ROOT/'reports/JEV_PHASE9/CAUSAL_DATASET_AUDIT.json').read_text());phase8=json.loads((ROOT/'reports/JEV_PHASE8/NATIVE_CORRECTIVE_ORACLE_AUDIT.json').read_text())
 effects_index=defaultdict(list)
 for r in phase8['raw_manifest']:
  p=Path(r['path'])
  if p.name=='EFFECTS.json':effects_index[str(p.parent.parent)].append(r)
 snapshots=defaultdict(list)
 for r in cfg['original_snapshot_records']:snapshots[tuple(r['key'])].append(r)
 oldevents={tuple(e['key'])+(e['row'],):e for e in old['events']};events=[]
 for index,frozen in enumerate(cfg['events']):
  key=tuple(frozen['key']);row=frozen['row'];e=oldevents[key+(row,)];assert e['group']==frozen['group']and sorted(e['branches'])==frozen['branches']
  choices=snapshots[key];original=None;record=None
  for r in choices:
   assert sha(r['path'])==r['sha256'];s=torch.load(r['path'],map_location='cpu')
   if row in s['descriptors']:
    original=s;record=r;break
  assert original is not None,(key,row)
  refs=tuple(map(int,original['proposal'].track_ids));d=original['descriptors'][row];branches={}
  if e['raw_artifact_root'] is not None:
   sources=[{'path':str(Path(e['raw_artifact_root'])/t/'EFFECTS.json'),'sha256':sha(Path(e['raw_artifact_root'])/t/'EFFECTS.json')}for t in frozen['branches']]
  else:
   # Original Phase VIII snapshots retain their immutable source index in
   # their name and the original raw-manifest binds every branch effect.
   candidates=[]
   for name,values in effects_index.items():
    if not Path(name).name.endswith(f'_row{row:03d}'):continue
    for r in values:
     b=json.loads(Path(r['path']).read_text())
     if Path(r['path']).parent.name=='CONTROL'and b['prediction_sha256']==e['branches']['CONTROL']['prediction_sha256']:candidates.append(values)
   assert len(candidates)==1,(key,row,len(candidates));sources=candidates[0]
  lookup={Path(r['path']).parent.name:r for r in sources}
  for tag in frozen['branches']:
   r=lookup[tag];assert sha(r['path'])==r['sha256'];b=json.loads(Path(r['path']).read_text());pairs=None
   if tag.startswith(('CANDIDATE_','ALTERNATIVE_'))or tag=='JOINT_FEASIBLE':
    p=b['native']['submitted_pairs'];assert max(map(int,p.values()),default=-1)<len(refs),(key,row,tag,len(refs),max(map(int,p.values()),default=-1),r['path'],record['path']);pairs={int(rr):refs[int(cc)]for rr,cc in p.items()}
    assert len(set(pairs.values()))==len(pairs)
   branches[tag]={'key':list(key),'row':row,'tag':tag,'candidate_ids':list(refs),'pairs':pairs,'old_effect_path':r['path'],'old_effect_SHA256':r['sha256']}
  events.append({'index':index,**frozen,'distribution':e['distribution'],'original_snapshot':record,'offline_GT':d['offline_GT'],'correct_candidate_ids':d['correct_candidate_ids'],'proposal_correct':d['proposal_correct'],'branch_specs':branches})
 result={'status':'FROZEN_ORIGINAL_BRANCHES','events':events,'events_count':len(events),'branch_count':sum(len(e['branch_specs'])for e in events),'source_protocol_SHA256':sha(REPORTS/'PHASE10_PREREGISTRATION.json'),'old_dataset_SHA256':sha(ROOT/'reports/JEV_PHASE9/CAUSAL_DATASET_AUDIT.json'),'future_policy':'fresh native GMT OFF, native lifecycle OFF, no prerecorded actions','no_GT_in_live_override':True,'heldout_sealed':True}
 assert result['events_count']==162 and result['branch_count']==840
 p=REPORTS/'PHASE10_NATIVE_FORK_PLAN.json'
 if p.exists():assert json.loads(p.read_text())==result
 else:save(p,result)
 print('FROZEN_BRANCHES',result['events_count'],result['branch_count'])
if __name__=='__main__':main()
