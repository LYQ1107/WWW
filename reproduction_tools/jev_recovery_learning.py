"""Reuse immutable inputs and labels, with dependence-aware certified conflict sampling."""
import collections,torch
from jev_recovery_common import *
from jev_oneweek_learning import batch,comparisons,loss,assess,load_records as inherited_load_records
from gtr.modeling.jev_recovery.model import RecoveryTrack

def initialize(arm,seed,revision=1,base='XIV',maximum=4):
 from jev_oneweek_common import seed_all,PHASE14
 source_revision=revision
 if revision==3:source_revision=read(REPORTS/'R3_PREREGISTRATION.json')['mechanism_revision']
 seed_all(seed);model=RecoveryTrack(arm,source_revision,base,maximum).cuda()
 if base=='XIV':
  checkpoint=read(PHASE14/'formal_v2/multi_question/availability_joint'/f'seed{seed}/RESULT.json')['checkpoint']
  assert sha(checkpoint['path'])==checkpoint['SHA256'];model.identity.load_state_dict(torch.load(checkpoint['path'],map_location='cpu')['model'],strict=True)
 else:
  registry=read(ROOT/'reports/JEV_ONEWEEK/REUSED_EVIDENCE_REGISTRY.json')
  checkpoint=next(r['asset'] for r in registry['weights'] if r['label']==f'XIII fixed_question seed{seed}')
  assert sha(checkpoint['path'])==checkpoint['SHA256'];model.identity.core.load_state_dict(torch.load(checkpoint['path'],map_location='cpu')['model'],strict=True)
 assert sum(p.numel() for p in model.parameters() if p.requires_grad)<=500000
 return model,checkpoint

def load_records(revision=1):
 train,validation,sources=inherited_load_records()
 source_revision=revision if revision!=3 else read(REPORTS/'R3_PREREGISTRATION.json')['mechanism_revision']
 if source_revision>=2:
  from jev_phase14_artifacts import load_dense
  result=read(REPORTS/'R2_QUESTION_SUPERVISION.json');cp=result['labels'];assert sha(cp['path'])==cp['SHA256']
  labels={tuple(r['key']):r for r in load_dense(cp['path'])['labels']}
  train=[dict(r,oneweek_labels=labels[tuple(r['key'])]) for r in train]
  validation=[dict(r,oneweek_labels=labels[tuple(r['key'])]) for r in validation]
  sources.append(cp)
 return train,validation,sources

def load_model(arm,seed,phase,revision=1,base='XIV'):
 folder=OUT/f'training_R{revision}'/base/arm/f'seed{seed}'/phase
 result=read(folder/'RESULT.json');checkpoint=result['checkpoint'];assert sha(checkpoint['path'])==checkpoint['SHA256']
 model,initializer=initialize(arm,seed,revision,base)
 saved=torch.load(checkpoint['path'],map_location='cuda:0')
 missing,unexpected=model.load_state_dict(saved['new_modules'],strict=False)
 assert not unexpected and all(n.startswith('identity.') for n in missing)
 assert initializer==saved['initializer'];return model.eval(),checkpoint,result

def conflict_groups(train):
 patch=read(OUT/'audit/PROPOSAL_PATCH.json')['records'];index={tuple(r['key']):r for r in train}
 groups=collections.defaultdict(set)
 for row in patch:
  key=tuple(row['key'])
  if key not in index:continue
  assert row['train_gradient']
  for group in row['proposals']['P1_4']['independent_group_keys']:
   # Lists in JSON become a hashable conservative video/person/ID/time key.
   g=(group[0],tuple(group[1]),tuple(group[2]),group[3],group[4]);groups[g].add(key)
 return {g:[index[k] for k in sorted(keys)] for g,keys in sorted(groups.items())}
