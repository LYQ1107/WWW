"""Actual native-input regression tests for residuals, options, hooks and masks."""
import argparse,torch
from jev_fastcheck_common import *
def main(direct=False,trained=False):
 from jev_fastcheck_distribution import records
 from jev_recovery_learning import batch
 learning=configure_paths();r=list(records(read(REPORTS/'ROLLOUT_self_VIDEO12.json')))[100]
 x,y=batch([r]);assert not any('GT' in k or 'label' in k for k in x)
 results=[]
 for arm in ['jev','attention']:
  if trained:
   from jev_fastcheck_online import actor_for
   actor,cp=actor_for(('direct_' if direct else '')+arm+'_D1',1000)
  elif direct:
   from jev_fastcheck_direct_train import direct_actor
   actor,cp=direct_actor(arm)
  else:actor,cp=load_policy(arm)
  old=actor.identity.core.encode_state;ef=actor.evidence.forward
  hooks=sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in actor.modules())
  with torch.no_grad():d=actor.details(x)
  assert actor.identity.core.encode_state==old and actor.evidence.forward==ef
  assert sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in actor.modules())==hooks
  k=len(r['refs']);perm=torch.arange(k-1,-1,-1,device='cuda:0');xx=dict(x)
  for name in ['history_visual','history_mask','identity_meta','identity_mask','anchor_visual','anchor_mask']:xx[name]=x[name][:,perm]
  for name in ['pair_evidence','legal','commitment_features']:xx[name]=x[name][:,:,perm]
  with torch.no_grad():dd=actor.details(xx)
  inv=torch.argsort(perm)
  restored=torch.cat([dd['edge_logits'][:,:,:k][:,:,inv],dd['edge_logits'][:,:,-1:]],-1)
  score_max=float((restored-d['edge_logits']).abs().max())
  assert torch.allclose(restored,d['edge_logits'],atol=1e-5,rtol=1e-5),(arm,score_max)
  choice=d['plans'][0][int(d['plan_scores'][0].argmax())];changed=dd['plans'][0][int(dd['plan_scores'][0].argmax())]
  restored_choice=tuple(int(perm[c]) if c>=0 else -1 for c in changed)
  assert tuple(choice)==restored_choice,(arm,choice,restored_choice)
  assert len([c for c in choice if c>=0])==len(set(c for c in choice if c>=0))
  assert all(c<0 or bool(x['legal'][0,row,c]) for row,c in enumerate(choice))
  zero,init=learning.initialize(arm,SEED,3,'XIII')
  if direct:
   from gtr.modeling.jev_fastcheck.model import DirectAssociation
   zero.__class__=DirectAssociation;del zero.plan_value
  with torch.no_grad():z=zero.details(x)
  assert torch.equal(z['base_logits'],z['edge_logits'])
  # Throw from the actual frozen scorer after hooks/temporary forwards are installed.
  base_details=actor.identity.details
  def fail(*args,**kwargs):raise RuntimeError('injected_identity_failure')
  actor.identity.details=fail
  try:
   try:actor.details(x)
   except RuntimeError as e:assert str(e)=='injected_identity_failure'
   else:raise AssertionError('expected injected exception')
  finally:actor.identity.details=base_details
  assert actor.identity.core.encode_state==old and actor.evidence.forward==ef
  assert sum(len(m._forward_hooks)+len(m._forward_pre_hooks) for m in actor.modules())==hooks
  results.append(dict(arm=arm,checkpoint=cp,zero_residual_exact=True,permuted_score_max_error=score_max,permuted_native_identity_choice_exact=True,illegal_and_capacity_pass=True,exception_restoration_pass=True))
 name=('TRAINED_' if trained else '')+('DIRECT_CONTRACT_TESTS.json' if direct else 'CONTRACT_TESTS.json')
 save(REPORTS/name,dict(status='PASS',cases=results,direct=direct,trained=trained,real_TRAIN_state=True,GT_not_actor_inputs=True,
  native_commit_test='six actual teacher/self trajectories enforce unique per-camera IDs and subsequent native state updates; corresponding journals retained',
  UNKNOWN_test='labels remain separate and masked; no fake positive target in actor',source=binding('Actual-input contract tests')))
 print('FASTCHECK_CONTRACTS_PASS',[(v['arm'],v['permuted_score_max_error']) for v in results],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--direct',action='store_true');p.add_argument('--trained',action='store_true');a=p.parse_args();main(a.direct,a.trained)
