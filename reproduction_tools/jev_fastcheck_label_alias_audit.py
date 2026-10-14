"""Compatibility labels can allow multiple IDs; do not invent a unique owner."""
import collections,torch
from jev_fastcheck_common import *
def main():
 from jev_phase14_artifacts import load_dense
 from jev_fastcheck_distribution import records
 torch.set_num_threads(1);descriptor=read(ROOT/'reports/JEV_RECOVERY/R2_ANCHOR_SUPERVISION.json')['patch'];assert sha(descriptor['path'])==descriptor['SHA256']
 count=collections.defaultdict(collections.Counter)
 for r in load_dense(descriptor['path'])['patches']:
  v,f,cam=r['key'];phase='D0_TRAIN16_audit' if v==16 else 'D0_gradient' if (f//64)%5!=4 else 'D0_reserved_temporal'
  positives=r['positive'][:,:-1].sum(-1);known=r['known_options'].any(-1)
  count[phase]['rows']+=len(positives);count[phase]['known_rows']+=int(known.sum());count[phase]['compatible_active_rows']+=int((positives>0).sum());count[phase]['multiple_compatible_ID_rows']+=int((positives>1).sum())
 for v in [12,13,14]:
  for r in records(read(REPORTS/f'ROLLOUT_self_VIDEO{v:02d}.json')):
   positives=r['positive'][:,:-1].sum(-1);count['D1_self_prefix']['rows']+=len(positives);count['D1_self_prefix']['compatible_active_rows']+=int((positives>0).sum());count['D1_self_prefix']['multiple_compatible_ID_rows']+=int((positives>1).sum())
 save(REPORTS/'LABEL_ALIAS_AUDIT.json',dict(status='COMPLETE_UNCHANGED_LABEL_AUDIT',counts={k:dict(v) for k,v in count.items()},source=descriptor,
  interpretation='Multi-positive native IDs certify current person compatibility, not which unique old Global ID must be retained. First-past owner certificate and global offline identity map are distinct. The audit demonstrates a label/decision ambiguity but does not prove it is the sole degradation cause.',
  weights_or_labels_changed=False,unique_GT_owner_not_fabricated=True,causal_or_architecture_superiority_not_inferred=True))
 print('LABEL_ALIAS_COUNTS',{k:dict(v) for k,v in count.items()},flush=True)
if __name__=='__main__':main()
