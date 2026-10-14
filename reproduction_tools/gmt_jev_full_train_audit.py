"""Fixed original GMT full TRAIN temporal coverage, no corpus or gradients."""
import argparse,collections
from gmt_jev_opportunity import collect
from gmt_jev_runtime import *

def main(video):
 assert video in [12,13,14]
 _,frames,_=metadata(video);buckets={'TRAIN_BLOCKS':collections.Counter(),'HELDOUT_TEMPORAL_BLOCKS':collections.Counter()}
 def observed(s):
  c=buckets['HELDOUT_TEMPORAL_BLOCKS' if s['audit_block'] else 'TRAIN_BLOCKS'];c['rows']+=len(s['native_ids']);c.update(s['categories'])
  for row,(t,y,g,alternative) in enumerate(zip(s['native_ids'],s['targets'].tolist(),s['GT_labels_OFFLINE_ONLY'],s['rule_accepted_cols'])):
   if y<0:continue
   target_id=s['refs'][y];correct=t==target_id
   c['unique_active_rows']+=1;c['GMT_correct_unique_active']+=correct;c['GMT_wrong_unique_active']+=not correct
   before=s['baseline_accepted_cols'][row]
   if alternative==before:continue
   c['rule_changed_unique_rows']+=1
   if alternative<0:c['rule_reject_transition_UNKNOWN_bank_outcome']+=1
   elif s['refs'][alternative]==target_id and not correct:c['rule_local_N01']+=1
   elif correct and s['owners'].get(s['refs'][alternative]) is not None and s['owners'][s['refs'][alternative]]!=g:c['rule_local_N10']+=1
   else:c['rule_changed_UNKNOWN_or_fragment_semantics']+=1
 samples,r=collect(video,frames=frames,records=False,report=False,diagnostic_sample_observer=observed)
 assert samples==[] and r['frames']==frames
 summed=buckets['TRAIN_BLOCKS']+buckets['HELDOUT_TEMPORAL_BLOCKS']
 for key in ['rows','unique_active_rows','GMT_correct_unique_active','GMT_wrong_unique_active','rule_local_N01','rule_local_N10']:
  assert summed[key]==r['counts'].get(key,0),key
 r.update(status='COMPLETE_FULL_TRAIN_TEMPORAL_COVERAGE',buckets={k:dict(v) for k,v in buckets.items()},
  initial_prefix_reference=ref(REPORTS/f'OPPORTUNITY_observe_VIDEO{video:02d}.json'),
  no_persistent_corpus=True,no_gradient_updates=True,training_authorized_by_this_diagnostic_alone=False,
  interpretation='All registered TRAIN videos, fixed coefficients/labels, no development parameter choice. Prefix reports remain untouched. Temporal buckets are diagnostic; do not infer independent validation or authorize training without actual native rule signal and a qualified sampling protocol.')
 save(REPORTS/f'FULL_TRAIN_OPPORTUNITY_VIDEO{video:02d}.json',r)
 print('FULL_TRAIN_AUDIT_COMPLETE',video,r['counts'],r['buckets'],flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--video',type=int,choices=[12,13,14],required=True);main(p.parse_args().video)
