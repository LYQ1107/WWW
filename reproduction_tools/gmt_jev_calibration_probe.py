"""One bounded TRAIN-only feasibility/initialization diagnostic after sparse P1."""
import collections,torch,math
from gmt_jev_opportunity import *
def main():
 assert read(REPORTS/'GMT_ERROR_OPPORTUNITY.json')['learned_training_signal_gate'] is False
 cases=[]
 for v in [12,13,14]:
  records,result=collect(v,report=False);stats=collections.Counter()
  for s in records:
   raw=s['raw_scores'];length=s['lengths'];x=s['features'];avg=raw/length[None]
   confidence=raw-(.1-avg).clamp(0,.05)*length[None]
   proposals={'confidence_feasibility_calibration':confidence}
   if s['key'][1]==0:
    eligible=(x[...,3]<=.1)&(x[...,12]>=.8)
    proposals['first_camera_contrast']=raw+.05*eligible*((x[...,12]-x[...,12].mean(-1,keepdim=True))/.1).clamp(-1,1)*length[None]
   for name,scores in proposals.items():
    chosen=native_assignment(scores,length)
    targets=s['targets'].tolist();owners=s['owners'];gt=s['GT_labels_OFFLINE_ONLY']
    if s['key'][1]==0:
     targets=s['INIT_GT_bridge_targets_DIAGNOSTIC_ONLY'];owners=s['INIT_direct_GT_person_by_committed_ID_DIAGNOSTIC_ONLY'];gt=s['INIT_current_GT_strong_DIAGNOSTIC_ONLY']
    for row,(b,a,t,g) in enumerate(zip(s['baseline_accepted_cols'],chosen,targets,gt)):
     if t<0:continue
     stats[name+'_known_unique_rows']+=1
     if a==b:continue
     stats[name+'_changed_active_assignment_rows']+=1
     target_id=s['refs'][t];old=s['native_ids'][row];new=s['refs'][a] if a>=0 else None
     if a<0:stats[name+'_UNKNOWN_rejection_Bank_outcome']+=1
     elif new==target_id and old!=target_id and (old>s['id_count_before'] or (owners.get(old) is not None and owners[old]!=g)):stats[name+'_N01']+=1
     elif old==target_id and owners.get(new) is not None and owners[new]!=g:stats[name+'_N10']+=1
     else:stats[name+'_UNKNOWN_changed_companion']+=1
   for row,y in enumerate(s['targets'].tolist()):
    if y<0 or s['native_ids'][row]==s['refs'][y]:continue
    stats['GMT_wrong_unique_rows']+=1;stats['wrong_inside_original_cross_conflict_gate']+=bool(s['eligible'][row].any())
    stats['correct_target_GMT_score_above_native_threshold']+=float(avg[row,y])>.1
    stats['wrong_with_other_camera_3plus_support']+=float(x[row,y,10])>=math.log1p(3)/10-1e-7
  cases.append(dict(video=v,counts=dict(stats),original_teacher_source=result['source'],no_corpus_persisted=True));print('CALIBRATION_PROBE',v,dict(stats),flush=True)
 counts=collections.Counter()
 for c in cases:counts.update(c['counts'])
 save(REPORTS/'BOUNDED_CALIBRATION_PROBE.json',dict(status='COMPLETE',cases=cases,counts=dict(counts),
  GT_actor_input=False,first_GT_bridge_scope='Only offline current MATCH diagnosis: IoU>=.8, runner-up GT IoU<.3, unique GT person link to an actually committed first-camera ID. Not a temporal owner/future-safety certificate; original training targets unchanged.',
  first_contrast_rule='Only observable cosine centered across lawful options, first camera link, low GTA margin and cosine>=.8; average residual bound.05.',
  feasibility_calibration='Subtract min(.05,max(0,.1−GMT_average)) times original support, preserving all already-acceptable edge scores and original solver/threshold/Bank/WRITE.',
  rejected_alternatives_not_claimed_causal_corrections=True,source=binding()))
if __name__=='__main__':main()
