"""Audit actual submit, unknown masking, bank support and dependent events."""
import collections,gzip,torch
from jev_fastcheck_common import *
def main():
 torch.set_num_threads(1)
 from jev_fastcheck_distribution import records
 from jev_fastcheck_rollout import LabelState
 from jev_phase13_runtime import cache_inputs
 summaries=[]
 for video in [12,13,14]:
  result=read(REPORTS/f'ROLLOUT_self_VIDEO{video:02d}.json');r_by_key={tuple(r['key']):r for r in records(result)}
  _,_,reader=cache_inputs(video);state=LabelState(video,reader);counts=collections.Counter();dedup=set();owners=set();previous_bank=[]
  events=list(map(json.loads,gzip.open(result['journal']['path'],'rt')));state.boot(0,1-events[0]['key'][2])
  for e in events:
   r=r_by_key.get(tuple(e['key']));frame,view=e['key'][1:]
   if r:
    k=len(r['refs']);legal=r['inputs']['legal'][0]
    assert state.labels.current(frame,view)==r['GT_labels_OFFLINE_ONLY']
    for row,g in enumerate(r['GT_labels_OFFLINE_ONLY']):
     col=r['native_choice'][row]
     if col>=0:assert e['ids'][row]==r['refs'][col];counts['actual_active_submit_exact']+=1
     if g is None:
      assert not r['known_options'][row].any() and not r['positive'][row].any() and not r['supervised'][row];counts['UNKNOWN_current_mask_pass']+=1;continue
     active=bool(r['positive'][row,:k].any())
     stale=[identity for identity in r.get('stale_refs',[]) if state.owner.get(identity)==g]
     # Bank support known from actual REACT candidate sets is lawful; membership
     # in the prior Bank alone is reported separately, never called executable.
     if not active and stale:counts['certified_correct_only_actual_legal_Bank']+=1
     elif not active and any(state.owner.get(i)==g for i in previous_bank):counts['prior_Bank_support_without_current_REACT_candidate']+=1
     elif not active:counts['no_certified_legal_active_or_queried_Bank']+=1
     mixed=any(len(state.votes[i])>1 and g in state.votes[i] for i in r['refs'])
     counts['mixed_history_rows']+=mixed
     knownwrong=r['known_options'][row,:k]&~r['positive'][row,:k]&legal[row]
     similarity=r['inputs']['pair_evidence'][0,row,:,:4].max(-1).values
     hard=bool((knownwrong&(similarity>=.75)).any())
     counts['high_similarity_wrong_identity_rows']+=hard
     category=r['eligibility_categories'][row];dedup.add((video,g,view,frame//64,category));owners.add((video,g))
    for p in r['plans']:
     cols=[k if col<0 else col for col in p]
     counts['fully_certified_plans_once']+=int(all(r['known_options'][row,col] for row,col in enumerate(cols)))
   state.update(frame,view,e['ids']);previous_bank=e['Bank_IDs']
  summaries.append(dict(video=video,counts=dict(counts),person_camera_time_stratum_groups=len(dedup),dependent_person_components=len(owners),source=ref(REPORTS/f'ROLLOUT_self_VIDEO{video:02d}.json')))
 save(REPORTS/'TRAIN_SAMPLE_INTERFACE_AUDIT.json',dict(status='PASS_ACTUAL_SUBMIT_AND_UNKNOWN_MASK',rows=summaries,
  old_distribution_plan_counter='fully_certified_plans in initial STATE_DISTRIBUTION_AUDIT counts plan-row visits, not unique plans; this supplement provides fully_certified_plans_once and that earlier field must not be used as unique-plan support',
  lawful_Bank_not_same_as_prior_membership=True,labels_unchanged=True,no_new_rollout=True,no_extra_gradients=True,no_independent_safety_claim=True))
 print('SAMPLE_INTERFACE_AUDIT_PASS',[(x['video'],x['counts']) for x in summaries],flush=True)
if __name__=='__main__':main()
