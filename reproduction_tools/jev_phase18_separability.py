"""Descriptive full-census evidence audit; no fitted policy or heldout access."""
from jev_phase18_common import *
from jev_phase18_outcomes import rows
import collections
def main():
 protect();out={}
 for v in TRAIN+DIAGNOSTIC:
  result=read(OUT/'census_v2'/f'video{v:02d}/RESULT.json');events=rows(result['events']['path']);classes={}
  for category in ['AVAILABLE_ERROR','UNAVAILABLE_ERROR','CORRECT_CONTROL','UNKNOWN_OWNER_OR_NEW','UNKNOWN_CURRENT']:
   items=[e for e in events if e['category']==category];c=collections.Counter();recall=collections.Counter();den=0
   for e in items:
    good=set(e['correct_owner_refs_OFFLINE']);legal=[j for j,m in enumerate(e['legal']) if m];c['events']+=1
    c['actual_REACT_queried']+=e['actual_REACT_refs'] is not None
    c['known_anchored_correct_in_MATCH']+=bool(good)
    c['current_pure_correct_in_MATCH']+=bool(e['current_pure_correct_refs_OFFLINE'])
    c['correct_in_GT_free_alternatives']+=bool(e['alternative_owner_coverage_OFFLINE'])
    if not good or not legal:continue
    den+=1;best=max(legal,key=lambda j:e['scores'][j]);cosbest=max(legal,key=lambda j:e['cosine'][j])
    c['best_legal_NN_has_correct_anchored_owner']+=e['refs'][best] in good
    c['best_legal_cosine_has_correct_anchored_owner']+=e['refs'][cosbest] in good
    c['best_NN_owner_correct_but_original_MATCH_not_correct']+=e['refs'][best] in good and e['original_MATCH_ref'] not in good
    assert len(e['scores'])==len(e['refs'])+1
    preferred=e['refs'][best] in good and e['scores'][best]>e['scores'][-1]
    c['best_correct_NN_candidate_above_private_DEFER']+=preferred
    c['strict_row_score_preference_displaced_in_joint_MATCH']+=preferred and e['original_MATCH_ref'] not in good and (e['original_MATCH_ref'] is None or e['scores'][best]>e['scores'][e['refs'].index(e['original_MATCH_ref'])])
    c['all_correct_owner_refs_mixed_history']+=all(e['mixed_history'][e['refs'].index(g)] for g in good)
    for k in [1,4,8,16,32]:recall[str(k)]+=bool(e['numeric_Recall_at_K'].get(str(k)))
   classes[category]=dict(counts=dict(c),correct_anchored_pool_denominator=den,numeric_cosine_Recall_at_K={k:dict(numerator=n,denominator=den,value=n/den if den else None) for k,n in recall.items()})
  out[str(v)]=dict(source=ref(OUT/'census_v2'/f'video{v:02d}/RESULT.json'),classes=classes,duplicate_masked_owner_labels=True)
 save(REPORTS/'FULL_CENSUS_EVIDENCE_SEPARABILITY.json',dict(status='DESCRIPTIVE_COMPLETE',binding=binding(),per_video=out,scope='full frozen policy trajectories; anchored-owner ranking proxies, not supervised accuracy or independent policy validation',availability='active legal MATCH only; accepted MATCH counterfactual stale-Bank owner coverage unobserved',past_only_GT_offline=True,no_policy_fit=True,no_threshold_search=True,candidate_generation_recall_vs_ranking_distinct=True,information_impossibility_proven=False,solver_note='NN best row candidate can differ from native joint Hungarian outcome due one-column capacity and private DEFER; no isolated row decision attribution'))
 print('CENSUS_EVIDENCE_AUDIT_COMPLETE')
if __name__=='__main__':main()
