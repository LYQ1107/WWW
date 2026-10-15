"""Three-scene cold native comparison; no prefix restoration, GT only diagnoses."""
import gc,gzip
from gmt_jev_runtime import *
from gmt_jev_opportunity import OfflineCertificates,native_assignment

def main():
 from gtr.modeling.gmt_jev import IdentityCorrection
 from gtr.modeling.jev_native_state import prefix_state,fingerprint
 from gmt_jev_learned_checkpoint import load_actor
 from run_jev_phase10_closed_loop import raw_predictions
 event=read(REPORTS/'LEARNED_EARLIEST_DIVERGENCE.json')['methods']['mlp']['earliest_native_numeric_discrepancy']
 assert [event['frame'],event['view']]==[2,1]
 values,frames,images=metadata(16);cases={}
 for kind in ['observe','mlp']:
  seed();m,_,_=model()
  actor=IdentityCorrection('observe').cuda().eval() if kind=='observe' else load_actor('mlp')[0]
  m.gmt_identity_correction=actor;seed();labeler=OfflineCertificates(16,images);boot=[False];record={};id_count=[None]
  def before(**d):
   id_count[0]=d['id_count']
   if d['first'] and not boot[0]:
    other=1-d['view'];inst=d['instances'][other];labeler.update(inst,0,other);actor.committed(inst,other,0);boot[0]=True
   if [d['frame'],d['view']]==[2,1]:
    state=prefix_state(m,**d)
    record['before_native_state_SHA256']=fingerprint(state)
    record['before_derived_summary_SHA256']=fingerprint(actor.summaries)
    record['id_count_before']=d['id_count']
  def observed(p,current,galleries):
   if [p['frame'],p['view']]!=[2,1]:return
   labels=labeler.labels(current,2,1);refs=p['ids'].tolist();targets,categories=labeler.targets(labels,refs)
   r=event['row'];base=native_assignment(p['base_scores'],p['lengths'],m.overlap_thresh)
   changed=native_assignment(p['corrected_scores'],p['lengths'],m.overlap_thresh)
   record.update(base_scores_SHA256=fingerprint(p['base_scores']),feature_SHA256=fingerprint(p['features']),
    candidate_ids=refs,lengths=p['lengths'].cpu().tolist(),
    inspected_row=dict(row=r,GT_label_OFFLINE_ONLY=labels[r],target=targets[r],category=categories[r],
     certified_target_native_ID=refs[targets[r]] if targets[r]>=0 else None,
     base_accepted_col=base[r],modified_accepted_col=changed[r],
     base_accepted_ID=refs[base[r]] if base[r]>=0 else None,
     modified_accepted_ID=refs[changed[r]] if changed[r]>=0 else None,
     base_average_scores=(p['base_scores'][r]/p['lengths']).cpu().tolist(),
     modified_average_scores=(p['corrected_scores'][r]/p['lengths']).cpu().tolist(),
     historical_owners_OFFLINE_ONLY={t:labeler.clean(t) for t in refs}))
  actor.observer=observed;m.jev_native_prefix_observer=before
  def after(**d):
   inst=d['instances'][-1]
   if [d['frame'],d['view']]==[2,1]:
    r=event['row'];actual=int(inst.track_ids[r]);record['inspected_row']['actual_native_ID']=actual
    record['inspected_row']['actual_is_newborn']=actual>record['id_count_before']
    record['after_native_state_SHA256']=fingerprint(dict(galleries=d['galleries'],old_reids=d['old_reids'],
     old_ids=d['old_ids'],possible_ids=d['possible_ids'],hits=d['hits'],id_count=d['id_count'],instances=d['instances']))
   labeler.update(inst,d['frame'],d['view']);actor.committed(inst,d['view'],d['frame'])
  m.jev_candidate_commit_observer=after
  with torch.no_grad():raw,_=run(m,values,frames,2)
  pred=raw_predictions(raw,images)
  reference=read(REPORTS/'ORIGINAL_GMT_METRICS.json')['predictions'] if kind=='observe' else read(REPORTS/'ONLINE_mlp_learned.json')['raw_predictions']
  assert sha(reference['path'])==reference['SHA256']
  with gzip.open(reference['path'],'rt') as f:retained=json.load(f)
  ids={i['id'] for (f,v),i in images.items() if f<=2}
  assert pred==[p for p in retained if p['image_id'] in ids], 'Cold three-scene prefix differs from retained full trajectory'
  record.update(retained_full_run=reference,exact_three_scene_predictions=True)
  cases[kind]=record;del m,actor,raw;gc.collect();torch.cuda.empty_cache()
 checks={key:cases['observe'][key]==cases['mlp'][key] for key in
  ['before_native_state_SHA256','before_derived_summary_SHA256','base_scores_SHA256','feature_SHA256','candidate_ids','lengths']}
 result=dict(status='COMPLETE_NATIVE_FIRST_DISCREPANCY_DIAGNOSTIC',event=[16,2,1,event['row']],cases=cases,
  common_input_checks=checks,all_common_input_checks_pass=all(checks.values()),GT_actor_input=False,
  labels_are_observer_only=True,new_gradient_updates=0,no_persistent_corpus=True,no_prefix_restore=True,
  scope='Real cold three-scene trajectories, original-GMT observe vs fixed300 MLP. Full native state/RNG and derived summaries hashed before current get_asso. Actual native Bank/birth outcomes recorded. This does not measure subsequent intervention-causal propagation duration or full-video first-event-only benefit.',source=binding())
 save(REPORTS/'LEARNED_FIRST_DIVERGENCE_NATIVE.json',result)
 print('FIRST_DISCREPANCY_NATIVE_CHECKS',checks,flush=True)
 print('FIRST_DISCREPANCY_NATIVE_OUTCOMES',{k:r['inspected_row'] for k,r in cases.items()},flush=True)

if __name__=='__main__':main()
