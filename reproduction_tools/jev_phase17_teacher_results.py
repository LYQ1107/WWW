"""Measured no-head probes; owner/visual labels and unavailable controls kept separate."""
from jev_phase17_common import *
import gzip,math,collections,statistics

def rows(path):
 with gzip.open(path,'rt') as f:return [json.loads(line) for line in f]
def metrics(items):
 if not items:return dict(n=0,accuracy=None,NLL=None,Brier=None,ECE=None)
 correct=[max(range(len(p)),key=p.__getitem__)==y for p,y in items];nll=[-math.log(max(1e-12,p[y])) for p,y in items];brier=[sum((v-(i==y))**2 for i,v in enumerate(p)) for p,y in items];ece=0
 for b in range(10):
  ix=[i for i,(p,y) in enumerate(items) if min(9,int(max(p)*10))==b]
  if ix:ece+=len(ix)/len(items)*abs(sum(max(items[i][0]) for i in ix)/len(ix)-sum(correct[i] for i in ix)/len(ix))
 return dict(n=len(items),accuracy=sum(correct)/len(items),NLL=statistics.mean(nll),Brier=statistics.mean(brier),ECE=ece,temperature=1,calibration_fitted=False)
def main():
 protect();r=read(OUT/'teacher_inference_v2/RESULT.json');boundary=read(OUT/'teacher_inference/BOUNDARY_AND_CACHE.json');events=read(OUT/'owner_audit/TEACHER_EVENTS.json');pairs=read(OUT/'teacher_data/PAIRS.json');evmap={i:e for i,e in enumerate(events)};summary={};detail=[]
 for variant in r['variants']:
  records=rows(variant['records']['path']);assert len(records)==261;byq=collections.defaultdict(list);groups=collections.defaultdict(list);n01=n10=0;unknown=collections.Counter();extra=[];good_same=good_diff=0
  for item in records:
   group=item['group'];ev=evmap[group];cand=next(c for c in ev['candidates'] if c['identity']==item['candidate_identity']);evaluation=item['evaluation'];labels=[evaluation[k] for k in ['WHO_label_OFFLINE','OWNER_visual_consistency_label_OFFLINE','AMBIGUITY_label_OFFLINE','CROSS_VIEW_label_OFFLINE','FALSE_PREMISE_label_OFFLINE']]
   for q,p,y in zip(item['questions'],item['probabilities'],labels):
    pred=max(range(len(p)),key=p.__getitem__)
    if y is not None:byq[q].append((p,y))
    if q=='WHO':
     if y is None:unknown['unknown_or_mixed_pairs']+=1;unknown['abstained_unknown_pairs']+=pred==2
     else:
      # Existing native REACT .75 is a frozen appearance diagnostic, not a MATCH rule.
      baseline=0 if cand['visual_cosine']>=.75 else 1;baseok=baseline==y;ok=pred==y;n01+=not baseok and ok;n10+=baseok and not ok;groups[group].append((p,y));good_same+=y==0 and ok;good_diff+=y==1 and ok
     if ev['wrong_owner'] and ev['certified_correct_available']:
      extra.append(dict(event_key=ev['GT_free_event_key'],candidate=item['candidate_identity'],candidate_is_qualified_correct=item['candidate_identity'] in ev['certified_correct_available'],selected_by_frozen_JEV=cand['identity']==ev['selected_ref'],visual_label=y,predicted_choice=pred,probabilities=p,ReID_cosine=cand['visual_cosine']))
  summary[variant['variant']]=dict(questions={q:metrics(v) for q,v in byq.items()},WHO_group_mean_accuracy=statistics.mean(metrics(v)['accuracy'] for v in groups.values()),groups_with_known_pairs=len(groups),same_correct=good_same,different_correct=good_diff,unknown=dict(unknown),pair_N01_vs_fixed_cosine075=n01,pair_N10_vs_fixed_cosine075=n10,independence='pairs are clustered within12 events; no significance claim; count improvements are not independent safe native corrections',only_native_correctable_event=extra)
  detail.extend(extra)
 parity_paths=[q for p in r['parity'] for q in p['paths']];allparity=all(q['PASS'] for q in parity_paths) and r['actual_token_boundaries_verified'] and all(p['prefix_positions_identical'] and p['KV_fork_no_alias'] for p in r['parity'])
 parity=dict(status='COMPLETE',native_cache_qualification='PASS_BOUNDED_EXECUTION_PARITY' if allparity else 'NO_GO_CACHE_PARITY',tests=r['parity'],token_boundary=dict(original_failed_attempt=ref(OUT/'teacher_inference/BOUNDARY_AND_CACHE.json'),all_new_actual_full_tokens_verified=r['actual_token_boundaries_verified']),probability_tolerance=.02,valid_logit_tolerance=.05,option_order_exact_tests=sum(p['option_reverse_decision_exact'] for p in r['parity']),option_order_total=len(r['parity']),all_option_order_exact=all(p['option_reverse_decision_exact'] for p in r['parity']),candidate_count_stress={v['variant']:ref(OUT/'teacher_inference_v2'/(v['variant']+'_COUNT_STRESS.json')) for v in r['variants']},candidate_count_scope='K2/3/4/8/16 duplicated uncertainty option stress; not a dynamic multi-ID ranking or proof of order invariance',cross_state_cache_policy=boundary['cache_policy'])
 save(REPORTS/'PHASE17_SHARED_PREFIX_PARITY.json',parity)
 recalls=collections.defaultdict(list);sample_recalls=collections.defaultdict(list)
 for event in rows(OUT/'owner_audit/EVENTS.jsonl.gz'):
  if not event['strict']:continue
  for k,v in event['Recall_at_K'].items():
   if v is not None:recalls[k].append(int(v))
 for ev in events:
  for k,v in ev['Recall_at_K'].items():
   if v is not None:sample_recalls[k].append(int(v))
 controls=dict(D0_GMT_GTA=dict(status='NOT_RUN',metrics=None,reason='original GTA scores not captured in exact factorial frontend; frozen JEV logits are not GMT GTA'),D1_cosine=dict(status='COMPLETE_DIAGNOSTIC',rule='max4slot cosine .75 frozen REACT threshold; numeric visual probe only, not native MATCH ownership control'),D2_pretrained=dict(status='COMPLETE',new_head=False),D3_identity_task_LoRA=dict(status='NOT_RUN',metrics=None,reason='qualified independent identity supervision unavailable'),published_generic_QA_LoRA=dict(status='COMPLETE',identity_task_training=False,new_head=False),D4_same_image_MLP_DeepSets_Attention=dict(status='NOT_RUN',metrics=None,reason='qualified train/reserved correction supervision unavailable; no architecture superiority inference'),trained_linear_head=dict(status='NOT_RUN',metrics=None,reason='same qualification barrier'))
 provenance=dict(actor_commit=r['actual_actor_commit'],actor_source_SHA256={n:hashlib.sha256(subprocess.check_output(['git','show',r['actual_actor_commit']+':'+n],cwd=ROOT)).hexdigest() for n in ['reproduction_tools/jev_phase17_teacher.py','reproduction_tools/jev_phase17_common.py','reproduction_tools/jev_phase17_token_verified_vdm.py']},result=ref(OUT/'teacher_inference_v2/RESULT.json'),note='actual actor commit captured before launch; original failed outputs retained; final repository binding not used as actor commit')
 comparison=dict(status='COMPLETE_BOUNDED_TEACHER',binding=binding(),execution=provenance,models=ref(OUT/'teacher_models/MANIFEST.json'),environment=r['environment'],groups=12,legal_candidate_pairs=261,comparisons=summary,controls=controls,all_legal_candidates=True,top16_truncation=False,global_cross_softmax_ranking=False,Recall_at_K_strict={k:dict(n=len(v),recall=sum(v)/len(v)) for k,v in recalls.items()},Recall_at_K_teacher={k:dict(n=len(v),recall=sum(v)/len(v)) for k,v in sample_recalls.items()},parity=ref(REPORTS/'PHASE17_SHARED_PREFIX_PARITY.json'),new_identity_training=False,new_head=False,scientific_ownership_gain='NOT_ESTABLISHED',qualified_native_teacher_corrections=0,reason='visual person compatibility does not certify owner; only1 correctable group and no teacher-driven native intervention/fair same-image trained control')
 save(REPORTS/'VISUAL_JEV_COMPARISON.json',comparison);save(REPORTS/'PHASE17_NO_NEW_HEAD_COMPARISON.json',comparison)
 save(REPORTS/'NO_NEW_HEAD_ABLATION.json',dict(status='COMPLETE_WITH_UNQUALIFIED_CONTROLS_NOT_RUN',comparison=ref(REPORTS/'VISUAL_JEV_COMPARISON.json'),controls=controls,claim='No extra head was instantiated; published QA adapter is trained, not evidence that no training is required.'))
 save(REPORTS/'PHASE17_VISUAL_EVIDENCE_VALUE.json',dict(status='NEGATIVE_FOR_QUALIFIED_INCREMENTAL_OWNERSHIP_EVIDENCE',comparison=ref(REPORTS/'VISUAL_JEV_COMPARISON.json'),qualified_safe_correction_groups=0,sampled_correctable_groups=1,required_groups=8,isolated_same_person_pair_benefits_do_not_establish_GlobalID_owner_gain=True,larger_model_next=False))
 save(REPORTS/'LATENCY_RESULTS.json',dict(status='COMPLETE_TEACHER_OFFLINE_DIAGNOSTIC',environment=r['environment'],measurements=r['latency'],sample_repeats=3,p95_is_max_of_three=True,N_counts_questions_not_identity_candidates=True,repeated_questions_when_N_gt5=True,online_GMT_new_teacher_latency=None,new_teacher_scene_FPS=None,deployment_qualification=False))
 print('TEACHER_RESULTS_COMPLETE',summary,'parity',allparity,flush=True)
if __name__=='__main__':main()
