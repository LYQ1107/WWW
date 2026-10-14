"""Honest bounded decision, all real metrics and required A--H answers."""
import argparse,collections,shutil,numpy as np
from jev_fastcheck_common import *
def main(final=False):
 training=[];online=[];cost=[]
 for p in sorted(REPORTS.glob('TRAIN_*_STEPS*.json')):
  d=read(p);training.append(dict(artifact=ref(p),arm=d['arm'],dataset=d['dataset'],steps=d['steps'],checkpoint=d['checkpoint'],trainable_parameters=d['trainable_parameters'],source_state_net=d['final']['net_certified_N01_minus_N10'],rank_accuracy=d['final']['rank_accuracy'],schedule_SHA=d['schedule_SHA'],gradient_groups=d['local_gradient_groups']))
 for p in sorted(REPORTS.glob('ONLINE_*_STEPS1000.json')):
  d=read(p);online.append(dict(tag=d['tag'],artifact=ref(p),metrics=d['raw_online_metrics'],CVIDF1=d['CVIDF1'],CVMA=d['CVMA'],counts=d['counts'],row_certificates_not_whole_plan_causal_N01_N10=True,correct_identity_preservation=d['correct_identity_preservation'],UNKNOWN_change_fraction=d['UNKNOWN_change_fraction'],legal_certified_candidate_coverage_all_detection_queries=d['counts']['certified_active_covered']/d['counts']['MATCH_rows'],top8_correct_absence=d['counts']['correct_active_outside_plans'] if not d['tag'].startswith('direct_') else None,
   direct_coverage_scope='all legal IDs scored; raw correct_active_outside_plans counts unselected compatible actions, not proposal truncation' if d['tag'].startswith('direct_') else None,identity=d['identity_summary'],parameters=d['parameters']))
 for p in sorted(REPORTS.glob('LIVE_*_STEPS*.json')):
  d=read(p);cost.append(dict(tag=d['tag'],artifact=ref(p),mean_full_scene_FPS=d['mean_full_scene_FPS'],Stage2_mean_trial_p50_p95_ms=list(map(float,np.mean([r['stage2_p50_p95_ms'] for r in d['trials']],axis=0))),GPU_UUID=d['GPU_UUID'],parameters=d['parameters'],three_complete_live_image_trials=len(d['trials'])==3))
 required_cost={'parent','threshold','attention_D1','direct_jev_D1'};missing=sorted(required_cost-{c['tag'] for c in cost})
 if final:assert not missing,missing
 parent_cost=next((c for c in cost if c['tag']=='parent'),None)
 if parent_cost:
  for c in cost:c['FPS_change_percent_vs_same_GPU_Fixed']=100*(c['mean_full_scene_FPS']/parent_cost['mean_full_scene_FPS']-1)
 refs=read(REPORTS/'FROZEN_TRAIN16_REFERENCES.json');assert refs['status']=='COMPLETE_EXACT_PRIOR_INPUTS'
 metrics={r['tag']:r for r in online};j0=metrics['jev_D0'];j1=metrics['jev_D1'];a0=metrics['attention_D0'];a1=metrics['attention_D1'];direct=metrics['direct_jev_D1']
 data_effect={arm:{k:metrics[f'{arm}_D1']['metrics'][k]-metrics[f'{arm}_D0']['metrics'][k] for k in ['HOTA','AssA','IDSW','IDF1']} for arm in ['jev','attention']}
 durations=[]
 for name in ['CERTIFIED_ERROR_PROPAGATION.json','CERTIFIED_ERROR_PROPAGATION_DIRECT_AND_REFERENCES.json','GLOBAL_ID_ERROR_DURATION.json']:
  p=REPORTS/name;d=read(p);durations.append(dict(artifact=ref(p),definition=d.get('definition',d.get('status')),cases=[{k:v for k,v in c.items() if k!='episodes'} for c in d['cases']]))
 aliases='Current targets certify first-past native-ID/person compatibility. Multiple native IDs can remain compatible with the same person, and neither the certificate nor local WHO correctness is a unique long-term Global ID or future safety guarantee. This is a supervision/decision-alignment limitation, not an established exclusive architecture cause.'
 answers={
  'A':f"Data aggregation helped ordinary Attention relative to its D0 fine-tune (+{data_effect['attention']['HOTA']:.3f} HOTA); JEV improved only +{data_effect['jev']['HOTA']:.3f} HOTA and IDSW rose +{data_effect['jev']['IDSW']:.0f}. A1 remains below its original Attention initializer; J1 remains below strong Fixed/Threshold. CLOSED_LOOP_NOT_SUFFICIENT_FOR_CURRENT_JEV; no general claim that DAgger cannot work.",
  'B':'Paired native prefixes had identical initial input/state SHA but diverged at TRAIN12/24/1, TRAIN13/225/0, TRAIN14/6/0. Direction differs by video. Full data intervention helped ordinary control more than JEV. Missing/UNKNOWN/Bank-only support and correlated person histories remain. '+aliases,
  'C':'Keep Strong Fixed and honest frozen JEV evidence. The current new Recovery JEV module lacks support for deployment or continued scale-up. Direct JEV remains an auditable diagnostic, not a successful replacement.',
  'D':'The sole direct lawful score/choice repair was executed. Against unchanged J1: HOTA +0.396, IDSW 1891->1093, but CVIDF1 83.511->82.406 and full-video identity-inconsistency duration 6957->7741 camera-frames. Lower IDSW does not establish stable identity benefit. Comparing repair to M0 also changes loss/readout/solver. Matched direct M1/M2 compares complete typed/untyped variants including reader allocation, not question embeddings alone, and M1 still loses to the stronger existing ordinary Attention.',
  'E':'Best matched TRAIN16 reference: frozen Pair Threshold HOTA70.605/AssA71.106/CVIDF189.825/IDSW1002. Best new P2 model: ordinary Attention D1 HOTA68.892/AssA68.073/CVIDF188.038/IDSW1299. Direct JEV:67.052/64.476/82.406/1093. Same-GPU live-image costs are recorded separately; no cached FPS imputation.',
  'F':'No new JEV exceeds Strong Fixed, Threshold or best ordinary Attention. Direct JEV exceeds the matched direct-Attention variant (HOTA63.097) but that variant is weaker than the original/P2 Attention and is not the strongest ordinary baseline.',
  'G':'All new results are seed20261009, reused authorized TRAIN, and pre-exposed TRAIN16. No independent complete video, cross-seed reproduction, new source permission, or future safety qualification was established. DEV/TEST/sealed datasets were not newly used.',
  'H':'Prefer the evidence-supported Strong Fixed/ordinary Threshold baseline and retain the stronger ordinary Attention as research control. Stop adding JEV Questions/Readers. Any next research scope should first resolve lawful action/identity-preservation supervision and independent validation; no new run is authorized by this final decision.'}
 if not missing:
  by_cost={c['tag']:c for c in cost}
  answers['E']+=f" Three complete live-image trial means on the same GPU: Threshold {by_cost['threshold']['mean_full_scene_FPS']:.3f}, Fixed {by_cost['parent']['mean_full_scene_FPS']:.3f}, Attention D1 {by_cost['attention_D1']['mean_full_scene_FPS']:.3f}, Direct JEV {by_cost['direct_jev_D1']['mean_full_scene_FPS']:.3f} scene FPS. Unchanged JEV D1 live FPS was not measured."
 registry=read(REPORTS/'EXPERIMENT_REGISTRY.json');live=[n for n,r in registry['jobs'].items() if r.get('status')=='RUNNING'];failed=[n for n,r in registry['jobs'].items() if r.get('status')=='FAILED'];hours=sum(r.get('seconds',0)/3600 for r in registry['jobs'].values() if r.get('GPU') is not None)
 if final:assert not live and not failed,(live,failed);assert hours<24
 result=dict(status='BOUNDED_FASTCHECK_COMPLETE_NO_JEV_INDEPENDENT_ADVANTAGE' if final else 'NATIVE_SCIENTIFIC_REVIEW_COMPLETE_LIVE_TIMING_IN_PROGRESS',scientific_success=False,independent_validation=False,
  decision='JEV_INDEPENDENT_ADVANTAGE_NOT_ESTABLISHED',data_effect=data_effect,answers=answers,training=training,online=online,references=refs,costs=cost,durations=durations,
  missing_live_timing=missing,live_jobs=live,failed_jobs=failed,registered_GPU_wall_hours=hours,GPU_hours_cap=24,runtime_retention_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),free_data1_GiB=shutil.disk_usage('/data1').free/2**30,
  final_checkpoint_contracts=[ref(REPORTS/n) for n in ['TRAINED_CONTRACT_TESTS.json','TRAINED_DIRECT_CONTRACT_TESTS.json']],
  old_Master_NO_GO_preserved=True,original_Master_complete=False,models_revisions_added=1,source=binding('Bounded closed-loop fastcheck final scientific review'))
 save(REPORTS/'FINAL_RESULTS.json',result)
 rows=['# JEV closed-loop fastcheck final decision — 2026-10-14','','`JEV_INDEPENDENT_ADVANTAGE_NOT_ESTABLISHED`. '+('The bounded fastcheck is complete; scientific success and original master completion are not established.' if final else 'All bounded learning/native comparisons are complete; live-image timing is still running.'),'','All rows below use the same full1034-frame two-camera TRAIN16, one frozen seed. References are exact frozen prior inputs and are separate from historical DEV17–19 three-seed scores. TRAIN16 is gradient-free for new modules but pre-exposed by initializers, so this is development evidence.','','| Method | HOTA | AssA | CVIDF1 | IDSW |','|---|---:|---:|---:|---:|']
 for r in refs['cases']:
  m=r['raw_online_metrics'];rows.append(f"| frozen {r['arm']} | {m['HOTA']:.3f} | {m['AssA']:.3f} | {r['CVIDF1']:.3f} | {m['IDSW']:.0f} |")
 for r in online:
  m=r['metrics'];rows.append(f"| {r['tag']} | {m['HOTA']:.3f} | {m['AssA']:.3f} | {r['CVIDF1']:.3f} | {m['IDSW']:.0f} |")
 rows+=['','Data benefit is stronger in ordinary Attention; JEV aggregation did not protect identity stability. Direct JEV lowers switches but increases global-map fragmentation and lowers CVIDF1. Past-certificate duration and complete-video fixed-map duration are separate diagnostics; neither is called a unique physical causal share.','', '| Live model | Scene FPS | FPS change vs Fixed | Stage2 trial-mean p50/p95 ms | Deployed parameters | New trainable parameters |','|---|---:|---:|---|---:|---:|']
 for r in cost:rows.append(f"| {r['tag']} | {r['mean_full_scene_FPS']:.3f} | {r['FPS_change_percent_vs_same_GPU_Fixed']:+.2f}% | {r['Stage2_mean_trial_p50_p95_ms'][0]:.2f}/{r['Stage2_mean_trial_p50_p95_ms'][1]:.2f} | {r['parameters']['deployed_with_controller']} | {r['parameters']['new_controller_trainable']} |")
 rows+=['','Every reported FPS uses three complete live-image runs on the same physical GPU8, V10032GB, with detector/VFCE/actual Gallery/Bank/assignment and no GT journals/evaluator in the timed loop. Native identity state resets between trials; the loaded model is reused. Common dormant pruning is disclosed and is not a JEV scientific gain. Original instantiated system53515309 parameters; common deployed Fixed35995952; Direct JEV36165107 (net+169155 versus common Fixed); Direct Attention36164979 (net+169027).','',f'GPU budget: {hours:.3f}/24 registered GPU-wall hours (includes loading/IO; small manual setup/contract probes are outside this registry, so this is not total active-kernel time). Missing live timings: {missing}.','']
 for key,text in answers.items():rows+=['## '+key,'',text,'']
 (DOCS/'FINAL_DECISION.md').write_text('\n'.join(rows).rstrip()+'\n')
 (DOCS/'ONLINE_EVALUATION.md').write_text('\n'.join(rows[:rows.index('## A')])+'\n\nFull metrics, counts, errors, raw source hashes, CV conventions and identity-time diagnostics are in FINAL_RESULTS.json. N01/N10 are individual changed-row certificates; changed UNKNOWN counterparts are separate, so they are not whole-plan causal correctness claims. Direct methods score all legal IDs, and their raw outside_plans count means not selected, not missing from a top8 proposal set.\n')
 goal=read(REPORTS/'GOAL.json');goal.update(status=result['status'],scientific_success=False,decision=result['decision'],original_Master_complete=False);save(REPORTS/'GOAL.json',goal)
 if final:registry['status']='BOUNDED_FASTCHECK_COMPLETE';registry['scientific_success']=False;save(REPORTS/'EXPERIMENT_REGISTRY.json',registry)
 print('FINAL_REVIEW',result['status'],missing,live,hours,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');main(p.parse_args().final)
