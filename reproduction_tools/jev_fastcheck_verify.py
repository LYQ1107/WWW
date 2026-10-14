"""Check finished evidence, parity, retained artifacts and bounded scope."""
import argparse, shutil
from jev_fastcheck_common import *

def main(final=False):
 checks={};training={}
 protocol=read(REPORTS/'PREREGISTRATION.json')
 assert protocol['gpu_hours_max']==24 and protocol['authorized_TRAIN']==[12,13,14,16]
 assert protocol['gradient_videos']==[12,13,14] and protocol['new_version_limit']==1
 for p in REPORTS.glob('TRAIN_*_STEPS*.json'):
  d=read(p);training[d['arm'],d['dataset'],d['steps']]=d
  assert d['status']=='COMPLETE' and d['training_signal_eligible']
  assert sha(d['checkpoint']['path'])==d['checkpoint']['SHA256']
  assert d['binding']['protocol']['SHA256']==sha(REPORTS/'PREREGISTRATION.json')
 assert len(training)==10
 for dataset in ['D0','D1']:
  for steps in [300,1000]:
   a=training['attention',dataset,steps];j=training['jev',dataset,steps]
   assert a['schedule_SHA']==j['schedule_SHA']
   assert a['common_loss_weights']==j['common_loss_weights']
   assert a['original_events']==j['original_events'] and a['self_events']==j['self_events']
   assert a['actual_new_updates']==j['actual_new_updates']==(300 if steps==300 else 700)
 a=training['direct_attention','D1',1000];j=training['direct_jev','D1',1000]
 assert a['schedule_SHA']==j['schedule_SHA'] and a['loss_weights']==j['loss_weights']
 assert a['one_actual_MATCH_solve'] and j['one_actual_MATCH_solve']
 checks['training_pool_schedule_loss_budget_checkpoint_parity']='PASS'
 for name in ['CONTRACT_TESTS.json','DIRECT_CONTRACT_TESTS.json','TRAINED_CONTRACT_TESTS.json','TRAINED_DIRECT_CONTRACT_TESTS.json']:
  d=read(REPORTS/name);assert d['status']=='PASS'
  for c in d['cases']:
   assert all(c[k] for k in ['zero_residual_exact','illegal_and_capacity_pass','exception_restoration_pass','permuted_native_identity_choice_exact'])
   assert sha(c['checkpoint']['path'])==c['checkpoint']['SHA256']
 checks['initial_and_final_checkpoint_actual_input_contracts']='PASS'
 online=list(REPORTS.glob('ONLINE_*_STEPS1000.json'));assert len(online)==6
 for p in online:
  d=read(p);assert d['status']=='COMPLETE' and d['video']==16 and d['frames']==1034
  assert d['actual_mutated_state'] and not d['GT_actor_inputs'] and not d['independent_validation']
  assert d['counts']['MATCH_rows']==58890 and d['full_image_FPS'] is None
  for k in ['raw_predictions','metrics_source','source','checkpoint']:
   assert sha(d[k]['path'])==d[k]['SHA256']
 checks['six_complete_native_GT_separate_official_metrics']='PASS'
 correction=read(REPORTS/'COUNT_SCHEMA_CORRECTION.json')
 for c in correction['cases']:
  assert sha(c['prior'])==c['original_SHA256'] and sha(c['log'])==c['log_SHA256']
  old=read(c['prior']);now=read(REPORTS/pathlib.Path(c['prior']).name)
  assert all(old[k]==now[k] for k in ['raw_online_metrics','CVIDF1','CVMA','checkpoint'])
 # The activation gate cites the immutable pre-correction report snapshots.
 for r in read(REPORTS/'P3_ACTIVATION_GATE.json')['evidence']:
  assert sha(REPORTS/'schema_snapshots'/pathlib.Path(r['path']).name)==r['SHA256']
 checks['count_schema_correction_preserves_metrics_and_activation_evidence']='PASS'
 live={read(p)['tag']:read(p) for p in REPORTS.glob('LIVE_*_STEPS*.json')}
 if final:assert set(live)=={'parent','threshold','attention_D1','direct_jev_D1'}
 assert len({d['GPU_UUID'] for d in live.values()})==1
 for d in live.values():
  assert d['status']=='COMPLETE' and len(d['trials'])==3 and d['native_prediction_SHA']
  assert all(t['scene_frames']==1034 for t in d['trials'])
 checks['same_physical_GPU_three_full_image_trials']='PASS_COMPLETE' if len(live)==4 else 'PARTIAL_PENDING_THRESHOLD'
 changed=subprocess.check_output(['git','diff','--name-only',BASE],cwd=ROOT,text=True).splitlines()
 allowed=('docs/JEV_FASTCHECK/','reports/JEV_FASTCHECK/','gtr/modeling/jev_fastcheck/','reproduction_tools/jev_fastcheck_')
 assert all(p.startswith(allowed) for p in changed),changed
 assert subprocess.check_output(['git','rev-parse',BRANCH+'~0'],cwd=ROOT,text=True).strip()
 original='/home/liuyeqiang/WWW_jev_recovery_20261012'
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=original,text=True).strip()==BASE
 checks['old_source_and_recovery_HEAD_preserved']='PASS'
 registry=read(REPORTS/'EXPERIMENT_REGISTRY.json')
 active=[n for n,d in registry['jobs'].items() if d['status']=='RUNNING']
 failed=[n for n,d in registry['jobs'].items() if d['status']=='FAILED']
 if final:assert not active and not failed,(active,failed)
 hours=sum(d.get('seconds',0)/3600 for d in registry['jobs'].values() if d.get('GPU') is not None)
 retained=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
 free=shutil.disk_usage('/data1').free/2**30
 assert hours<24 and retained<2*2**30 and free>=30
 docs=['PROTOCOL','BASELINE_AND_ACCESS_AUDIT','STATE_DISTRIBUTION_REPORT','CLOSED_LOOP_TRAINING_RESULTS','ARCHITECTURE_DIAGNOSIS','ONLINE_EVALUATION','REVIEWER_FIVE_QUESTIONS','FINAL_DECISION']
 assert all((DOCS/(n+'.md')).is_file() for n in docs)
 assert all(p.stat().st_size<2**20 for p in REPORTS.rglob('*') if p.is_file())
 subprocess.run(['git','diff','--check'],cwd=ROOT,check=True)
 checks['budget_retention_disk_required_artifacts']='PASS'
 save(REPORTS/'COMPLETION_VERIFICATION.json',dict(status='PASS' if final else 'PARTIAL_LIVE_TIMING_IN_PROGRESS',checks=checks,live_jobs=active,failed_jobs=failed,
  completed_registered_GPU_wall_hours=hours,manual_setup_probes_not_registered=True,runtime_bytes=retained,free_data1_GiB=free,
  scientific_success=False,independent_validation=False,source=binding('Final evidence and scope verification')))
 print('VERIFICATION',checks,active,hours,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');main(p.parse_args().final)
