"""Reuse exact frozen full-TRAIN16 references; recompute only missing CV counts."""
from jev_fastcheck_common import *
def main():
 from jev_fastcheck_online import official_cv
 cases=[]
 for arm in ['parent','threshold','mlp','attention','jev']:
  path=ROOT/f'reports/JEV_RECOVERY/R3_XIII/ONLINE_{arm}_pilot_SEED{SEED}_VIDEO16.json';r=read(path)
  for cp in [r['raw_predictions'],r['metrics_source']]:assert sha(cp['path'])==cp['SHA256']
  folder=OUT/'references'/arm;folder.mkdir(parents=True,exist_ok=True)
  if (folder/'CV_NATIVE.json').exists():
   native=read(folder/'CV_NATIVE.json');assert native['status']=='COMPLETE'
   ids=[x['Identity'] for x in native['results'] if x['convention']=='sequential'];clear=[x['CLEAR'] for x in native['results'] if x['convention']=='interleaved']
   n=sum(x['n_gt'] for x in ids);tr=sum(x['n_tr'] for x in ids);cv=dict(CVIDF1=200*sum(x['IDTP'] for x in ids)/(n+tr),CVMA=100*(1-sum(x['fn']+x['fp']+x['id_switches'] for x in clear)/n),source=ref(folder/'CV_NATIVE.json'),reused_completed_exact_CV_input=True)
  else:cv=official_cv(r,folder)
  case=dict(arm=arm,raw_online_metrics=r['raw_online_metrics'],raw_predictions=r['raw_predictions'],native_tracking_source=ref(path),identity=r['identity_summary'],existing_same_base_same_video_same_seed=True,not_new_tracking=True,**cv)
  cases.append(case);save(folder/'RESULT.json',case)
  save(REPORTS/'FROZEN_TRAIN16_REFERENCES.json',dict(status='IN_PROGRESS' if len(cases)<5 else 'COMPLETE_EXACT_PRIOR_INPUTS',cases=cases,no_weights_or_labels_changed=True,DEV_history_not_mixed_with_TRAIN=True))
  print('EXACT_REFERENCE',arm,r['raw_online_metrics']['HOTA'],cv['CVIDF1'],flush=True)
if __name__=='__main__':main()
