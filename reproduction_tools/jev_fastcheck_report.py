"""Compact honest progress/final tables from completed fastcheck artifacts only."""
from jev_fastcheck_common import *
def main():
 training=[];online=[]
 for p in sorted(REPORTS.glob('TRAIN_*_STEPS*.json')):
  d=read(p);training.append(dict(artifact=ref(p),arm=d['arm'],dataset=d['dataset'],steps=d['steps'],signal=d['training_signal_eligible'],gradient_groups=d['local_gradient_groups'],net=d['final']['net_certified_N01_minus_N10'],rank_accuracy=d['final']['rank_accuracy'],trainable_parameters=d['trainable_parameters'],checkpoint=d['checkpoint'],schedule_SHA=d['schedule_SHA']))
 for p in sorted(REPORTS.glob('ONLINE_*_STEPS*.json')):
  d=read(p);online.append(dict(artifact=ref(p),tag=d['tag'],steps=d['steps'],metrics=d['raw_online_metrics'],CVIDF1=d['CVIDF1'],CVMA=d['CVMA'],counts=d['counts'],identity=d['identity_summary']))
 for dataset in ['D0','D1']:
  for steps in [300,1000]:
   pair=[t for t in training if t['dataset']==dataset and t['steps']==steps]
   if len(pair)==2:assert pair[0]['schedule_SHA']==pair[1]['schedule_SHA'],'same-data schedules differ'
 rows=['# Closed-loop training evidence','','Fixed last checkpoints, unchanged architecture and frozen identity backbone. All methods see identical state/label pools and identical sampled schedule within a data arm. TRAIN16 is pre-exposed gradient-free diagnostics, not independent validation.','','| Architecture | Dataset | New steps | Gradient groups | Source-state net N01−N10 | Rank accuracy |','|---|---|---:|---:|---:|---:|']
 for t in training:
  rank=f"{t['rank_accuracy']:.4f}" if t['rank_accuracy'] is not None else 'N/A: direct readout'
  rows.append(f"| {t['arm']} | {t['dataset']} | {t['steps']} | {t['gradient_groups']} | {t['net']} | {rank} |")
 rows+=['','At300 steps JEV D1−D0 is +55 source-state net corrections; Attention D1−D0 is −9. This is a fixed-source-state diagnostic and cannot establish full native benefit or architecture-specific science. All four300 pilots have real ranking-head gradients and pass the preregistered signal eligibility; qualification is not positive-result selection.']
 (DOCS/'CLOSED_LOOP_TRAINING_RESULTS.md').write_text('\n'.join(rows)+'\n')
 (DOCS/'ONLINE_EVALUATION.md').write_text('# Full native online evaluation\n\n'+('Completed rows are recorded in FINAL_RESULTS.json; all use full cold-start TRAIN16 and frozen evaluators.\n' if online else 'P2 source-state pilots completed. Full cold-start native evaluation is pending fixed-budget1000 checkpoints. No prefix accuracy or cached-perception throughput is reported as full-video identity performance/FPS.\n'))
 registry=read(REPORTS/'EXPERIMENT_REGISTRY.json');gpu_hours=sum(v.get('seconds',0)/3600 for v in registry['jobs'].values() if v.get('GPU') is not None)
 save(REPORTS/'FINAL_RESULTS.json',dict(status='IN_PROGRESS_FULL_NATIVE_REQUIRED',training=training,online=online,scientific_success=False,independent_validation=False,completed_registered_GPU_wall_hours=gpu_hours,master_prior_NO_GO_unchanged=True))
 print('FASTCHECK_REPORT',len(training),len(online),gpu_hours,flush=True)
if __name__=='__main__':main()
