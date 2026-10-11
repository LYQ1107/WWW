"""Prepare unchanged raw GT/detections; never use the duplicate-ID override."""
from jev_phase19_common import *
def prepare(folder,video,predictions):
 allowed(video);out=folder/'strict_eval'
 import run_early_pilot_tracking as pilot
 d=read(ANNOTATIONS);ims=[i for i in d['images'] if i['video_id']==video];ids={i['id'] for i in ims}
 subset=dict(d,images=ims,annotations=[a for a in d['annotations'] if a['image_id'] in ids],videos=[v for v in d['videos'] if v['id']==video])
 pilot.PILOT=out;dataset=pilot.prepare_eval_dataset(subset)
 prepared=out/'tracking_eval_runtime_state/native/prepared';evaluated=out/'tracking_eval_runtime_state/native/evaluation'
 subprocess.run([PYTHON,str(ROOT/'reproduction_tools/prepare_visiontrack_predictions.py'),'--predictions',str(predictions),'--dataset',str(dataset),'--split','train','--output',str(prepared)],cwd=ROOT,check=True)
 manifest=read(prepared/'manifest.json');assert manifest['status']=='PASS'
 duplicates=manifest.get('gt_duplicate_rows',{})
 if any(duplicates.values()):
  return dict(status='NOT_EVALUABLE_DUPLICATE_GT',prepared=ref(prepared/'manifest.json'),duplicates=duplicates,metrics=None,permissive_override=False,raw_GT_unchanged=True)
 subprocess.run([PYTHON,str(ROOT/'reproduction_tools/evaluate_visiontrack.py'),'--prepared',str(prepared),'--output',str(evaluated)],cwd=ROOT,check=True)
 return dict(status='STRICT_COMPLETE',prepared=ref(prepared/'manifest.json'),metrics=pilot.extract_metrics(evaluated),raw_metrics=ref(evaluated/'metrics.json'),permissive_override=False,official_TEST=False)

