"""Offline past-anchor persistence for retained learned predictions; no GT actor."""
import collections,gzip
from gmt_jev_common import *
from gmt_jev_past_anchor_audit import audit

def evaluate_temporal(kind):
 assert kind in ['mlp','attention','cic']
 source=read(REPORTS/f'ONLINE_{kind}_learned.json');path=REPORTS/f'LEARNED_PAST_ANCHOR_{kind}.json'
 predictions=source['raw_predictions'];assert sha(predictions['path'])==predictions['SHA256']
 if path.exists():
  result=read(path);assert result['predictions']==predictions
  assert result['annotations']['SHA256']==source['annotations']['SHA256']
  return result
 from jev_phase7_offline import IdentityEvaluator
 e=IdentityEvaluator(16)
 with gzip.open(predictions['path'],'rt') as f:pred=json.load(f)
 aligned=e.align(pred);labels={key:r['gt'] for key,r in aligned.items()}
 duplicates={image_id:{g for g,n in collections.Counter(int(a['instance_id']) for a in gt).items() if n>1} for image_id,gt in e.gt.items()}
 image_by_key={(int(i['frame_id'])-1,int(i['view_id'])-1):image_id for image_id,i in e.images.items()}
 masked=0
 for key,g in labels.items():
  if g in duplicates[image_by_key[key[:2]]]:labels[key]=None;masked+=1
 assert sha(ANNOTATIONS)==source['annotations']['SHA256']
 result=audit(pred,labels,e.images)
 original=read(REPORTS/'PAST_ANCHOR_TEMPORAL_AUDIT.json')['methods']['original']
 result.update(status='COMPLETE_POST_HOC_PAST_ANCHOR_DIAGNOSTIC',kind=kind,predictions=predictions,
  annotations=source['annotations'],duplicate_GT_observations_masked=masked,
  duration_ratio_to_original=result['wrong_person_observed_duration_camera_frames']/original['wrong_person_observed_duration_camera_frames'],
  limitations=['Only strictly earlier-scene first-certified >=3/80% owner anchors; no future owner relabeling.',
   'UNKNOWN/newborn/alias coverage and censoring can change between trajectories; consult all counts and IDSW/Frag.',
   'Camera frames, not seconds. Observed persistence is not an individual-intervention causal propagation effect.',
   'Post-hoc audit, never used to select head, step, threshold or architecture.'],
  no_new_GMT_run=True,no_gradient_updates=True,GT_actor_input=False,scientific_success=False)
 save(path,result);return result
