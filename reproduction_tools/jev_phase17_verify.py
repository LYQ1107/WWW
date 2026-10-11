"""Read-only experimental provenance and failed-gate verification, no simulated PASS."""
from jev_phase17_common import *
from jev_master_goal_controller import validate_evidence
import argparse,collections

def main():
 protect();validate_evidence(read(MASTER/'EVIDENCE_REGISTRY.json')['entries'])
 for e in read(MASTER/'EXPERIMENT_REGISTRY.json')['experiments']:validate_evidence(e['evidence'])
 checked=[];digest_cache={}
 def walk(value):
  if isinstance(value,dict):
   if set(['path','SHA256'])<=set(value):
    path=value['path'];assert Path(path).is_file(),path
    if path not in digest_cache:digest_cache[path]=sha(path)
    assert digest_cache[path]==value['SHA256'],path;checked.append(path)
   for v in value.values():walk(v)
  elif isinstance(value,list):
   for v in value:walk(v)
 # Every actual run output, raw predictions, trace and captured native state is checked.
 for namespace in ['factorial','soft']:
  for path in (OUT/namespace).rglob('RESULT.json'):walk(read(path))
 for path in [OUT/'teacher_inference_v2/RESULT.json',OUT/'teacher_data/PAIRS.json',OUT/'owner_audit/TEACHER_EVENTS.json']:walk(read(path))
 for model in read(OUT/'teacher_models/MANIFEST.json'):
  for item in model['files']:
   if any(s in Path(item['path']).parts for s in ['seed1','seed2']):continue
   walk(item)
 f=read(REPORTS/'WHO_Q4_FACTORIAL.json');assert f['actual_metrics']['B0']['IDSW']==147 and f['actual_metrics']['B3']['IDSW']==1494;assert f['interaction_B3_minus_B2_minus_B1_plus_B0']['IDSW']==1044
 p=read(REPORTS/'PREFIX_WHO_Q4_FACTORIAL.json');assert p['original24_prefix_IDs_exact']==24 and p['hard24_prefix_IDs_exact']==24
 for name,count in [('factorial_queue_v1',28),('soft_queue_v1',10)]:
  q=read(OUT/name/'RESULT.json');assert q['status']=='COMPLETE' and not q['failed'] and len(q['done'])==count
 g=read(REPORTS/'FINAL_GO_NO_GO.json');assert g['scientific_GO'] is False and g['data_access']['sealed_20_21_22'] and not g['data_access']['official_TEST'] and not g['data_access']['Full24']
 assert read(REPORTS/'TRAINING_RESULTS.json')['updates']==0
 assert not read(REPORTS/'LEARNING_ELIGIBILITY.json')['qualified']
 assert read(MASTER/'MASTER_GOAL.json')['master_status']=='PIVOT_REQUIRED'
 print('PHASE17_VERIFICATION_PASS',len(set(checked)),'unique artifact hashes; failed scientific/cache gates preserved',flush=True)
if __name__=='__main__':main()
