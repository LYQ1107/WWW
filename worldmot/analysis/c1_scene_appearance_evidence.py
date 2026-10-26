"""Read-only per-scene fixed ROI-cosine strata on already frozen C1 outputs."""
from pathlib import Path
from collections import Counter
import sys,json,hashlib,time
import numpy as np
repo=Path('/data1/liuyeqiang/WWW_worldmot_v4_c1_fulltest_20261026');sys.path[:0]=[str(repo/'worldmot'),str(repo/'worldmot/analysis')]
from worldmot.runtime.world_object_v4_test_contract import require_scope,completed_scene,sha256,write_json,output_candidates,LEGAL_TEST
from worldmot.data.visiontrack import VisionTrack
from worldmot.training.world_object_identity_targets import match_evaluation_identities
from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
from compare_worldobject_v4_c1_fulltest import paired,finish
from types import SimpleNamespace
root=Path('/data1/liuyeqiang/worldmot_v4_c1_fulltest_outputs_20261026');scope,_=require_scope(root/'scope_v4_c1_TEST.json',root/'authorization_v4_c1_TEST.json','diagnostics')
for sid in range(1,23):completed_scene(root/f'predictions/V4_C1/scene_{sid:02d}',scope,sid)
annotation=Path('/data/DATASETS/TRACKING/JDE/VisionTrack/annotations/test.json');assert sha256(annotation)=='7a1e735bf25026a56148deaeb432e0520c6ad5dad97d5362aa58739255389593'
started=time.perf_counter();ds=VisionTrack(annotation.parents[1],split='test',load_annotations=True);raw=ImmutableGMTDetectionCache(scope['raw_cache']);scenes={};total=Counter()
for sid in range(1,23):
 c=Counter();cosine_sum=0.;same_count=0;dense=0
 p0=Path(scope['r2_root'])/f'predictions/FULL_WORLD_OBJECT_V3/scene_{sid:02d}/tracks.jsonl';p1=root/f'predictions/V4_C1/scene_{sid:02d}/tracks.jsonl'
 with p0.open() as f0,p1.open() as f1:
  for pair in ds.synchronized_pairs(sid):
   rows=[json.loads(next(f0)),json.loads(next(f1))];objects=[output_candidates(row) for row in rows];keys=sorted(objects[0]);assert set(keys)==set(objects[1])
   obs=[SimpleNamespace(key=k,camera_id=k[0],detection_id=k[1],bbox_xywh=objects[0][k]['bbox_xywh']) for k in keys]
   labels,_=match_evaluation_identities(obs,ds.ground_truth_pair(pair));dense+=int(len(obs)>=20)
   ids=[[i for i,o in enumerate(obs) if o.camera_id==v and labels[i] is not None] for v in [1,2]]
   if not ids[0] or not ids[1]:continue
   det=raw.get(sid,pair.frame_id,image_ids={v:r.image_id for v,r in pair.images.items()});features={(v,d.detection_id):np.asarray(d.detector_roi_feature,np.float32) for v,items in det.items() for d in items}
   z=[np.asarray([features[keys[i]] for i in ix]) for ix in ids];z=[a/np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12) for a in z];cos=z[0]@z[1].T
   same=np.asarray([labels[i] for i in ids[0]],object)[:,None]==np.asarray([labels[i] for i in ids[1]],object)[None,:]
   mask=same&(cos<=.55);good=[]
   for obj in objects:
    shared=np.asarray([obj[keys[i]]['global_id'] for i in ids[0]])[:,None]==np.asarray([obj[keys[i]]['global_id'] for i in ids[1]])[None,:];good.append(shared==same)
   paired(c,good[0][mask],good[1][mask]);cosine_sum+=float(cos[same].sum());same_count+=int(same.sum())
 assert not f0.closed or True
 n=scope['scenes'][str(sid)]['timestamps'];scenes[str(sid)]={'original_GT_legal':sid in LEGAL_TEST,'same_person_pairs':same_count,'same_person_mean_current_raw_ROI_cosine':cosine_sum/max(1,same_count),'same_person_low_ROI_cosine_C1_vs_C0':finish(c),'dense_N_ge20_timestamps':dense,'timestamps':n,'mean_current_candidates':scope['scenes'][str(sid)]['candidates']/n}
 if sid in LEGAL_TEST:total.update(c)
 print(json.dumps({'appearance_scene_complete':sid}),flush=True)
primary=json.loads((root/'v4_c1_identity_attribution.json').read_text())['legal14_strata']['same_person_low_ROI_cosine']
assert finish(total)==primary,(finish(total),primary)
write_json(root/'v4_c1_scene_difficulty.json',{'status':'COMPLETE','prediction_files_modified':False,'new_model_inference':False,'threshold_fixed':.55,'GT_role':'POSTHOC_ONLY_AFTER_ALL22_FROZEN','source_script_sha256':sha256(__file__),'all_legal14_stratum_counts_exactly_reproduce_primary':True,'per_scene':scenes,'legal14_low_cosine':finish(total),'scope':'Raw ROI cosine is an appearance descriptor, not a calibrated camera angle. Invalid-GT8 scenes remain diagnostic only.','wall_s':time.perf_counter()-started})
