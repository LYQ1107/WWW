#!/usr/bin/env python3
"""Metadata-only RGB -> delivered IDs; complete causal independent videos."""
from __future__ import annotations
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from worldmot.data.visiontrack import VisionTrack
from worldmot.models.omega_geometry_backend import OmegaGeometryBackend
from worldmot.models.person_detector import PersonDetector
from worldmot.models.persistent_world_queries import PersistentWorldQueryTransformer
from worldmot.tracking.persistent_query_runtime import PersistentQueryRuntime
from worldmot.runtime.detection_first import DetectionFirstWorldMOT
from worldmot.runtime.online_worldmot_v2 import OnlineWorldMOTV2
from cache_fulltrain import DEFAULT_DETECTOR,DEFAULT_WEIGHTS,DEFAULT_OMEGA,DEFAULT_OMEGA_WEIGHTS,atomic_json,sha

def stats(values):
    a=np.asarray(values,np.float64)
    return {'count':len(a),'total_s':float(a.sum()),'p50_s':float(np.quantile(a,.5)) if len(a) else None,'p95_s':float(np.quantile(a,.95)) if len(a) else None,'mean_s':float(a.mean()) if len(a) else None}

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-root',default='/data/DATASETS/TRACKING/JDE/VisionTrack');p.add_argument('--split',choices=['train','test'],required=True)
    p.add_argument('--metadata',required=True);p.add_argument('--method',choices=['v1','v2'],required=True);p.add_argument('--checkpoint');p.add_argument('--lock');p.add_argument('--output',required=True)
    p.add_argument('--scenes',nargs='+',type=int);p.add_argument('--device',default='cuda:0');p.add_argument('--limit',type=int);a=p.parse_args()
    torch.set_num_threads(2)
    import cv2;cv2.setNumThreads(1)
    if a.split=='test' and a.limit:raise ValueError('independent TEST requires complete sequences')
    metadata=json.loads(Path(a.metadata).read_text())
    if metadata.get('annotations'):raise RuntimeError('inference metadata must not contain annotations')
    ds=VisionTrack(a.dataset_root,annotation=a.metadata,split=a.split,load_annotations=False)
    if a.method=='v2':
        saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True);manifest=saved['manifest']
        if a.split=='test':
            if not a.lock:raise RuntimeError('TEST requires TRAIN-selected final lock')
            lock=json.loads(Path(a.lock).read_text())
            if sha(a.checkpoint) not in lock.get('checkpoint_sha256',[]):raise RuntimeError('checkpoint not in pre-TEST freeze lock')
            if not manifest.get('full_train_coverage') or manifest.get('status')!='FULLTRAIN_COMPLETE':raise RuntimeError('pilot/subset checkpoint forbidden for TEST')
        model=PersistentWorldQueryTransformer(**manifest['model_config'])
        query=PersistentQueryRuntime(model,device=a.device,checkpoint=a.checkpoint)
    detector=PersonDetector(DEFAULT_DETECTOR,DEFAULT_WEIGHTS,a.device,expose_intermediate_features=a.method=='v2')
    geometry=OmegaGeometryBackend(DEFAULT_OMEGA,DEFAULT_OMEGA_WEIGHTS,device=a.device,expose_intermediate_features=a.method=='v2')
    if detector.status!='READY' or geometry.status!='READY':raise RuntimeError('perception backend unavailable')
    pipeline=OnlineWorldMOTV2(detector,geometry,query) if a.method=='v2' else DetectionFirstWorldMOT(detector,geometry,device=a.device)
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    for sid in a.scenes or ds.scene_ids():
        out=root/f'scene_{sid:02d}';out.mkdir(exist_ok=True)
        if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text()).get('status')=='COMPLETE':continue
        pipeline.reset(sid);pairs=list(ds.synchronized_pairs(sid,limit=a.limit));times=[];reads=[];stages={};deliver=[];count=0;boxes=0;ids=set()
        atomic_json(out/'manifest.json',{'status':'RUNNING','split':a.split,'scene':sid,'method':a.method,'expected_frames':len(pairs),'GT_in_inference':False})
        with (out/'tracks.jsonl').open('w') as f,(out/'timings.jsonl').open('w') as tf:
            for pair in pairs:
                started=time.perf_counter();images={v:np.asarray(Image.open(ds.image_path(rec)).convert('RGB')) for v,rec in pair.images.items()};read=time.perf_counter()-started
                row=pipeline.step(images,pair.frame_id);row['image_ids']={str(v):rec.image_id for v,rec in pair.images.items()}
                # Full dense support and repeated manifests are unnecessary for
                # official 2D scoring; preserve detection/query/geometry audits.
                row.pop('tracks_3d',None);row.pop('detector_status',None);row.pop('omega_status',None)
                row.pop('detections_by_camera',None)
                for name, value in row.get('runtime', {}).get('frame_stage_s', row.get('runtime', {}).get('frame_stages', {})).items():
                    stages.setdefault(name, []).append(float(value))
                if a.method=='v1':row['world_observations']=[{k:o.get(k) for k in ('observation_id','camera_ids','detection_ids','center','covariance','confidence','geometry_reliable','fusion_status','rejection_reason','reprojection_coverage')} for o in row.get('world_observations',[])]
                raw=json.dumps(row,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n';f.write(raw);f.flush()
                total=time.perf_counter()-started;times.append(total);reads.append(read);count+=1
                tf.write(json.dumps({'timestamp':pair.frame_id,'rgb_read_s':read,'rgb_to_delivered_ids_s':total,'queue_s':0.,'synchronous':True})+'\n');tf.flush()
                for values in row['tracks_2d_by_camera'].values():
                    for obj in values:
                        if obj.get('evaluation_eligible',True) and obj.get('visible',True):boxes+=1;ids.add(obj['global_id'])
                if count%128==0:
                    atomic_json(out/'progress.json',{'frames':count,'expected':len(pairs),'boxes':boxes,'global_ids':len(ids),'e2e_fps':count/sum(times)})
                    print(json.dumps({'scene':sid,'frames':count,'expected':len(pairs),'boxes':boxes,'global_ids':len(ids),'e2e_fps':count/sum(times)}),flush=True)
        report={'status':'COMPLETE','split':a.split,'scene':sid,'method':a.method,'frames':count,'expected_frames':len(pairs),'scene_total_frames':len(ds.frames(sid)),
                'full_video':count==len(ds.frames(sid)),'GT_in_inference':False,'metadata_sha256':sha(a.metadata),'checkpoint_sha256':sha(a.checkpoint) if a.checkpoint else None,
                'final_lock_sha256':sha(a.lock) if a.lock else None,'trained_scope':manifest.get('training_scope') if a.method=='v2' else 'RULE_HUNGARIAN_FROZEN_BACKBONES',
                'real_e2e_fps':len(times)/sum(times),'e2e':stats(times),'rgb_read':stats(reads),'stages':{k:stats(v) for k,v in stages.items()},'prediction_sha256':sha(out/'tracks.jsonl'),
                'global_ids':len(ids),'delivered_boxes':boxes,'fps_unit':'synchronized dual-camera timestamps/second','timing_scope':'RGB decode + detector + Omega + all CPU/postprocessing + query/memory + serialization + flushed final-ID delivery; excludes initialization and offline evaluation',
                'device':a.device,'gpu':torch.cuda.get_device_name(a.device),'detector':detector.manifest(),'omega':geometry.diagnostics(),'perception_cache_used':False}
        atomic_json(out/'manifest.json',report);print(json.dumps({k:report[k] for k in ['scene','frames','global_ids','real_e2e_fps','full_video']}),flush=True)
if __name__=='__main__':main()
