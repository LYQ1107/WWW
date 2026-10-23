#!/usr/bin/env python3
"""Metadata-only RGB -> delivered IDs; complete causal independent videos."""
from __future__ import annotations
import argparse,collections,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from worldmot.data.visiontrack import VisionTrack
from worldmot.models.omega_geometry_backend import OmegaGeometryBackend
from worldmot.models.person_detector import PersonDetector
from worldmot.models.person_detector_factory import make_person_detector
from worldmot.models.persistent_world_queries import PersistentWorldQueryTransformer
from worldmot.tracking.persistent_query_runtime import PersistentQueryRuntime
from worldmot.runtime.detection_first import DetectionFirstWorldMOT
from worldmot.runtime.online_worldmot_v2 import OnlineWorldMOTV2
from cache_fulltrain import DEFAULT_DETECTOR,DEFAULT_WEIGHTS,DEFAULT_OMEGA,DEFAULT_OMEGA_WEIGHTS,atomic_json,sha
from experiment_provenance import source_stamp
from preserve_inference_attempts import prepare_scene

def stats(values):
    a=np.asarray(values,np.float64)
    return {'count':len(a),'total_s':float(a.sum()),'p50_s':float(np.quantile(a,.5)) if len(a) else None,'p95_s':float(np.quantile(a,.95)) if len(a) else None,'mean_s':float(a.mean()) if len(a) else None}

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-root',default='/data/DATASETS/TRACKING/JDE/VisionTrack');p.add_argument('--split',choices=['train','test'],required=True)
    p.add_argument('--metadata',required=True);p.add_argument('--method',choices=['v1','v2'],required=True);p.add_argument('--checkpoint');p.add_argument('--lock');p.add_argument('--output',required=True)
    p.add_argument('--scenes',nargs='+',type=int);p.add_argument('--device',default='cuda:0');p.add_argument('--limit',type=int)
    p.add_argument('--detector',choices=['rtmdet','gmt_pre_gta'],default='rtmdet');p.add_argument('--gmt-raw-cache')
    p.add_argument('--gmt-detector-mode',choices=['replay','live_verified'],default='live_verified')
    p.add_argument('--resource-mode',choices=['unspecified','exclusive_gpu','shared_gpu'],default='unspecified')
    p.add_argument('--v1-output-boxes',choices=['projection','current_detector']);a=p.parse_args()
    torch.set_num_threads(2)
    import cv2;cv2.setNumThreads(1)
    if a.split=='test' and a.limit:raise ValueError('independent TEST requires complete sequences')
    metadata=json.loads(Path(a.metadata).read_text())
    if metadata.get('annotations'):raise RuntimeError('inference metadata must not contain annotations')
    if a.split=='test' and a.detector=='gmt_pre_gta':
        if not a.lock:raise RuntimeError('GMT detector route TEST requires a TRAIN-only freeze lock')
        frontend_lock=json.loads(Path(a.lock).read_text())
        if frontend_lock.get('TEST_used_for_selection',True):raise RuntimeError('TEST selection is forbidden')
        if frontend_lock.get('source_provenance')!=source_stamp():raise RuntimeError('GMT route source changed after freeze')
    ds=VisionTrack(a.dataset_root,annotation=a.metadata,split=a.split,load_annotations=False)
    if a.method=='v2':
        saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True);manifest=saved['manifest']
        if a.split=='test':
            if not a.lock:raise RuntimeError('TEST requires TRAIN-selected final lock')
            lock=json.loads(Path(a.lock).read_text())
            if sha(a.checkpoint) not in lock.get('checkpoint_sha256',[]):raise RuntimeError('checkpoint not in pre-TEST freeze lock')
            if lock.get('source_provenance')!=source_stamp():raise RuntimeError('source changed after pre-TEST freeze')
            if lock.get('TEST_metadata_sha256')!=sha(a.metadata):raise RuntimeError('TEST metadata differs from freeze lock')
            if not manifest.get('full_train_coverage') or manifest.get('status')!='FULLTRAIN_COMPLETE':raise RuntimeError('pilot/subset checkpoint forbidden for TEST')
        model=PersistentWorldQueryTransformer(**manifest['model_config'])
        query=PersistentQueryRuntime(model,device=a.device,checkpoint=a.checkpoint)
    detector=make_person_detector(a.detector,a.device,DEFAULT_DETECTOR,DEFAULT_WEIGHTS,a.method=='v2',a.gmt_raw_cache,a.gmt_detector_mode)
    geometry=OmegaGeometryBackend(DEFAULT_OMEGA,DEFAULT_OMEGA_WEIGHTS,device=a.device,expose_intermediate_features=a.method=='v2')
    if detector.status!='READY' or geometry.status!='READY':raise RuntimeError('perception backend unavailable')
    pipeline=OnlineWorldMOTV2(detector,geometry,query) if a.method=='v2' else DetectionFirstWorldMOT(detector,geometry,device=a.device)
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    signature={'split':a.split,'method':a.method,'metadata_sha256':sha(a.metadata),
               'checkpoint_sha256':sha(a.checkpoint) if a.checkpoint else None,
               'source_sha256':source_stamp()['sha256'],'detector_frontend':a.detector,
               'detector_mode':a.gmt_detector_mode if a.detector=='gmt_pre_gta' else 'live',
               'resource_mode':a.resource_mode,
               'v1_output_boxes':a.v1_output_boxes or ('current_detector' if a.detector=='gmt_pre_gta' else 'projection')}
    for sid in a.scenes or ds.scene_ids():
        out=root/f'scene_{sid:02d}'
        if not prepare_scene(out,signature):continue
        pipeline.reset(sid);pairs=list(ds.synchronized_pairs(sid,limit=a.limit));times=[];reads=[];stages={};count=0;boxes=0;ids=set();id_frame_counts=collections.Counter()
        atomic_json(out/'manifest.json',{'status':'RUNNING',**signature,'scene':sid,'expected_frames':len(pairs),'GT_in_inference':False})
        scene_started=time.perf_counter();last_delivery=scene_started
        with (out/'tracks.jsonl').open('w') as f,(out/'timings.jsonl').open('w') as tf:
            for pair in pairs:
                started=time.perf_counter();images={v:np.asarray(Image.open(ds.image_path(rec)).convert('RGB')) for v,rec in pair.images.items()};read=time.perf_counter()-started
                if hasattr(detector,'set_context'):detector.set_context(sid,pair.frame_id,{v:rec.image_id for v,rec in pair.images.items()})
                row=pipeline.step(images,pair.frame_id);row['image_ids']={str(v):rec.image_id for v,rec in pair.images.items()}
                if a.method=='v1' and signature['v1_output_boxes']=='current_detector':
                    row['tracks_2d_projection_by_camera']=row['tracks_2d_by_camera']
                    row['tracks_2d_by_camera']=row['tracks_2d_observation_boxes_by_camera']
                    row['output_policy']='CURRENT_SHARED_DETECTION_BOX_WITH_ACTUAL_V1_ID'
                # Full dense support and repeated manifests are unnecessary for
                # official 2D scoring; preserve detection/query/geometry audits.
                row.pop('tracks_3d',None);row.pop('detector_status',None);row.pop('omega_status',None)
                row['detections_by_camera']={str(view):[{k:d.get(k) for k in ('detection_id','bbox_xywh','score','mask_area','source')}
                    for d in dets] for view,dets in row.get('detections_by_camera',{}).items()}
                frame_stages=dict(row.get('runtime', {}).get('frame_stage_s', row.get('runtime', {}).get('frame_stages', {})))
                for src,dst in [('omega_forward_s','omega_forward'),('detector_forward_s','detector_forward')]:
                    if src in row.get('runtime',{}):frame_stages[dst]=float(row['runtime'][src])
                for name, value in frame_stages.items():
                    stages.setdefault(name, []).append(float(value))
                if a.method=='v1':row['world_observations']=[{k:o.get(k) for k in ('observation_id','camera_ids','detection_ids','center','covariance','confidence','geometry_reliable','fusion_status','rejection_reason','reprojection_coverage','geometry_support_sources')} for o in row.get('world_observations',[])]
                raw=json.dumps(row,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x))+'\n';f.write(raw);f.flush()
                last_delivery=time.perf_counter();total=last_delivery-started;times.append(total);reads.append(read);count+=1
                tf.write(json.dumps({'timestamp':pair.frame_id,'rgb_read_s':read,'rgb_to_delivered_ids_s':total,'stages_s':frame_stages,'queue_s':0.,'synchronous':True})+'\n');tf.flush()
                frame_ids=set()
                for values in row['tracks_2d_by_camera'].values():
                    for obj in values:
                        if obj.get('evaluation_eligible',True) and obj.get('visible',True):boxes+=1;ids.add(obj['global_id']);frame_ids.add(obj['global_id'])
                id_frame_counts.update(frame_ids)
                if count%128==0:
                    atomic_json(out/'progress.json',{'frames':count,'expected':len(pairs),'boxes':boxes,'global_ids':len(ids),'e2e_fps':count/sum(times)})
                    print(json.dumps({'scene':sid,'frames':count,'expected':len(pairs),'boxes':boxes,'global_ids':len(ids),'e2e_fps':count/sum(times)}),flush=True)
        report={'status':'COMPLETE','split':a.split,'scene':sid,'method':a.method,'frames':count,'expected_frames':len(pairs),'scene_total_frames':len(ds.frames(sid)),
                'full_video':count==len(ds.frames(sid)),'GT_in_inference':False,'metadata_sha256':sha(a.metadata),'checkpoint_sha256':sha(a.checkpoint) if a.checkpoint else None,
                'final_lock_sha256':sha(a.lock) if a.lock else None,'trained_scope':manifest.get('training_scope') if a.method=='v2' else 'RULE_HUNGARIAN_FROZEN_BACKBONES',
                'real_e2e_fps':len(times)/(last_delivery-scene_started),'real_e2e_seconds':last_delivery-scene_started,
                'e2e':stats(times),'rgb_read':stats(reads),'stages':{k:stats(v) for k,v in stages.items()},'prediction_sha256':sha(out/'tracks.jsonl'),
                'global_ids':len(ids),'delivered_boxes':boxes,'fps_unit':'synchronized dual-camera timestamps/second','timing_scope':'RGB decode + detector + Omega + all CPU/postprocessing + query/memory + serialization + flushed final-ID delivery; excludes initialization and offline evaluation',
                'max_ID_distinct_timestamps':max(id_frame_counts.values(),default=0),
                'IDs_with_at_least_two_timestamps':sum(n>=2 for n in id_frame_counts.values()),
                'device':a.device,'visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
                'gpu':torch.cuda.get_device_name(a.device),'detector':detector.manifest(),'omega':geometry.diagnostics(),'perception_cache_used':False}
        report.update(signature)
        if a.detector=='gmt_pre_gta' and a.gmt_detector_mode=='replay':
            report['association_replay_FPS']=report['real_e2e_fps'];report['association_replay_seconds']=report['real_e2e_seconds']
            report['real_e2e_fps']=None;report['real_e2e_seconds']=None;report['perception_cache_used']=True
            report['timing_scope']='RGB decode + cached GMT observations + real Omega + association/projection/output; association replay, not system E2E'
        atomic_json(out/'manifest.json',report);print(json.dumps({k:report[k] for k in ['scene','frames','global_ids','real_e2e_fps','full_video']}),flush=True)
    if hasattr(detector,'close'):detector.close()
if __name__=='__main__':main()
