#!/usr/bin/env python3
"""Complete resumable object-only TRAIN cache; no dense/attention cache."""
from __future__ import annotations
import argparse, collections, copy, hashlib, json, os, time
from pathlib import Path
import numpy as np
from PIL import Image
from worldmot.data.visiontrack import VisionTrack
from worldmot.geometry.relative_geometry_encoding import GeometryNormalization
from worldmot.geometry.world_alignment import Sim3, Sim3Aligner
from worldmot.models.omega_geometry_backend import OmegaGeometryBackend
from worldmot.models.person_detector import PersonDetector
from worldmot.models.world_observation_encoder import MultiViewObservationEncoder
from worldmot.runtime.online_worldmot import OnlineWorldMOT
from worldmot.runtime.online_worldmot_v2 import OnlineWorldMOTV2
from worldmot.training.identity_targets import match_detections_to_gt
SCHEMA = 'worldmot_object_v3_rgb128_detector128_omega256_geometry18_scores2'
DEFAULT_DETECTOR = '/data1/liuyeqiang/WWW_fulltrain/worldmot/assets/mmdetection/rtmdet-ins_tiny_8xb32-300e_coco.py'
DEFAULT_WEIGHTS = '/data1/liuyeqiang/models/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth'
DEFAULT_OMEGA = '/data1/liuyeqiang/WWW_omega_runtime_20261016/official_vggt_omega'
DEFAULT_OMEGA_WEIGHTS = '/data1/liuyeqiang/WWW_omega_runtime_20261016/weights/vggt_omega_1b_512.pt'
INPUT_CONTRACT = 'caller_RGB_to_contiguous_BGR_for_MMDetection'
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def atomic_json(path, obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)); os.replace(tmp,path)
def read_images(ds,pair):
    return {v:np.asarray(Image.open(ds.image_path(rec)).convert('RGB')) for v,rec in pair.images.items()}
class SceneExtractor:
    def __init__(self,detector,omega):
        self.detector,self.omega=detector,omega
        self.encoder=MultiViewObservationEncoder(); self.aligner=Sim3Aligner(); self.previous=None; self.previous_world={}; self.reference=None
    def align_encode(self,images,timestamp):
        started=time.perf_counter(); dets=self.detector.infer(images); frame=self.omega.reconstruct(images,timestamp)
        OnlineWorldMOTV2._set_dynamic_masks(frame,dets); astart=time.perf_counter()
        if self.previous is None: alignment=self.aligner.initialize()
        else:
            old=copy.copy(self.previous); old.pointmaps=self.previous_world
            view=sorted(set(frame.pointmaps)&set(old.pointmaps))[0]; src,dst=OnlineWorldMOT._static_correspondences(old,frame,view)
            alignment=self.aligner.update(src,dst)
        align_s=time.perf_counter()-astart; aligned=alignment.success and frame.status=='OK'
        tokens=self.encoder.encode(dets,images,frame,timestamp,alignment.transform if aligned else None,self.detector.last_feature_maps,frame.omega_patch_features)
        if aligned:
            self.previous=frame
            self.previous_world={v:alignment.transform.apply(pm.reshape(-1,3)).reshape(pm.shape) for v,pm in frame.pointmaps.items()}
            t=alignment.transform; self.reference={'timestamp':timestamp,'scale':t.scale,'rotation':t.rotation.tolist(),'translation':t.translation.tolist()}
        return tokens,dets,{'timestamp':timestamp,'alignment_success':bool(aligned),'alignment_reason':alignment.reason,'alignment_inliers':alignment.inliers,'reference':self.reference,
                           'normalization':{'origin':self.encoder.normalization.origin.tolist(),'scale':self.encoder.normalization.scale} if self.encoder.normalization else None,
                           'runtime':{'feature_pipeline_s':time.perf_counter()-started,'omega_s':frame.diagnostics['model_only_s'],'detector_s':sum(self.detector.last_timings.values()),'alignment_s':align_s}}
    def restore(self,ds,sid,meta):
        n=meta.get('normalization')
        if n:self.encoder.normalization=GeometryNormalization(np.asarray(n['origin'],np.float32),n['scale'])
        r=meta.get('reference')
        if not r:return
        pair=next(ds.synchronized_pairs(sid,start_frame=r['timestamp'],limit=1)); images=read_images(ds,pair)
        dets=self.detector.infer(images); frame=self.omega.reconstruct(images,pair.frame_id); OnlineWorldMOTV2._set_dynamic_masks(frame,dets)
        t=Sim3(r['scale'],np.asarray(r['rotation']),np.asarray(r['translation'])); self.previous=frame
        self.previous_world={v:t.apply(pm.reshape(-1,3)).reshape(pm.shape) for v,pm in frame.pointmaps.items()};self.reference=r
        self.aligner.world_from_current=t;self.aligner.initialized=True

def cache_scene(ds,sid,destination,detector,omega,shard_frames=128,limit=None):
    dest=Path(destination)/f'scene_{sid:02d}'; dest.mkdir(parents=True,exist_ok=True)
    pairs=list(ds.synchronized_pairs(sid));pairs=pairs[:limit] if limit else pairs
    expected=[p.frame_id for p in pairs];ex=SceneExtractor(detector,omega);completed=[];records=[];buffer=[];totals=collections.Counter();start=time.perf_counter()
    for path in sorted(dest.glob('shard_*.json')):
        entry=json.loads(path.read_text());payload=dest/entry['file']
        if entry['schema']!=SCHEMA or entry.get('detector_input_contract')!=INPUT_CONTRACT or not payload.exists() or sha(payload)!=entry['sha256']:raise RuntimeError(f'cache integrity/input contract failure: {path}')
        if entry['frames']!=expected[len(completed):len(completed)+len(entry['frames'])]:raise RuntimeError('noncontiguous cache prefix')
        completed.extend(entry['frames']);totals.update(entry['counts']);records.append(entry)
    if completed and len(completed)<len(expected):ex.restore(ds,sid,records[-1]['last_frame'])
    def flush():
        if not buffer:return
        offs=[0];features=[];geo=[];cameras=[];valid=[];targets=[];boxes=[];xyz=[];cov=[];pure=[];timestamps=[];geom_valid=[];counts=collections.Counter()
        for pair,tokens,tmap,meta,missed in buffer:
            timestamps.append(pair.frame_id);counts['timestamps']+=1;counts['gt_misses']+=missed
            for token in tokens:
                target=tmap[(token.camera_id,token.detection_id)];raw=-1
                if target.identity_key:raw=int(target.identity_key.split(':id')[1]);counts['positive_observations']+=1
                elif target.source=='ambiguous_duplicate_identity_masked':raw=-2;counts['ambiguous_observations']+=1
                else:counts['negative_observations']+=1
                if not np.isfinite(token.feature).all() or not np.isfinite(token.geometry_feature).all():raise RuntimeError('nonfinite source feature; refusing silent cache substitution')
                features.append(token.feature);geo.append(token.geometry_feature)
                cameras.append(token.camera_id);valid.append(token.valid);targets.append(raw);boxes.append(token.bbox)
                xyz.append(token.xyz_world);cov.append(token.covariance);geom_valid.append(token.diagnostics['geometry_coordinate_valid']);pure.append(token.diagnostics.get('pure_3d_bbox_xywh') or [np.nan]*4)
            offs.append(len(features));counts['observations']+=len(tokens);counts['valid_observations']+=sum(t.valid for t in tokens)
        arrays={'frame_offsets':np.asarray(offs,np.int64),'timestamps':np.asarray(timestamps,np.int32),'observations':np.asarray(features,np.float16).reshape(-1,532),
                'geometry':np.asarray(geo,np.float32).reshape(-1,18),'camera_ids':np.asarray(cameras,np.int8),'observation_valid':np.asarray(valid,bool),'identity':np.asarray(targets,np.int32),
                'bbox':np.asarray(boxes,np.float32).reshape(-1,4),'xyz_world':np.asarray(xyz,np.float32).reshape(-1,3),'covariance':np.asarray(cov,np.float32).reshape(-1,3,3),
                'geometry_valid':np.asarray(geom_valid,bool),'pure_projection_bbox':np.asarray(pure,np.float32).reshape(-1,4)}
        if not np.isfinite(arrays['observations']).all():raise RuntimeError('FP16 feature overflow')
        path=dest/f'shard_{len(records):04d}.npz';tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
        os.replace(tmp,path)
        entry={'schema':SCHEMA,'detector_input_contract':INPUT_CONTRACT,'scene':sid,'frames':timestamps,'file':path.name,'sha256':sha(path),'counts':dict(counts),'bytes':path.stat().st_size,
               'last_frame':buffer[-1][3],'per_frame':[r[3] for r in buffer]}
        atomic_json(path.with_suffix('.json'),entry);records.append(entry);totals.update(counts);completed.extend(timestamps);buffer.clear()
        atomic_json(dest/'progress.json',{'scene':sid,'done':len(completed),'expected':len(expected),'counts':dict(totals),'elapsed_s':time.perf_counter()-start})
        print(json.dumps({'scene':sid,'done':len(completed),'expected':len(expected),'observations':totals['observations'],'elapsed_s':round(time.perf_counter()-start,2)}),flush=True)
    for pair in pairs[len(completed):]:
        read_start=time.perf_counter();images=read_images(ds,pair);read_s=time.perf_counter()-read_start
        tokens,dets,meta=ex.align_encode(images,pair.frame_id);meta['runtime']['rgb_read_s']=read_s;meta['image_ids']={str(v):r.image_id for v,r in pair.images.items()}
        gt=ds.ground_truth_pair(pair);tmap=match_detections_to_gt(dets,gt,sid);matched={x.identity_key for x in tmap.values() if x.identity_key}
        gt_unique={f'scene{sid}:id{a["instance_id"]}' for anns in gt.values() for a in anns}
        buffer.append((pair,tokens,tmap,meta,len(gt_unique-matched)))
        if len(buffer)>=shard_frames:flush()
    flush()
    result={'schema':SCHEMA,'scene':sid,'status':'COMPLETE' if len(completed)==len(expected) else 'PARTIAL','timestamps':len(completed),'expected_timestamps':len(expected),
            'shards':[{'file':x['file'],'sha256':x['sha256'],'frames':len(x['frames']),'bytes':x['bytes']} for x in records],
            'counts':dict(totals),'raw_gt_modified':False,'dense_pointcloud_cache':False,'identity_targets_only':True,'detector':detector.manifest(),'omega':omega.diagnostics()}
    atomic_json(dest/'manifest.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset-root',default='/data/DATASETS/TRACKING/JDE/VisionTrack');p.add_argument('--output',required=True)
    p.add_argument('--scenes',nargs='+',type=int,required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--shard-frames',type=int,default=128);p.add_argument('--limit',type=int)
    a=p.parse_args();import torch;torch.set_num_threads(2)
    import cv2;cv2.setNumThreads(1)
    ds=VisionTrack(a.dataset_root,split='train');det=PersonDetector(DEFAULT_DETECTOR,DEFAULT_WEIGHTS,a.device,expose_intermediate_features=True)
    om=OmegaGeometryBackend(DEFAULT_OMEGA,DEFAULT_OMEGA_WEIGHTS,device=a.device,expose_intermediate_features=True)
    if det.status!='READY' or om.status!='READY':raise RuntimeError(f'backend not ready: {det.reason} {om.reason}')
    for sid in a.scenes:cache_scene(ds,sid,a.output,det,om,a.shard_frames,a.limit)
if __name__=='__main__':main()
