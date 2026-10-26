#!/usr/bin/env python3
"""Fixed TRAIN4/8/12 x61 real RGB-to-ID timing after all formal scores finish."""
import argparse,json,time,os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from worldmot.runtime.world_object_v5_contract import require_v5_scope,v5_source_fingerprint,atomic_json,load_v5_membership_gate
from worldmot.tracking.world_object_v5_config import registered_config
from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
from worldmot.models.gmt_detector_adapter import GMTDetectorAdapter,GMT_PYTHON
from worldmot.models.omega_geometry_backend import OmegaGeometryBackend
from worldmot.models.world_object_v4_checkpoint import load_frozen_r2_head
from worldmot.runtime.world_object_v5_core import ObservationAnchoredWorldObjectCore
from worldmot.runtime.online_world_object_v5 import OnlineObservationAnchoredTracker
from worldmot.runtime.world_object_tracking_core import WorldObjectTrackingCore
from worldmot.runtime.online_world_object_tracker import OnlineWorldObjectTracker
from worldmot.tracking.world_object_v4_variants import state_configs
from worldmot.data.visiontrack import VisionTrack


def main():
    p=argparse.ArgumentParser()
    for key in ["scope-lock","authorization","calibration","output-root","metadata-root"]:p.add_argument("--"+key,required=True,type=Path)
    p.add_argument("--device",default="cuda:0");a=p.parse_args();torch.set_num_threads(1)
    scope,_=require_v5_scope(a.scope_lock,a.authorization,"live","train","B3");source=v5_source_fingerprint()
    for v in ["B0","B1","B2","B3","B4"]:
        metric=json.loads((a.output_root/f"evaluation/{v}/metrics.json").read_text())
        if metric["status"]!="COMPLETE_VALID_SCOPE":raise ValueError("FINISH_ALL_REGISTERED_NATIVE_SCORES_BEFORE_LIVE")
    folder=a.output_root/"live_fixed_train_prefix";folder.mkdir(exist_ok=False)
    payload=json.loads((a.metadata_root/"annotations/train.json").read_text())
    if payload.get("annotations"):raise ValueError("PHYSICAL_NO_GT_METADATA_REQUIRED_FOR_LIVE")
    ds=VisionTrack(a.metadata_root,split="train",load_annotations=False);raw=ImmutableGMTDetectionCache(scope["splits"]["train"]["raw_cache"])
    cal=json.loads(a.calibration.read_text());results={};worker_start=time.perf_counter()
    detector=GMTDetectorAdapter(device=a.device,python=GMT_PYTHON)
    try:
        detector_manifest=detector.manifest()
        for key in ["checkpoint_sha256","config_sha256","threshold","precision"]:
            if detector_manifest[key]!=raw.lock[key]:raise ValueError("REAL_DETECTOR_CONTRACT_CHANGED")
        geometry=OmegaGeometryBackend('/data1/liuyeqiang/WWW_omega_runtime_20261016/official_vggt_omega',
            '/data1/liuyeqiang/WWW_omega_runtime_20261016/weights/vggt_omega_1b_512.pt',device=a.device)
        if geometry.status!="READY":raise RuntimeError("VERIFIED_OMEGA512_UNAVAILABLE: "+geometry.reason)
        for variant in ["B0","B1","B2","B3","B4"]:
            cfg=None if variant=="B0" else registered_config(variant,cal)
            if variant in {"B0","B1"}:
                gate=Path(scope["splits"]["train"]["support_cache"]).parent/"train/FULL_WORLD_OBJECT_V3/train_gate.json"
                model,weight=load_frozen_r2_head(gate,raw.lock,a.device)
            else:
                g=scope["final_weights"][variant];model,weight=load_v5_membership_gate(g["train_gate"],cfg,a.device,g["checkpoint_sha256"])
            if variant=="B0":tracker=OnlineWorldObjectTracker(detector,geometry,state_configs(1024)["C1"],core=WorldObjectTrackingCore(state_configs(1024)["C1"],model,a.device))
            else:tracker=OnlineObservationAnchoredTracker(detector,geometry,ObservationAnchoredWorldObjectCore(cfg,model,a.device))
            tracker.reset(4)
            for pair in ds.synchronized_pairs(4,limit=5):
                images={v:np.asarray(Image.open(ds.image_path(r)).convert('RGB')) for v,r in pair.images.items()};tracker.step(images,pair.frame_id)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();rows=[];calls=0;variant_started=time.perf_counter()
            variant_dir=folder/variant;variant_dir.mkdir()
            for scene in [4,8,12]:
                tracker.reset(scene);count=0
                with (variant_dir/f'scene_{scene:02d}.jsonl').open('x') as stream:
                    for pair in ds.synchronized_pairs(scene,limit=61):
                        started=time.perf_counter();read_started=time.perf_counter()
                        images={v:np.asarray(Image.open(ds.image_path(r)).convert('RGB')) for v,r in pair.images.items()};read_s=time.perf_counter()-read_started
                        result=tracker.step(images,pair.frame_id);torch.cuda.synchronize();t=time.perf_counter()
                        stream.write(json.dumps(result,allow_nan=False,separators=(',',':'))+'\n');stream.flush();serialized=time.perf_counter()-t
                        stage=dict(tracker.last_timings)
                        if variant=='B0':stage={k if k.endswith('_s') else k+'_s':value for k,value in stage.items()}
                        rows.append({'RGB_read_s':read_s,'final_output_s':serialized,'RGB_to_final_ID_s':time.perf_counter()-started,**stage})
                        calls+=int(variant!='B4');count+=1
                if count!=61:raise ValueError("FIXED_LIVE_PREFIX_COVERAGE")
            keys=sorted(set().union(*(r.keys() for r in rows)))
            timing={k:{'P50_s':float(np.percentile([r.get(k,0.) for r in rows],50)),'P95_s':float(np.percentile([r.get(k,0.) for r in rows],95)),'sum_s':float(sum(r.get(k,0.) for r in rows))} for k in keys}
            elapsed=sum(r['RGB_to_final_ID_s'] for r in rows);omega=sum(r.get('omega_model_only_s',0.) for r in rows)
            results[variant]={'status':'COMPLETE_FINITE_REAL_LIVE','paired_times':len(rows),'TRAIN_scenes':[4,8,12],'first_paired_times_per_scene':61,
                'LIVE_RGB_to_final_ID_FPS':len(rows)/elapsed,'variant_wall_FPS_excluding_shared_initialization':len(rows)/(time.perf_counter()-variant_started),
                'Omega_calls':calls,'Omega_model_only_calls_per_s':calls/omega if omega else None,'timings':timing,'checkpoint':weight,
                'GPU_parent_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'GPU_memory_scope':'parent Ω and head allocator only, excludes separate GMT worker',
                'warmup_paired_times':5,'tracking_state_reset_after_warmup':True,'full_video_TEST_FPS':False,'model_training':False,'GT_used':False}
            atomic_json(variant_dir/'runtime.json',results[variant]);print(json.dumps({'live_variant_complete':variant,'FPS':results[variant]['LIVE_RGB_to_final_ID_FPS']}),flush=True)
            del tracker,model
        if v5_source_fingerprint()!=source:raise RuntimeError("SOURCE_CHANGED_DURING_REAL_LIVE")
        atomic_json(folder/'runtime.json',{'status':'ALL5_FINITE_REAL_LIVE_COMPLETE','variants':results,'v5_source_sha256':source,'device_name':torch.cuda.get_device_name(),
            'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'detector':detector_manifest,'Omega':geometry.diagnostics(),
            'shared_initialization_plus_all_variants_wall_s':time.perf_counter()-worker_start,'TEST_read':False,'GT_used':False,'single_worker_not_six_GPU_throughput':True})
    finally:detector.close()


if __name__=='__main__':main()
