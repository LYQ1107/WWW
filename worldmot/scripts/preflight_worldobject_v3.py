#!/usr/bin/env python3
"""Bounded, authorized TRAIN-only RGB/Ω/ROI/identity-gradient engineering check."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from worldmot.runtime.experiment_authorization import require_experiment_authorization,source_commit,source_fingerprint


def main():
    p=argparse.ArgumentParser()
    for name in ("authorization","dataset-root","raw-cache","legacy-cache","omega-source","omega-weights","output"):
        p.add_argument("--"+name,required=True)
    p.add_argument("--device",default="cuda:0");p.add_argument("--normal-frames",type=int,default=130)
    p.add_argument("--other-frames",type=int,default=16)
    a=p.parse_args();approval=require_experiment_authorization(a.authorization,"preflight","train","FULL_WORLD_OBJECT_V3")
    import torch
    from PIL import Image
    from worldmot.data.visiontrack import VisionTrack
    from worldmot.data.worldobject_cache import WorldObjectCache
    from worldmot.data.worldobject_observations import replay_observations
    from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
    from worldmot.models.omega_geometry_backend import OmegaGeometryBackend
    from worldmot.models.world_object_update_head import WorldObjectUpdateHead
    from worldmot.runtime.world_object_tracking_core import WorldObjectTrackingCore
    from worldmot.training.world_object_causal_training import CausalWorldObjectTraining
    from worldmot.training.world_object_identity_targets import match_train_identities
    from worldmot.tracking.world_object_config import WorldObjectConfig
    from cache_worldobject_support import SceneState
    torch.set_num_threads(2)
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:pass
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    ds=VisionTrack(a.dataset_root,split="train",load_annotations=True)
    raw=ImmutableGMTDetectionCache(a.raw_cache);config=WorldObjectConfig(raw.feature_dim)
    omega=OmegaGeometryBackend(a.omega_source,a.omega_weights,device=a.device)
    if omega.status!="READY":raise RuntimeError(omega.reason)
    torch.manual_seed(20261025);model=WorldObjectUpdateHead(raw.feature_dim,128).to(a.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4)
    trainer=CausalWorldObjectTraining(WorldObjectTrackingCore(config,model,a.device))
    totals=Counter();scenes=[];gradients=Counter();started=time.perf_counter()
    with (out/"frames.jsonl").open("x") as log:
        for scene in (1,3,8,24):
            state=SceneState(omega,raw,scene);trainer.reset(scene)
            live_core=WorldObjectTrackingCore(config);live_core.reset(scene)
            replay_core=WorldObjectTrackingCore(config);replay_core.reset(scene)
            legacy=iter(WorldObjectCache(a.legacy_cache,scene,raw_cache=raw,require_geometry=False).frames())
            count=0;summary=Counter();limit=a.normal_frames if scene==1 else a.other_frames
            for pair in ds.synchronized_pairs(scene,limit=limit):
                t=time.perf_counter()
                images={v:np.asarray(Image.open(ds.image_path(record)).convert("RGB")) for v,record in pair.images.items()}
                old=next(legacy)
                if old["timestamp"]!=pair.frame_id:raise ValueError("LEGACY_RAW_TIMESTAMP_JOIN_MISMATCH")
                rows,frame,_,alignment,_,aligned,runtime=state.step(images,pair.frame_id)
                observations=replay_observations({"timestamp":pair.frame_id,"rows":rows})
                # Independent paths consume exactly equal frozen observations.
                left=live_core.step(observations,pair.frame_id);right=replay_core.step(replay_observations({"timestamp":pair.frame_id,"rows":rows}),pair.frame_id)
                if left["tracks_2d_by_camera"]!=right["tracks_2d_by_camera"]:raise ValueError("LIVE_REPLAY_ID_MISMATCH")
                labels,label_audit=match_train_identities(observations,ds.ground_truth_pair(pair))
                _,losses=trainer.step(observations,pair.frame_id,labels,optimizer)
                for key,value in trainer.last_gradient_diagnostics.items():gradients[key]=max(gradients[key],float(value))
                for index,view in enumerate(sorted(frame.pointmaps)):
                    camera_depth=frame.cameras[view].forward_depth(frame.pointmaps[view])
                    original=np.asarray(omega.last_predictions["depth"])[index].squeeze()
                    error=float(np.nanmax(np.abs(camera_depth-original)))
                    if not np.isfinite(error) or error>1e-3:raise ValueError("CAMERA_FORWARD_DEPTH_INCONSISTENT")
                if count in (63,64,127,128):
                    summary["clip_boundaries_checked"]+=1
                    if not trainer.core.bank.active():raise ValueError("EMPTY_BANK_AT_CLIP_BOUNDARY")
                summary.update(label_audit);summary.update(trainer.last_diagnostics)
                summary["timestamps"]+=1;summary["observations"]+=len(observations);summary["aligned_timestamps"]+=int(aligned)
                summary["fusion_eligible"]+=sum(o.cross_view_fusion_eligible for o in observations)
                summary["weak_geometry"]+=sum(o.geometry_valid and not o.cross_view_fusion_eligible for o in observations)
                log.write(json.dumps({"scene":scene,"timestamp":pair.frame_id,"observations":len(observations),
                    "aligned":aligned,"alignment_reason":alignment.result.reason,"losses":losses,
                    "gradient_norms":trainer.last_gradient_diagnostics,"diagnostics":trainer.last_diagnostics,
                    "wall_s":time.perf_counter()-t,"runtime":{k:v for k,v in runtime.items() if k.endswith("_s")}},allow_nan=False)+"\n")
                log.flush();count+=1
                if count%16==0:print(json.dumps({"scene":scene,"preflight_frames":count,"loss":losses["total"]}),flush=True)
            scenes.append({"scene":scene,**dict(summary)});totals.update(summary)
    if gradients["assignment_head"]<=0 or gradients["joint_birth_head"]<=0:raise ValueError("PREFLIGHT_HEAD_GRADIENT_MISSING")
    report={"schema":"worldobject_v3_birthfix_preflight_v1","status":"REAL_PREFLIGHT_PASS","approval":approval,
        "source_commit":source_commit(),"source_sha256":source_fingerprint(),"split":"train","scenes":scenes,
        "totals":dict(totals),"gradient_max_norms":dict(gradients),"roi_feature_dim":raw.feature_dim,
        "omega":omega.diagnostics(),"detector":raw.manifest(),"elapsed_s":time.perf_counter()-started,
        "gpu_peak_bytes":torch.cuda.max_memory_allocated(a.device),"live_replay_equal":True,
        "identity_model_parameters":sum(parameter.numel() for parameter in model.parameters()),
        "frozen_omega_parameters":sum(parameter.numel() for parameter in omega.model.parameters()),
        "full_training_started":False,"test_used":False,"accuracy_use":"ENGINEERING_ONLY_NO_FORMAL_MOT_SCORE"}
    (out/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({"status":report["status"],"frames":totals["timestamps"],"elapsed_s":report["elapsed_s"]}),flush=True)


if __name__=="__main__":
    try:main()
    except PermissionError:raise
    except Exception as exc:
        import sys,traceback
        if "--output" in sys.argv:
            path=Path(sys.argv[sys.argv.index("--output")+1]);path.mkdir(parents=True,exist_ok=True)
            (path/"report.json").write_text(json.dumps({"status":"REAL_PREFLIGHT_BLOCKED",
                "error":str(exc),"traceback":traceback.format_exc(),"source_commit":source_commit(),
                "source_sha256":source_fingerprint(),"full_tasks_started":False},indent=2))
        raise
