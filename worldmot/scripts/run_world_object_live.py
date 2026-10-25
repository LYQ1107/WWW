#!/usr/bin/env python3
"""V3 RGB entry. Requires approval; cache-detector timing is explicitly labelled."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from worldmot.runtime.experiment_authorization import require_experiment_authorization,source_commit
from worldmot.tracking.world_object_config import METHODS,METHOD_ALIASES,WorldObjectConfig
from worldmot.data.worldobject_cache import sha256
from worldmot.runtime.legacy_worldobject_tracking_core import LEGACY_ROI_METHOD


def run_scene(ds,scene,tracker,out_scene,max_timestamps=None):
    from PIL import Image
    out_scene=Path(out_scene);out_scene.mkdir(parents=True,exist_ok=False)
    tracker.reset(scene)
    expected=len(ds.frames(scene));count=0
    with (out_scene/"tracks.jsonl").open("x") as output,(out_scene/"timings.jsonl").open("x") as timing:
        for pair in ds.synchronized_pairs(scene):
            if max_timestamps is not None and count>=max_timestamps:
                break
            started=time.perf_counter();t=time.perf_counter()
            images={v:np.asarray(Image.open(ds.image_path(rec)).convert("RGB")) for v,rec in pair.images.items()}
            rgb_s=time.perf_counter()-t
            if hasattr(tracker.detector,"set_context"):
                tracker.detector.set_context(scene,pair.frame_id,{v:r.image_id for v,r in pair.images.items()})
            result=tracker.step(images,pair.frame_id)
            output_started=time.perf_counter()
            output.write(json.dumps(result,allow_nan=False)+"\n")
            output.flush()
            output_s=time.perf_counter()-output_started
            timing.write(json.dumps({"timestamp":pair.frame_id,"rgb_read_s":rgb_s,
                "output_s":output_s,"live_rgb_to_id_s":time.perf_counter()-started,"stages":tracker.last_timings,
                "detector_stages":getattr(tracker.detector,"last_timings",{})})+"\n")
            count+=1
    cache_detector=hasattr(tracker.detector,"set_context")
    manifest={"schema":"worldobject_live_manifest_v3","method_id":getattr(tracker.core,"method_id",tracker.config.method_id),
        "source_commit":source_commit(),"scene":scene,"timestamps":count,"expected_timestamps":expected,
        "status":"COMPLETE" if count==expected else "PARTIAL_TIMING_SAMPLE",
        "GT_in_inference":False,"causal":True,"detector_is_frozen_replay":cache_detector,
        "fps_definition":"RGB_CACHE_DETECTOR_OMEGA_TO_ID" if cache_detector else "LIVE_RGB_GMT_OMEGA_TO_ID",
        "tracking_config":tracker.config.to_dict(),"tracks_sha256":sha256(out_scene/"tracks.jsonl"),
        "detector":tracker.detector.manifest(),
        "omega":tracker.geometry.diagnostics() if tracker.geometry is not None else {"status":"NOT_USED"}}
    (out_scene/"manifest.json").write_text(json.dumps(manifest,indent=2))
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authorization");p.add_argument("--dataset-root",required=True)
    p.add_argument("--raw-cache",required=True,help="feature dimension and frozen detector contract")
    p.add_argument("--detector-backend",choices=["gmt","cache"],default="gmt")
    p.add_argument("--output",required=True);p.add_argument("--scenes",nargs="+",type=int)
    p.add_argument("--split",choices=["train","test"],default="test")
    p.add_argument("--method-id",choices=list(METHODS)+[LEGACY_ROI_METHOD],default="FULL_WORLD_OBJECT_V3")
    p.add_argument("--checkpoint-manifest");p.add_argument("--rule-based",action="store_true")
    p.add_argument("--scope-lock",required=True)
    p.add_argument("--device",default="cuda:0");p.add_argument("--max-timestamps",type=int)
    p.add_argument("--omega-source");p.add_argument("--omega-weights")
    p.add_argument("--gmt-python",default="/home/liuyeqiang/anaconda3/envs/GMT/bin/python")
    a=p.parse_args()
    a.method_id=METHOD_ALIASES.get(a.method_id,a.method_id)
    approval=require_experiment_authorization(a.authorization,"live",a.split,a.method_id)
    from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
    from worldmot.models.world_object_checkpoint import load_identity_checkpoint,detector_contract
    from worldmot.data.visiontrack import VisionTrack
    from worldmot.runtime.world_object_tracking_core import WorldObjectTrackingCore
    from worldmot.runtime.online_world_object_tracker import OnlineWorldObjectTracker
    raw=ImmutableGMTDetectionCache(a.raw_cache)
    if raw.lock["split"]!=a.split:
        raise ValueError("DETECTOR_SPLIT_MISMATCH")
    legacy=a.method_id==LEGACY_ROI_METHOD
    config=WorldObjectConfig(raw.feature_dim,"FULL_WORLD_OBJECT_V3" if legacy else a.method_id)
    from worldmot.data.worldobject_scope import load_scope_lock
    scope=load_scope_lock(a.scope_lock,a.split,config.method_id)
    if scope["detector_contract"]!=detector_contract(raw.lock):
        raise ValueError("LIVE_SCOPE_DETECTOR_MISMATCH")
    model=None
    if a.checkpoint_manifest:
        model,gate=load_identity_checkpoint(a.checkpoint_manifest,config,a.device)
        if gate["detector_contract"]!=detector_contract(raw.lock):
            raise ValueError("TRAIN_TEST_DETECTOR_CONTRACT_MISMATCH")
    elif not a.rule_based:
        raise ValueError("V3_FULL_TRAIN_CHECKPOINT_REQUIRED")
    detector=raw
    if a.detector_backend=="gmt":
        from worldmot.models.gmt_detector_adapter import GMTDetectorAdapter
        detector=GMTDetectorAdapter(device=a.device,python=a.gmt_python)
        manifest=detector.manifest()
        if (manifest.get("roi_feature_dim")!=raw.feature_dim
                or any(manifest.get(k)!=raw.lock[k] for k in ("checkpoint_sha256","config_sha256","threshold","precision"))):
            raise ValueError("LIVE_GMT_FROZEN_DETECTOR_CONTRACT_MISMATCH")
    geometry=None
    if config.use_geometry:
        from worldmot.models.omega_geometry_backend import OmegaGeometryBackend
        if not a.omega_source or not a.omega_weights:
            raise ValueError("VERIFIED_OMEGA_SOURCE_AND_CHECKPOINT_REQUIRED")
        geometry=OmegaGeometryBackend(a.omega_source,a.omega_weights,device=a.device)
        if geometry.status!="READY":
            raise RuntimeError(geometry.reason)
    core=WorldObjectTrackingCore(config,model,a.device)
    if legacy:
        from worldmot.runtime.legacy_worldobject_tracking_core import LegacyWorldObjectTrackingCore
        core=LegacyWorldObjectTrackingCore(config)
    tracker=OnlineWorldObjectTracker(detector,geometry,config,core=core,device=a.device)
    ds=VisionTrack(a.dataset_root,split=a.split,load_annotations=False)
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    reports=[run_scene(ds,sid,tracker,root/f"scene_{sid:02d}",a.max_timestamps) for sid in a.scenes or ds.scene_ids()]
    (root/"coverage.json").write_text(json.dumps({"schema":"worldobject_live_coverage_v3","approval":approval,
        "scenes":reports,"all_complete":all(r["status"]=="COMPLETE" for r in reports)},indent=2))


if __name__=="__main__":
    main()
