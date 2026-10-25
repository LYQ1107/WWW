#!/usr/bin/env python3
"""Approved causal replay through the same V3 core as live RGB inference."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from worldmot.runtime.experiment_authorization import require_experiment_authorization,source_commit
from worldmot.tracking.world_object_config import WorldObjectConfig,METHODS
from worldmot.runtime.world_object_tracking_core import WorldObjectTrackingCore
from worldmot.data.worldobject_observations import replay_observations
from worldmot.data.worldobject_cache import WorldObjectCache,sha256


def run_scene(cache,out_scene,core):
    out_scene=Path(out_scene)
    out_scene.mkdir(parents=True,exist_ok=False)
    core.reset(cache.scene)
    count=0
    with (out_scene/"tracks.jsonl").open("x") as output,(out_scene/"timings.jsonl").open("x") as timing:
        iterator=iter(cache.frames())
        while True:
            started=time.perf_counter()
            try:
                frame=next(iterator)
            except StopIteration:
                break
            observations=replay_observations(frame)
            result=core.step(observations,frame["timestamp"])
            output.write(json.dumps(result,allow_nan=False)+"\n")
            output.flush()
            timing.write(json.dumps({"timestamp":frame["timestamp"],"cache_replay_s":time.perf_counter()-started})+"\n")
            count+=1
    complete=count==cache.manifest["timestamps"]
    manifest={"schema":"worldobject_prediction_manifest_v3","method_id":core.config.method_id,
        "source_commit":source_commit(),"scene":cache.scene,"timestamps":count,
        "expected_timestamps":cache.manifest["timestamps"],"status":"COMPLETE" if complete else "PARTIAL",
        "GT_in_inference":False,"causal":True,"fps_definition":"CACHE_REPLAY_INCLUDING_IO_NO_RGB_NO_OMEGA",
        "tracks_sha256":sha256(out_scene/"tracks.jsonl"),"tracking_config":core.config.to_dict(),
        "cache_manifest_sha256":sha256(cache.path/"manifest.json")}
    (out_scene/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False))
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authorization");p.add_argument("--cache",required=True);p.add_argument("--raw-cache",required=True)
    p.add_argument("--output",required=True);p.add_argument("--scenes",nargs="+",type=int)
    p.add_argument("--scope-lock",required=True)
    p.add_argument("--split",choices=["train","test"],default="test")
    p.add_argument("--method-id",choices=METHODS,default="FULL_WORLD_OBJECT_V3")
    p.add_argument("--checkpoint-manifest");p.add_argument("--rule-based",action="store_true")
    p.add_argument("--device",default="cpu")
    a=p.parse_args()
    approval=require_experiment_authorization(a.authorization,"replay",a.split,a.method_id)
    from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
    from worldmot.models.world_object_checkpoint import load_identity_checkpoint,detector_contract
    raw=ImmutableGMTDetectionCache(a.raw_cache)
    if raw.lock["split"]!=a.split:
        raise ValueError("DETECTOR_SPLIT_MISMATCH")
    config=WorldObjectConfig(raw.feature_dim,a.method_id)
    from worldmot.data.worldobject_scope import load_scope_lock,validate_scene_lock
    scope=load_scope_lock(a.scope_lock,a.split,a.method_id)
    model=None;gate=None
    if a.checkpoint_manifest:
        model,gate=load_identity_checkpoint(a.checkpoint_manifest,config,a.device)
        if gate["detector_contract"]!=detector_contract(raw.lock):
            raise ValueError("TRAIN_TEST_DETECTOR_CONTRACT_MISMATCH")
    elif not a.rule_based:
        raise ValueError("V3_FULL_TRAIN_CHECKPOINT_REQUIRED: --rule-based identifies the untrained baseline")
    core=WorldObjectTrackingCore(config,model,a.device)
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    scenes=a.scenes or [int(p.name.split("_")[1]) for p in sorted(Path(a.cache).glob("scene_*"))]
    reports=[]
    for sid in scenes:
        cache=WorldObjectCache(a.cache,sid,raw_cache=raw,require_features=config.use_appearance,require_geometry=config.use_geometry)
        validate_scene_lock(scope,cache,a.split)
        reports.append(run_scene(cache,root/f"scene_{sid:02d}",core))
    full_scope=(len(scenes)==22 and sum(r["timestamps"] for r in reports)==29019) if a.split=="test" else len(scenes)==24
    (root/"coverage.json").write_text(json.dumps({"schema":"worldobject_coverage_v3","approval":approval,
        "split":a.split,"method_id":a.method_id,"model_status":"TRAINED" if model else "RULE_BASED_BASELINE",
        "scenes":reports,"full_split_scope":full_scope,"all_complete":all(r["status"]=="COMPLETE" for r in reports),
        "checkpoint_manifest_sha256":sha256(a.checkpoint_manifest) if gate else None},indent=2))


if __name__=="__main__":
    main()
