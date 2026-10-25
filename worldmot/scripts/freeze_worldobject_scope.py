#!/usr/bin/env python3
"""Prepare V3 source/cache locks after approval; no model loading."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from worldmot.runtime.experiment_authorization import require_experiment_authorization,source_commit,source_fingerprint
from worldmot.tracking.world_object_config import METHODS,WorldObjectConfig
from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
from worldmot.models.world_object_checkpoint import detector_contract
from worldmot.models.omega_geometry_backend import OFFICIAL_OMEGA_CHECKPOINT_SHA256
from worldmot.data.worldobject_cache import WorldObjectCache,sha256
from worldmot.data.worldobject_scope import SCOPE_SCHEMA


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authorization");p.add_argument("--cache-train",required=True);p.add_argument("--cache-test")
    p.add_argument("--raw-cache-train",required=True);p.add_argument("--raw-cache-test")
    p.add_argument("--method-id",choices=METHODS,default="FULL_WORLD_OBJECT_V3");p.add_argument("--output",required=True)
    a=p.parse_args();approval=require_experiment_authorization(a.authorization,"freeze",method_id=a.method_id)
    train_raw=ImmutableGMTDetectionCache(a.raw_cache_train)
    config=WorldObjectConfig(train_raw.feature_dim,a.method_id)
    lock={"schema":SCOPE_SCHEMA,"source_commit":source_commit(),"source_sha256":source_fingerprint(),"method_id":a.method_id,
          "tracking_config":config.to_dict(),"detector_contract":detector_contract(train_raw.lock),
          "omega_sha256":OFFICIAL_OMEGA_CHECKPOINT_SHA256 if config.use_geometry else None,
          "approval":approval,"splits":{}}
    for split,root,raw_root,total_scenes,total_timestamps in (
        ("train",a.cache_train,a.raw_cache_train,24,28754),("test",a.cache_test,a.raw_cache_test,22,29019)):
        if root is None:
            continue
        if raw_root is None:
            raise ValueError("SPLIT_RAW_CACHE_REQUIRED")
        raw=ImmutableGMTDetectionCache(raw_root)
        if raw.lock.get("split")!=split or detector_contract(raw.lock)!=lock["detector_contract"]:
            raise ValueError("FROZEN_DETECTOR_SPLIT_OR_CONTRACT_MISMATCH")
        scenes={}
        for scene in range(1,total_scenes+1):
            cache=WorldObjectCache(root,scene,raw_cache=raw,require_features=config.use_appearance,
                                   require_geometry=config.use_geometry)
            if config.use_geometry and cache.manifest.get("omega",{}).get("checkpoint_sha256")!=lock["omega_sha256"]:
                raise ValueError("OMEGA_CHECKPOINT_LOCK_MISMATCH")
            scenes[str(scene)]={"manifest_sha256":sha256(cache.path/"manifest.json"),
                                "timestamps":cache.manifest["timestamps"]}
        count=sum(r["timestamps"] for r in scenes.values())
        if count!=total_timestamps:
            raise ValueError("FROZEN_TIMESTAMP_SCOPE_MISMATCH")
        lock["splits"][split]={"scene_count":len(scenes),"timestamps":count,"scenes":scenes}
    with Path(a.output).open("x") as stream:
        json.dump(lock,stream,indent=2,allow_nan=False)


if __name__=="__main__":
    main()
