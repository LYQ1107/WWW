#!/usr/bin/env python3
"""Explicit E0/E1 legacy controls on common V2 inputs, never a V3 method."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import numpy as np
from worldmot.runtime.experiment_authorization import require_experiment_authorization,source_commit


def control_hypotheses(observations,feature_mode):
    from worldmot.models.legacy_worldobject_builder_v1 import LegacyPairThenMergeBuilder
    from worldmot.models.world_object_observation import fuse_identity_features
    hypotheses,diagnostic=LegacyPairThenMergeBuilder().build(observations)
    for hypothesis in hypotheses:
        if feature_mode=="restored":
            descriptor,_=fuse_identity_features(hypothesis.observations)
            if descriptor is None:
                raise ValueError("FEATURE_UNAVAILABLE_IN_RESTORED_CONTROL")
            hypothesis.feature=descriptor
        elif feature_mode=="legacy_disabled":
            # Intentionally reproduce the known zero-visual legacy path.
            # The route ID and manifest explicitly report FEATURE_OFF.
            hypothesis.feature=np.zeros(1,np.float32)
        else:
            raise ValueError("UNKNOWN_LEGACY_FEATURE_CONTROL")
    return hypotheses,diagnostic


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authorization");p.add_argument("--cache",required=True);p.add_argument("--raw-cache",required=True)
    p.add_argument("--shared-scope-lock",required=True);p.add_argument("--output",required=True)
    p.add_argument("--split",choices=["train","test"],required=True)
    p.add_argument("--feature-mode",choices=["restored","legacy_disabled"],required=True)
    p.add_argument("--scenes",nargs="+",type=int)
    a=p.parse_args()
    method="LEGACY_PAIR_THEN_MERGE_ROI_RESTORED_CONTROL" if a.feature_mode=="restored" else "LEGACY_PAIR_THEN_MERGE_FEATURE_OFF_CONTROL"
    require_experiment_authorization(a.authorization,"replay",a.split,method)
    from worldmot.data.worldobject_cache import WorldObjectCache,sha256
    from worldmot.data.worldobject_observations import replay_observations
    from worldmot.data.worldobject_scope import load_scope_lock,validate_scene_lock
    from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
    from worldmot.tracking.legacy_worldobject_bank_v1 import LegacyPersistentWorldObjectBank
    raw=ImmutableGMTDetectionCache(a.raw_cache)
    scope=load_scope_lock(a.shared_scope_lock,a.split,"FULL_WORLD_OBJECT_V3")
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    scenes=a.scenes or sorted(map(int,scope["splits"][a.split]["scenes"]))
    for sid in scenes:
        cache=WorldObjectCache(a.cache,sid,raw_cache=raw,require_geometry=True)
        validate_scene_lock(scope,cache,a.split)
        directory=root/f"scene_{sid:02d}";directory.mkdir()
        bank=LegacyPersistentWorldObjectBank(min_hits=1);count=0
        with (directory/"tracks.jsonl").open("x") as output,(directory/"timings.jsonl").open("x") as timing:
            for frame in cache.frames():
                started=time.perf_counter();observations=replay_observations(frame)
                hypotheses,diagnostic=control_hypotheses(observations,a.feature_mode)
                bank.step(hypotheses,frame["timestamp"])
                by_camera={}
                for index,hypothesis in enumerate(hypotheses):
                    gid=bank.last_assignments[index]
                    for observation in hypothesis.observations:
                        by_camera.setdefault(str(observation.camera_id),[]).append({"global_id":gid,
                            "detection_id":observation.detection_id,"bbox_xywh":observation.bbox_xywh.tolist(),
                            "score":observation.score,"visible":True,"evaluation_eligible":True})
                output.write(json.dumps({"timestamp":frame["timestamp"],"scene_id":sid,"method_id":method,
                    "tracks_2d_by_camera":by_camera,"legacy_formation":diagnostic,"GT_in_inference":False,
                    "invalid_geometry_observations":sum(not o.geometry_valid for o in observations),
                    "identity_feature_mode":a.feature_mode})+"\n")
                output.flush();timing.write(json.dumps({"timestamp":frame["timestamp"],"cache_replay_s":time.perf_counter()-started})+"\n")
                count+=1
        (directory/"manifest.json").write_text(json.dumps({"schema":"worldobject_legacy_control_v3",
            "method_id":method,"status":"COMPLETE" if count==cache.manifest["timestamps"] else "PARTIAL",
            "source_commit":source_commit(),"legacy_source_commit":"190c735957f1079ba3f770c8aa35a341d72ba60a",
            "timestamps":count,"tracks_sha256":sha256(directory/"tracks.jsonl"),"GT_in_inference":False,
            "identity_feature_mode":a.feature_mode,"cache_schema":cache.schema,
            "fps_definition":"LEGACY_CONTROL_CACHE_REPLAY_ONLY"},indent=2))


if __name__=="__main__":
    main()
