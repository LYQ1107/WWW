#!/usr/bin/env python3
"""Causal cache-replay runner for the persistent WorldObject route."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from worldmot.data.worldobject_cache import WorldObjectCache
from worldmot.models.multi_view_world_object import JointWorldObjectBuilder
from worldmot.models.world_object_support import WorldObjectSupport
from worldmot.tracking.persistent_world_object_bank import PersistentWorldObjectBank


def make_support(r):
    box=np.asarray(r["box"]);box=box[np.isfinite(box).all(1)];fg=np.asarray(r["foreground"]);fg=fg[np.isfinite(fg).all(1)]
    return WorldObjectSupport(box,fg,np.asarray(r["center"]),np.asarray(r["cov"]),float(r["quality"]),float(r["foreground_fraction"]),"CACHE",0.,1.)


def run_scene(cache, out_scene):
    out_scene.mkdir(parents=True,exist_ok=True); bank=PersistentWorldObjectBank(min_hits=1); builder=JointWorldObjectBuilder(); rows=[]; timings=[]
    for fr in cache.frames():
        started=time.perf_counter();obs=[]
        for r in fr["rows"]:
            class O:pass
            o=O();o.camera_id=r["view"];o.detection_id=r["detection_id"];o.bbox_xywh=r["bbox"];o.score=r["score"];o.feature=np.asarray([r["score"],r["quality"]],np.float32);o.support=make_support(r);obs.append(o)
        hyps,formation=builder.build(obs);objects,bd=bank.step(hyps,fr["timestamp"])
        by_view={"1":[],"2":[]}
        for i,h in enumerate(hyps):
            gid=bank.last_assignments.get(i)
            if gid is None: continue
            score=max(float(getattr(x,"score",1.0)) for x in h.observations)
            for x in h.observations:
                by_view[str(int(x.camera_id))].append({"global_id":int(gid),"bbox_xywh":[float(v) for v in x.bbox_xywh],"score":score,"visible":True,"evaluation_eligible":True,"visibility_state":"CURRENT_DETECTOR_BOX","visibility_evidence":"current_target_detection_bbox","projected_from_world":False,"status":"CONFIRMED"})
        rows.append({"timestamp":int(fr["timestamp"]),"scene_id":cache.scene,"tracks_2d_by_camera":by_view,"world_objects":[{"global_id":bank.last_assignments.get(i),"cameras":list(h.cameras),"status":h.status,"score":h.score} for i,h in enumerate(hyps)],"formation":formation,"bank":bd,"causal":True,"output_policy":"CURRENT_DETECTOR_BOX_ONLY"})
        timings.append({"timestamp":int(fr["timestamp"]),"cache_replay_s":float(time.perf_counter()-started),"observation_count":len(obs),"object_count":len(hyps)})
    with (out_scene/"tracks.jsonl").open("w") as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False,separators=(",",":"))+"\n")
    with (out_scene/"timings.jsonl").open("w") as f:
        for row in timings:f.write(json.dumps(row,allow_nan=False,separators=(",",":"))+"\n")
    manifest={"schema":"worldobject_prediction_v1","scene":cache.scene,"status":"COMPLETE","timestamps":len(rows),"first_frame":rows[0]["timestamp"] if rows else None,"last_frame":rows[-1]["timestamp"] if rows else None,"causal":True,"GT_in_inference":False,"cache_manifest":cache.manifest,"tracks_sha256":hashlib.sha256((out_scene/"tracks.jsonl").read_bytes()).hexdigest(),"timings_sha256":hashlib.sha256((out_scene/"timings.jsonl").read_bytes()).hexdigest(),"fps_definition":"cache_replay_only_not_live_E2E"}
    (out_scene/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False));return manifest


def main():
    p=argparse.ArgumentParser();p.add_argument("--cache",required=True);p.add_argument("--output",required=True);p.add_argument("--scenes",nargs="*",type=int)
    a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True);scenes=a.scenes or [int(x.name.split("_")[1]) for x in sorted(Path(a.cache).glob("scene_*")) if x.is_dir()]
    reports=[]
    for sid in scenes: reports.append(run_scene(WorldObjectCache(a.cache,sid),root/f"scene_{sid:02d}"))
    (root/"coverage.json").write_text(json.dumps({"schema":"worldobject_test_coverage_v1","scenes":reports,"scene_count":len(reports),"all_complete":all(x["status"]=="COMPLETE" for x in reports)},indent=2,allow_nan=False));print(json.dumps({"scenes":len(reports),"complete":all(x["status"]=="COMPLETE" for x in reports)},indent=2))


if __name__=="__main__":main()
