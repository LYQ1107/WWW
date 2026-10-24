#!/usr/bin/env python3
"""TRAIN replay for persistent WorldObjectBank (predicted vs oracle history)."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from worldmot.data.visiontrack import VisionTrack
from worldmot.data.worldobject_cache import iter_complete_scenes
from worldmot.models.multi_view_world_object import JointWorldObjectBuilder
from worldmot.models.world_object_support import WorldObjectSupport
from worldmot.tracking.persistent_world_object_bank import PersistentWorldObjectBank


def iou(a,b):
    ax,ay,aw,ah=map(float,a);bx,by,bw,bh=map(float,b);inter=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by));u=aw*ah+bw*bh-inter;return inter/u if u>0 else 0


def match(rows,gt):
    pairs=sorted([(iou(r["bbox"],a.get("bbox",[0]*4)),r,a) for r in rows for a in gt],key=lambda x:x[0],reverse=True);used=set();out={}
    for s,r,a in pairs:
        ident=int(a.get("instance_id",a.get("track_id",a.get("id",-1))))
        if s<.5 or ident in used:continue
        used.add(ident);out[int(r["detection_id"])] = ident
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument("--cache",required=True);p.add_argument("--dataset-root",default="/data/DATASETS/TRACKING/JDE/VisionTrack");p.add_argument("--output-dir",required=True)
    a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);ds=VisionTrack(a.dataset_root,split="train");scene_rows=[];failures=[]
    for cache in iter_complete_scenes(a.cache):
        pairs={int(x.frame_id):x for x in ds.synchronized_pairs(cache.scene)};bank=PersistentWorldObjectBank(min_hits=1);c=Counter();last_gid={};oracle_last={};
        for fr in cache.frames():
            pair=pairs[fr["timestamp"]];gt={int(v):ds.ground_truth(rec.image_id) for v,rec in pair.images.items()};byview={v:[r for r in fr["rows"] if r["view"]==v] for v in (1,2)};mm={v:match(byview[v],gt[v]) for v in byview}
            obs=[]
            for v in sorted(byview):
                for r in byview[v]:
                    class O:pass
                    o=O();o.camera_id=v;o.detection_id=r["detection_id"];o.bbox_xywh=r["bbox"];o.support=WorldObjectSupport(np.asarray(r["box"]),np.asarray(r["foreground"]),np.asarray(r["center"]),np.asarray(r["cov"]),float(r["quality"]),float(r["foreground_fraction"]),"CACHE",0.,1.);o.feature=np.asarray([r["score"]],np.float32);obs.append(o)
            hyps,_=JointWorldObjectBuilder().build(obs);objects,bd=bank.step(hyps,fr["timestamp"]);c.update({"frames":1,"hypotheses":len(hyps),"births":bd["births"],"matched":bd["matched"]})
            for j,h in enumerate(hyps):
                gid=bank.last_assignments.get(j);ids=[]
                for o in h.observations:
                    x=mm.get(o.camera_id,{}).get(o.detection_id)
                    if x is not None:ids.append(x)
                if len(set(ids))>1:c["wrong_world_object_merge"]+=1;failures.append({"scene":cache.scene,"timestamp":fr["timestamp"],"type":"WRONG_MERGE","ids":sorted(set(ids))})
                if len(set(ids))==1 and gid is not None:
                    ident=ids[0];c["identity_observations"]+=1;c["identity_gid_switch"]+=int(ident in last_gid and last_gid[ident]!=gid);last_gid[ident]=gid
                    oracle_gid=ident; c["oracle_observations"]+=1; c["oracle_gid_switch"]+=int(ident in oracle_last and oracle_last[ident]!=oracle_gid);oracle_last[ident]=oracle_gid
            # Current detections are always represented by a hypothesis or a
            # recorded abstention; this makes retention auditable rather than
            # silently dropping them in the bank.
            c["detected_rows"] += len(fr["rows"]); c["hypothesis_observations"] += sum(len(h.observations) for h in hyps)
        scene_rows.append({"scene":cache.scene,**dict(c),"predicted_switch_rate":c["identity_gid_switch"]/max(1,c["identity_observations"]),"oracle_switch_rate":c["oracle_gid_switch"]/max(1,c["oracle_observations"]),"wrong_merge_rate":c["wrong_world_object_merge"]/max(1,c["frames"] )})
    with (out/"G3_oracle_vs_predicted_all24.csv").open("w",newline="") as f:
        fields=list(scene_rows[0]) if scene_rows else ["scene"];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(scene_rows)
    (out/"G3_temporal_identity_failures.json").write_text(json.dumps({"schema":"worldobject_g3_identity_v1","scenes":scene_rows,"failures":failures[:100000],"history_policy":{"oracle":"identity-key diagnostic only","predicted":"PersistentWorldObjectBank autonomous causal replay"}},indent=2))
    print(json.dumps({"scenes":len(scene_rows),"failures":len(failures),"predicted_switches":sum(x.get("identity_gid_switch",0) for x in scene_rows),"wrong_merges":sum(x.get("wrong_world_object_merge",0) for x in scene_rows)},indent=2))


if __name__=="__main__":main()
