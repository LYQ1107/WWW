#!/usr/bin/env python3
"""Evaluate G2 joint world-object formation on all TRAIN scenes."""

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
from worldmot.models.world_object_support import WorldObjectSupport, support_compatibility


def iou(a, b):
    ax, ay, aw, ah = map(float, a); bx, by, bw, bh = map(float, b)
    inter=max(0.,min(ax+aw,bx+bw)-max(ax,bx))*max(0.,min(ay+ah,by+bh)-max(ay,by)); union=aw*ah+bw*bh-inter
    return inter/union if union>0 else 0.


def match(rows, gt):
    pairs=sorted([(iou(r["bbox"],a.get("bbox",[0]*4)),i,j) for i,r in enumerate(rows) for j,a in enumerate(gt)],reverse=True); used_i=set();used_j=set();out={}
    for s,i,j in pairs:
        if s<.5 or i in used_i or j in used_j: continue
        used_i.add(i);used_j.add(j);out[int(rows[i]["detection_id"])] = (a:=gt[j],float(s))
    return out


def as_support(r):
    fg=np.asarray(r["foreground"]); fg=fg[np.isfinite(fg).all(1)]; box=np.asarray(r["box"]); box=box[np.isfinite(box).all(1)]
    return WorldObjectSupport(box,fg,np.asarray(r["center"]),np.asarray(r["cov"]),float(r["quality"]),float(r["foreground_fraction"]),"CACHE",0.,1.)


def main():
    p=argparse.ArgumentParser();p.add_argument("--cache",required=True);p.add_argument("--dataset-root",default="/data/DATASETS/TRACKING/JDE/VisionTrack");p.add_argument("--output-dir",required=True)
    a=p.parse_args();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);ds=VisionTrack(a.dataset_root,split="train")
    per_scene={}; curve=[]; total=Counter(); rows=[]
    for cache in iter_complete_scenes(a.cache):
        frames={int(x.frame_id):x for x in ds.synchronized_pairs(cache.scene)}; c=Counter()
        for fr in cache.frames():
            pair=frames[fr["timestamp"]]; gt={int(v):ds.ground_truth(rec.image_id) for v,rec in pair.images.items()}
            byview={v:[r for r in fr["rows"] if r["view"]==v] for v in (1,2)}; mm={v:match(byview[v],gt[v]) for v in byview}
            obs=[]
            for v in sorted(byview):
                for r in byview[v]:
                    class O: pass
                    o=O();o.camera_id=v;o.detection_id=r["detection_id"];o.support=as_support(r);o.feature=np.asarray([r["score"]],np.float32);o.bbox_xywh=r["bbox"];obs.append(o)
            hyps,diag=JointWorldObjectBuilder().build(obs);c["frames"]+=1;c["hypotheses"]+=len(hyps);c["fused"]+=sum(len(h.observations)==2 for h in hyps);c["single"]+=sum(len(h.observations)==1 for h in hyps)
            for h in hyps:
                ids=[]
                for o in h.observations:
                    ann=mm.get(o.camera_id,{}).get(o.detection_id)
                    if ann: ids.append(int(ann[0].get("instance_id",ann[0].get("track_id",ann[0].get("id",-1)))))
                unique=set(ids); c["correct_fused"] += int(len(h.observations)==2 and len(unique)==1 and len(unique)>0); c["wrong_fused"] += int(len(h.observations)==2 and len(unique)>1)
            # Candidate scores for a fixed false-merge-rate comparison.
            left,right=byview[1],byview[2]
            for l in left:
                for r in right:
                    score,detail,reason=support_compatibility(as_support(l),as_support(r),max_normalized_distance=10.)
                    il=mm[1].get(l["detection_id"]);ir=mm[2].get(r["detection_id"]); same=bool(il and ir and il[0].get("instance_id")==ir[0].get("instance_id"))
                    curve.append({"score":float(score) if np.isfinite(score) else -1.,"same":int(same),"scene":cache.scene,"timestamp":fr["timestamp"]})
        per_scene[str(cache.scene)]=dict(c); total.update(c)
    # Sweep only TRAIN-derived thresholds; no TEST selection.
    for threshold in np.linspace(0.0,1.0,21):
        accepted=[x for x in curve if x["score"]>=threshold]; pos=sum(x["same"] for x in accepted); neg=sum(1-x["same"] for x in accepted); total_pos=sum(x["same"] for x in curve); total_neg=sum(1-x["same"] for x in curve)
        rows.append({"threshold":float(threshold),"accepted":len(accepted),"same_person_recall":pos/max(1,total_pos),"different_person_false_merge_rate":neg/max(1,total_neg)})
    with (out/"G2_false_merge_vs_correct_fusion.csv").open("w",newline="") as f:
        fields=list(rows[0]);w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    (out/"G2_multiview_object_all24.csv").write_text(json.dumps({"schema":"worldobject_g2_formation_v1","per_scene":per_scene,"summary":dict(total),"candidate_pairs":len(curve),"gt_role":"ORACLE_GT_MATCH_DIAGNOSTIC_ONLY","selection":"thresholds_swept_on_TRAIN_only"},indent=2))
    print(json.dumps({"scenes":len(per_scene),"frames":total["frames"],"fused":total["fused"],"correct_fused":total["correct_fused"],"wrong_fused":total["wrong_fused"],"candidate_pairs":len(curve)},indent=2))


if __name__=="__main__":main()
