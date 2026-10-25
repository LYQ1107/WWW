#!/usr/bin/env python3
"""Render actual delivered IDs, RGB boxes and sparse Ω supports after scoring."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from drive_worldobject_v3 import FULL,NG,NOBIRTH,ROI


def frame_row(path,timestamp):
    with path.open() as stream:
        for line in stream:
            row=json.loads(line)
            if row["timestamp"]==timestamp:return row
    raise ValueError("MISSING_DELIVERED_TIMESTAMP")


def main():
    p=argparse.ArgumentParser();p.add_argument("--run-root",required=True,type=Path);p.add_argument("--results",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path);p.add_argument("--cases",type=int,default=6);a=p.parse_args()
    report_path=a.results/"worldobject_v3_identity_error_attribution.json"
    report=json.loads(report_path.read_text())
    if report.get("fixed_candidate_boxes_verified") is not True:raise ValueError("POSTHOC_FIXED_CANDIDATE_AUDIT_REQUIRED")
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from worldmot.data.visiontrack import VisionTrack
    from worldmot.data.worldobject_cache import WorldObjectCache
    ds=VisionTrack(a.run_root/"metadata_only_dataset",split="test",load_annotations=False)
    cases=[];seen=set()
    for kind in ("FIRST_CLEAN_BIRTH_SPLIT","DIFFERENT_PERSON_MERGE","SAME_PERSON_SPLIT"):
        count_for_kind=0
        for case in report["failure_cases"]:
            key=(case["scene"],case["timestamp"],tuple((x["camera"],x["detection_id"]) for x in case["actors"]))
            if case["GT_scope_legal"] and case["error"]==kind and key not in seen:
                cases.append(case);seen.add(key)
                count_for_kind+=1
                if len(cases)>=a.cases or count_for_kind>=max(1,a.cases//3):break
        if len(cases)>=a.cases:break
    a.output.mkdir(parents=True,exist_ok=True);outputs=[]
    methods=(FULL,NOBIRTH,NG,ROI);names={FULL:"Full V3",NOBIRTH:"Frozen V3 / no learned new-birth",NG:"Full no-geometry",ROI:"Legacy / restored ROI"}
    for number,case in enumerate(cases):
        sid,timestamp=case["scene"],case["timestamp"]
        pair=next(x for x in ds.synchronized_pairs(sid,start_frame=timestamp,limit=1) if x.frame_id==timestamp)
        images={v:np.asarray(Image.open(ds.image_path(rec)).convert("RGB")) for v,rec in pair.images.items()}
        cache=WorldObjectCache(a.run_root/"cache_test_v2",sid,require_features=False,require_geometry=True)
        frame=next(x for x in cache.frames() if x["timestamp"]==timestamp)
        support={(r["view"],r["detection_id"]):r for r in frame["rows"]}
        delivered={m:frame_row(a.run_root/"predictions"/m/f"scene_{sid:02d}/tracks.jsonl",timestamp) for m in methods}
        figure=plt.figure(figsize=(14,4*len(methods)))
        for row_index,method in enumerate(methods):
            by_key={(int(v),x["detection_id"]):x for v,items in delivered[method]["tracks_2d_by_camera"].items() for x in items}
            for column,view in enumerate((1,2)):
                axis=figure.add_subplot(len(methods),3,3*row_index+column+1)
                axis.imshow(images[view]);selected=[x for x in case["actors"] if x["camera"]==view]
                limits=[]
                for actor in selected:
                    pred=by_key[view,actor["detection_id"]];x,y,w,h=pred["bbox_xywh"];color=plt.cm.tab20(pred["global_id"]%20)
                    axis.add_patch(Rectangle((x,y),w,h,fill=False,color=color,linewidth=2.5))
                    axis.text(x,y-5,f'ID {pred["global_id"]} / GT {actor["gt_identity"]}',color="white",fontsize=10,
                              bbox={"facecolor":color,"alpha":.9,"edgecolor":"none"})
                    limits.append((x,y,x+w,y+h))
                if limits:
                    bounds=np.asarray(limits);lo=bounds[:,:2].min(0);hi=bounds[:,2:].max(0);pad=.35*np.maximum(hi-lo,40)
                    axis.set_xlim(max(0,lo[0]-pad[0]),min(images[view].shape[1],hi[0]+pad[0]))
                    axis.set_ylim(min(images[view].shape[0],hi[1]+pad[1]),max(0,lo[1]-pad[1]))
                axis.set_title(f'{names[method]} / Camera {view}');axis.set_axis_off()
            axis=figure.add_subplot(len(methods),3,3*row_index+3,projection="3d")
            if method==NG:
                axis.text2D(.05,.5,"Geometry disabled in this method",transform=axis.transAxes);axis.set_axis_off()
            else:
                found=False
                for actor in case["actors"]:
                    key=(actor["camera"],actor["detection_id"]);record=support[key]
                    points=np.asarray(record["foreground"],float).reshape(-1,3)
                    points=points[np.isfinite(points).all(1)]
                    if len(points)==0:continue
                    gid=by_key[key]["global_id"];found=True
                    axis.scatter(points[:,0],points[:,1],points[:,2],s=9,color=plt.cm.tab20(gid%20),
                                 marker="o" if key[0]==1 else "^",label=f'Cam {key[0]} / ID {gid}')
                axis.set_title("Actual sparse world supports (up to 64/box)")
                axis.set_xlabel("World X");axis.set_ylabel("World Y");axis.set_zlabel("World Z")
                if found:axis.legend(fontsize=7)
                else:axis.text2D(.05,.5,"No valid persistent-world support",transform=axis.transAxes)
        figure.suptitle(f'Scene {sid}, synchronized time {timestamp}: {case["error"]} (example from {case["method"]})',fontsize=11)
        figure.text(.02,.01,"Display crops only; models consumed full RGB. GT labels are posthoc. Colors denote each method's own IDs. World scale is arbitrary; no mesh is claimed.",fontsize=8)
        figure.tight_layout(rect=(0,.025,1,.97));path=a.output/f"case_{number:02d}_scene{sid:02d}_t{timestamp:05d}.png"
        figure.savefig(path,dpi=140);plt.close(figure)
        outputs.append({"image":str(path),"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"source_case":case})
    (a.output/"manifest.json").write_text(json.dumps({"source_audit_sha256":hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "selection":"Earliest distinct legal failure examples, ordered by registered error type; illustrations, not aggregate quality evidence.",
        "model_inputs_cropped":False,"display_images_cropped":True,"examples":outputs},indent=2))
    print(json.dumps({"rendered_cases":len(outputs),"output":str(a.output)}))


if __name__=="__main__":main()
