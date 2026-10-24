#!/usr/bin/env python3
"""Train the small W4 WorldObject update head on complete TRAIN cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from worldmot.data.visiontrack import VisionTrack
from worldmot.data.worldobject_cache import iter_complete_scenes


VAL_SCENES = {4, 8, 12, 16, 20, 24}


def iou(a, b):
    ax, ay, aw, ah = map(float, a); bx, by, bw, bh = map(float, b)
    inter = max(0., min(ax + aw, bx + bw) - max(ax, bx)) * max(0., min(ay + ah, by + bh) - max(ay, by))
    union = max(0., aw) * max(0., ah) + max(0., bw) * max(0., bh) - inter
    return inter / union if union > 0 else 0.


def identity_key(ann):
    value = ann.get("instance_id", ann.get("track_id", ann.get("id")))
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def match(rows, gt):
    pairs = sorted([(iou(r["bbox"], a.get("bbox", [0] * 4)), r, a) for r in rows for a in gt], reverse=True, key=lambda x: x[0])
    used = set(); out = {}
    for score, row, ann in pairs:
        if score < .5: break
        key = identity_key(ann)
        if key is None:
            continue
        if key in used: continue
        used.add(key); out[int(row["detection_id"])] = key
    return out


def feature(row, width, height, no_geometry=False):
    center = np.nan_to_num(np.asarray(row["center"], np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    diag = np.nan_to_num(np.diag(np.asarray(row["cov"], np.float32)), nan=1.0, posinf=1.0, neginf=1.0)
    geo = np.zeros(7, np.float32) if no_geometry else np.concatenate([center, diag, [row["quality"]]]).astype(np.float32)
    b = np.asarray(row["bbox"], np.float32)
    visual = np.asarray([b[0] / max(width, 1), b[1] / max(height, 1), b[2] / max(width, 1), b[3] / max(height, 1),
                         row["score"], row["foreground_fraction"], float(row["view"] == 1), float(row["view"] == 2)], np.float32)
    return geo, visual


def build_samples(cache_root, dataset_root, scenes, no_geometry=False):
    ds = VisionTrack(dataset_root, split="train"); samples=[]; coverage=[]
    for cache in iter_complete_scenes(cache_root, scenes):
        pairs={int(p.frame_id):p for p in ds.synchronized_pairs(cache.scene)}; previous={}; n=0
        ambiguous_masked = 0; usable_annotations = 0; matched_detections = 0
        for fr in cache.frames():
            pair=pairs[fr["timestamp"]]; raw_gt={int(v):ds.ground_truth(rec.image_id) for v,rec in pair.images.items()}; gt={}
            for view, anns in raw_gt.items():
                counts={k: sum(identity_key(a) == k for a in anns) for k in {identity_key(a) for a in anns}}
                ambiguous={k for k, count in counts.items() if k is not None and count > 1}
                ambiguous_masked += sum(1 for a in anns if identity_key(a) in ambiguous or identity_key(a) is None)
                clean=[a for a in anns if identity_key(a) is not None and identity_key(a) not in ambiguous]
                usable_annotations += len(clean); gt[view]=clean
            byview={v:[r for r in fr["rows"] if r["view"]==v] for v in pair.images}
            ids={v:match(byview[v],gt[v]) for v in byview}
            matched_detections += sum(len(x) for x in ids.values())
            current={}
            for v, rows in byview.items():
                w,h=pair.images[v].width,pair.images[v].height
                for row in rows:
                    ident=ids[v].get(row["detection_id"]); geo,visual=feature(row,w,h,no_geometry)
                    mem=previous.get(ident, np.zeros(16,np.float32)) if ident is not None else np.zeros(16,np.float32)
                    samples.append({"geo":geo,"visual":visual,"memory":mem,"gate":float(ident is not None and ident in previous),"birth":float(ident is None or ident not in previous),"scene":cache.scene})
                    if ident is not None: current[ident]=np.pad(np.concatenate([geo,visual]),(0,max(0,16-len(geo)-len(visual))))[:16]
                    n+=1
            previous=current
        coverage.append({"scene":cache.scene,"timestamps":cache.manifest["timestamps"],"observations":n,
                         "usable_gt_annotations": usable_annotations,
                         "ambiguous_or_missing_identity_annotations_masked": ambiguous_masked,
                         "matched_detection_supervision": matched_detections})
    return samples,coverage


def run_epoch(model, optim, samples, device, train=True, batch=512):
    import torch
    rng=np.random.default_rng(0); order=np.arange(len(samples));
    if train: rng.shuffle(order); model.train()
    else: model.eval()
    total=0.; correct=0; count=0
    for start in range(0,len(order),batch):
        chunk=[samples[int(i)] for i in order[start:start+batch]]
        geo=torch.as_tensor(np.stack([x["geo"] for x in chunk]),device=device); visual=torch.as_tensor(np.stack([x["visual"] for x in chunk]),device=device); mem=torch.as_tensor(np.stack([x["memory"] for x in chunk]),device=device)
        gate=torch.as_tensor([x["gate"] for x in chunk],device=device); birth=torch.as_tensor([x["birth"] for x in chunk],device=device)
        with torch.set_grad_enabled(train):
            out=model(geo,visual,mem); loss=torch.nn.functional.binary_cross_entropy(out["update_gate"],gate)+torch.nn.functional.binary_cross_entropy_with_logits(out["birth_logit"],birth)
            if train: optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
        total+=float(loss.detach().cpu())*len(chunk); pred=(out["update_gate"]>.5).float(); correct+=int((pred==gate).sum().item());count+=len(chunk)
    return {"loss":total/max(1,count),"gate_accuracy":correct/max(1,count),"samples":count}


def main():
    p=argparse.ArgumentParser();p.add_argument("--cache",required=True);p.add_argument("--dataset-root",default="/data/DATASETS/TRACKING/JDE/VisionTrack");p.add_argument("--output",required=True);p.add_argument("--epochs",type=int,default=4);p.add_argument("--seed",type=int,default=20261025);p.add_argument("--device",default="cuda:0");p.add_argument("--no-geometry",action="store_true")
    a=p.parse_args();random.seed(a.seed);np.random.seed(a.seed)
    import torch
    torch.manual_seed(a.seed); torch.set_num_threads(2)
    from worldmot.models.world_object_update_head import WorldObjectUpdateHead
    all_scenes=list(range(1,25));fit_scenes=[x for x in all_scenes if x not in VAL_SCENES]
    val, val_cov=build_samples(a.cache,a.dataset_root,sorted(VAL_SCENES),a.no_geometry);fit,fit_cov=build_samples(a.cache,a.dataset_root,fit_scenes,a.no_geometry)
    if len(fit_cov)!=18 or len(val_cov)!=6: raise RuntimeError(f"full TRAIN selection coverage incomplete: fit={len(fit_cov)} val={len(val_cov)}")
    model=WorldObjectUpdateHead(geometry_dim=7,visual_dim=8,hidden_dim=128,memory_dim=16).to(a.device);optim=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4);history=[]
    for epoch in range(a.epochs): history.append({"epoch":epoch+1,"fit":run_epoch(model,optim,fit,a.device,True),"val":run_epoch(model,None,val,a.device,False)})
    threshold=0.5
    # Final model uses the same seed and initialization scheme on all 24 TRAIN
    # scenes after the validation configuration is frozen.
    torch.manual_seed(a.seed); final=WorldObjectUpdateHead(geometry_dim=7,visual_dim=8,hidden_dim=128,memory_dim=16).to(a.device);opt2=torch.optim.AdamW(final.parameters(),lr=2e-3,weight_decay=1e-4);full=fit+val;final_history=[]
    for epoch in range(a.epochs): final_history.append({"epoch":epoch+1,"train":run_epoch(final,opt2,full,a.device,True)})
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);torch.save(final.state_dict(),out.with_suffix(".pt"))
    report={"schema":"worldobject_w4_fulltrain_gate_v1","status":"FULL_TRAIN_GATE_PASSED","checkpoint":str(out.with_suffix(".pt")),"checkpoint_sha256":hashlib.sha256(out.with_suffix(".pt").read_bytes()).hexdigest(),"seed":a.seed,"epochs":a.epochs,"validation_scenes":sorted(VAL_SCENES),"fit_scenes":fit_scenes,"final_scenes":all_scenes,"final_scene_coverage":fit_cov+val_cov,"trainable_parameters":sum(p.numel() for p in final.parameters()),"geometry_mode":"NO_GEOMETRY_CONTROL" if a.no_geometry else "PERSISTENT_WORLD_OBJECT_GEOMETRY","selected_threshold":threshold,"selection_history":history,"final_history":final_history}
    out.write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps({"status":report["status"],"scenes":len(report["final_scene_coverage"]),"parameters":report["trainable_parameters"],"checkpoint":str(out.with_suffix(".pt"))},indent=2))


if __name__=="__main__":main()
