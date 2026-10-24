#!/usr/bin/env python3
"""Causal cache-replay runner for the persistent WorldObject route."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from worldmot.data.visiontrack import VisionTrack
from worldmot.data.worldobject_cache import WorldObjectCache
from worldmot.models.multi_view_world_object import JointWorldObjectBuilder
from worldmot.models.world_object_support import WorldObjectSupport
from worldmot.tracking.persistent_world_object_bank import PersistentWorldObjectBank


def make_support(r):
    box=np.asarray(r["box"]);box=box[np.isfinite(box).all(1)];fg=np.asarray(r["foreground"]);fg=fg[np.isfinite(fg).all(1)]
    return WorldObjectSupport(box,fg,np.asarray(r["center"]),np.asarray(r["cov"]),float(r["quality"]),float(r["foreground_fraction"]),"CACHE",0.,1.)


class FrozenUpdateHead:
    """Optional W4 gate; Ω/GMT remain frozen and no labels enter inference."""

    def __init__(self, checkpoint, device="cpu"):
        import torch
        from worldmot.models.world_object_update_head import WorldObjectUpdateHead
        self.torch = torch
        self.device = str(device)
        self.model = WorldObjectUpdateHead(geometry_dim=7, visual_dim=8,
                                          hidden_dim=128, memory_dim=16).to(self.device)
        state = torch.load(str(checkpoint), map_location=self.device, weights_only=True)
        self.model.load_state_dict(state, strict=True); self.model.eval()

    def __call__(self, row, width, height, memory):
        center = np.nan_to_num(np.asarray(row["center"], np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        diag = np.nan_to_num(np.diag(np.asarray(row["cov"], np.float32)), nan=1.0, posinf=1.0, neginf=1.0)
        geo = np.concatenate([center, diag, [float(row["quality"])]])[None].astype(np.float32)
        b = np.asarray(row["bbox"], np.float32)
        visual = np.asarray([[b[0] / max(width, 1), b[1] / max(height, 1),
                              b[2] / max(width, 1), b[3] / max(height, 1),
                              row["score"], row["foreground_fraction"],
                              float(row["view"] == 1), float(row["view"] == 2)]], np.float32)
        mem = np.asarray(memory, np.float32).reshape(1, 16)
        with self.torch.inference_mode():
            out = self.model(self.torch.as_tensor(geo, device=self.device),
                             self.torch.as_tensor(visual, device=self.device),
                             self.torch.as_tensor(mem, device=self.device))
        return (float(out["update_gate"].detach().cpu()[0]),
                float(self.torch.sigmoid(out["birth_logit"]).detach().cpu()[0]),
                out["memory_delta"].detach().cpu().numpy()[0].astype(np.float32))


def run_scene(cache, out_scene, pair_by_timestamp=None, learned=None):
    out_scene.mkdir(parents=True,exist_ok=True); bank=PersistentWorldObjectBank(min_hits=1); builder=JointWorldObjectBuilder(); rows=[]; timings=[]
    memory_by_gid = {}
    pair_by_timestamp = pair_by_timestamp or {}
    for fr in cache.frames():
        started=time.perf_counter();obs=[]
        for r in fr["rows"]:
            class O:pass
            o=O();o.camera_id=r["view"];o.detection_id=r["detection_id"];o.bbox_xywh=r["bbox"];o.score=r["score"];o.feature=np.asarray([r["score"],r["quality"]],np.float32);o.support=make_support(r);o.cache_row=r;obs.append(o)
        hyps,formation=builder.build(obs)
        delta_by_hyp = {}
        if learned is not None:
            pair = pair_by_timestamp.get(int(fr["timestamp"]))
            dims = {int(v): (int(rec.width), int(rec.height)) for v, rec in pair.images.items()} if pair is not None else {}
            for i, hyp in enumerate(hyps):
                rep = getattr(hyp.observations[0], "cache_row", None) if hyp.observations else None
                if rep is None: continue
                width, height = dims.get(int(rep["view"]), (1920, 1080))
                # Use the nearest pre-existing world object memory when the
                # frozen bank has one; new objects start from zero memory.
                memory = np.zeros(16, np.float32)
                if bank.objects:
                    nearest = min(bank.objects.values(), key=lambda x: float(np.linalg.norm(x.center - hyp.center)))
                    memory = memory_by_gid.get(int(nearest.global_id), memory)
                gate, birth, delta = learned(rep, width, height, memory)
                hyp.diagnostics["learned_update_gate"] = gate
                hyp.diagnostics["learned_birth_probability"] = birth
                delta_by_hyp[i] = delta
        # The optional gate is attached before the single causal bank update;
        # W3 has no such field and therefore follows the original path.
        objects,bd=bank.step(hyps,fr["timestamp"])
        for i, hyp in enumerate(hyps):
            gid = bank.last_assignments.get(i)
            if gid is not None and i in delta_by_hyp:
                memory_by_gid[int(gid)] = delta_by_hyp[i]
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
    manifest={"schema":"worldobject_prediction_v1","route":"W4_LEARNED_UPDATE" if learned is not None else "W3_PERSISTENT_BANK","scene":cache.scene,"status":"COMPLETE","timestamps":len(rows),"first_frame":rows[0]["timestamp"] if rows else None,"last_frame":rows[-1]["timestamp"] if rows else None,"causal":True,"GT_in_inference":False,"cache_manifest":cache.manifest,"tracks_sha256":hashlib.sha256((out_scene/"tracks.jsonl").read_bytes()).hexdigest(),"timings_sha256":hashlib.sha256((out_scene/"timings.jsonl").read_bytes()).hexdigest(),"fps_definition":"cache_replay_only_not_live_E2E"}
    (out_scene/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False));return manifest


def main():
    p=argparse.ArgumentParser();p.add_argument("--cache",required=True);p.add_argument("--output",required=True);p.add_argument("--scenes",nargs="*",type=int);p.add_argument("--dataset-root",default="/data/DATASETS/TRACKING/JDE/VisionTrack");p.add_argument("--checkpoint");p.add_argument("--device",default="cpu")
    a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True);scenes=a.scenes or [int(x.name.split("_")[1]) for x in sorted(Path(a.cache).glob("scene_*")) if x.is_dir()]; ds=VisionTrack(a.dataset_root,split="test",load_annotations=False) if a.checkpoint else None; learned=FrozenUpdateHead(a.checkpoint,a.device) if a.checkpoint else None
    reports=[]
    for sid in scenes:
        pairs = {int(p.frame_id): p for p in ds.synchronized_pairs(int(sid))} if ds is not None else {}
        reports.append(run_scene(WorldObjectCache(a.cache,sid),root/f"scene_{sid:02d}",pairs,learned))
    (root/"coverage.json").write_text(json.dumps({"schema":"worldobject_test_coverage_v1","scenes":reports,"scene_count":len(reports),"all_complete":all(x["status"]=="COMPLETE" for x in reports)},indent=2,allow_nan=False));print(json.dumps({"scenes":len(reports),"complete":all(x["status"]=="COMPLETE" for x in reports)},indent=2))


if __name__=="__main__":main()
