#!/usr/bin/env python3
"""Persistent single-GPU cache worker with atomic per-scene leases and safe recovery."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
from worldmot.runtime.experiment_authorization import require_experiment_authorization,source_fingerprint


def claim_scene(queue,prefer,shortest=False):
    choices=[]
    for path in Path(queue).glob("*.pending"):
        try:task=json.loads(path.read_text())
        except FileNotFoundError:continue  # Another worker already claimed this scene.
        choices.append((task["split"]!=prefer,task["frames"] if shortest else -task["frames"],path.name,path,task))
    for _,_,_,path,task in sorted(choices):
        running=path.with_suffix(".running")
        try:os.rename(path,running)
        except FileNotFoundError:continue
        task.update(pid=os.getpid(),claimed_at=time.time(),source_sha256=source_fingerprint())
        running.write_text(json.dumps(task));return running,task
    return None,None


def main():
    p=argparse.ArgumentParser()
    for key in ("authorization","queue","root","dataset-root","raw-train","raw-test","omega-source","omega-weights"):
        p.add_argument("--"+key,required=True)
    p.add_argument("--prefer",choices=["train","test"],default="train")
    p.add_argument("--shortest-first",action="store_true");p.add_argument("--stop-file")
    a=p.parse_args()
    for split in ("train","test"):require_experiment_authorization(a.authorization,"cache",split,"FULL_WORLD_OBJECT_V3")
    import torch
    from worldmot.data.visiontrack import VisionTrack
    from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
    from worldmot.models.omega_geometry_backend import OmegaGeometryBackend
    from cache_worldobject_support import cache_scene
    torch.set_num_threads(2)
    import cv2
    cv2.setNumThreads(1)
    if torch.cuda.device_count()!=1:raise RuntimeError("WORKER_MUST_SEE_EXACTLY_ONE_PHYSICAL_GPU")
    datasets={split:VisionTrack(a.dataset_root,split=split,load_annotations=False) for split in ("train","test")}
    raw={"train":ImmutableGMTDetectionCache(a.raw_train),"test":ImmutableGMTDetectionCache(a.raw_test)}
    omega=None
    while not (a.stop_file and Path(a.stop_file).exists()):
        lease,task=claim_scene(a.queue,a.prefer,a.shortest_first)
        if task is None:break
        try:
            if omega is None:
                omega=OmegaGeometryBackend(a.omega_source,a.omega_weights,device="cuda:0")
                if omega.status!="READY":raise RuntimeError(omega.reason)
            split,scene=task["split"],task["scene"]
            print(json.dumps({"event":"CACHE_SCENE_START","split":split,"scene":scene,"pid":os.getpid()}),flush=True)
            cache_scene(datasets[split],scene,Path(a.root)/f"cache_{split}_v2",omega,raw[split],split)
            task.update(status="COMPLETE",finished_at=time.time(),gpu_peak_bytes=torch.cuda.max_memory_allocated())
            lease.write_text(json.dumps(task));os.rename(lease,lease.with_suffix(".done"))
        except Exception as exc:
            task.update(status="FAILED",error=str(exc),finished_at=time.time())
            lease.write_text(json.dumps(task));os.rename(lease,lease.with_suffix(".failed"));raise


if __name__=="__main__":main()
