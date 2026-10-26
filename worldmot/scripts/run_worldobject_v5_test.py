#!/usr/bin/env python3
"""One complete causal scene from frozen V5 gates; no original TEST GT access."""
from __future__ import annotations
import argparse,json,os,time,fcntl,resource
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from worldmot.runtime.world_object_v5_contract import (require_v5_scope,verify_scene_scope,atomic_json,
    v5_source_fingerprint,load_v5_membership_gate)
from worldmot.runtime.experiment_authorization import source_commit
from worldmot.runtime.world_object_v4_test_contract import install_no_gt_guard,common_fields_equal,observation_signature
from worldmot.tracking.world_object_v5_config import registered_config
from worldmot.runtime.world_object_v5_core import ObservationAnchoredWorldObjectCore
from worldmot.models.world_object_v5_geometry import v5_observations
from worldmot.models.world_object_v4_checkpoint import load_frozen_r2_head
from worldmot.data.worldobject_cache import WorldObjectCache,sha256
from worldmot.data.worldobject_observations import replay_observations
from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache

COMMON_C1_FIELDS=["schema","timestamp","scene_id","method_id","tracks_2d_by_camera","decisions","tracks_3d","bank","causal","GT_in_inference","output_policy"]


def completed(path,scope,variant,scene):
    m=json.loads((path/"manifest.json").read_text());e=scope["splits"]["test"]["scenes"][str(scene)]
    if (m["status"]!="COMPLETE" or m["variant"]!=variant or m["scene"]!=scene or m["v5_source_sha256"]!=scope["v5_source_sha256"]
            or m["timestamps"]!=e["timestamps"] or m["candidate_count"]!=e["candidates"]
            or m.get("support_manifest_sha256")!=e["support_manifest_sha256"]
            or m.get("raw_manifest_sha256")!=e["raw_manifest_sha256"]):raise ValueError("V5_COMPLETED_SCENE_SCOPE_MISMATCH")
    expected=scope["frozen_C1_checkpoint_sha256"] if variant in {"B0","B1"} else scope["final_weights"][variant]["checkpoint_sha256"]
    if m["checkpoint_sha256"]!=expected or (variant!="B0" and m["config"]!=scope["variant_configs"][variant]):raise ValueError("V5_COMPLETED_WEIGHT_OR_CONFIG_MISMATCH")
    for name,digest in m["output_sha256"].items():
        if sha256(path/name)!=digest:raise ValueError("V5_COMPLETED_OUTPUT_CHANGED")
    return m


def main():
    p=argparse.ArgumentParser()
    for k in ["scope-lock","authorization","calibration","output"]:p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--variant",choices=["B0","B1","B2","B3","B4"],required=True)
    p.add_argument("--scene",required=True,type=int,choices=range(1,23));p.add_argument("--device",default="cuda:0")
    a=p.parse_args();scope,approval=require_v5_scope(a.scope_lock,a.authorization,"replay","test",a.variant)
    destination=a.output/f"predictions/{a.variant}/scene_{a.scene:02d}";destination.parent.mkdir(parents=True,exist_ok=True)
    lockpath=a.output/f"scene_locks/{a.variant}_scene{a.scene:02d}.lock";lockpath.parent.mkdir(exist_ok=True)
    lock=lockpath.open("a+");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (destination/"manifest.json").exists():
        m=completed(destination,scope,a.variant,a.scene);print(json.dumps({"status":"COMPLETE_REUSED","variant":a.variant,"scene":a.scene,"timestamps":m["timestamps"]}));return
    stage=destination.with_name(destination.name+f".incomplete.{os.getpid()}")
    if stage.exists():raise FileExistsError("Preserve previous incomplete attempts")
    stage.mkdir();started=time.perf_counter();torch.set_num_threads(1)
    frozen_source=v5_source_fingerprint();raw=ImmutableGMTDetectionCache(scope["splits"]["test"]["raw_cache"])
    cfg=None if a.variant=="B0" else registered_config(a.variant,json.loads(a.calibration.read_text()))
    if sha256(a.calibration)!=scope["calibration_sha256"]:raise ValueError("CALIBRATION_CHANGED_AFTER_FREEZE")
    if a.variant in {"B0","B1"}:
        gate=Path(scope["splits"]["train"]["support_cache"]).parent/"train/FULL_WORLD_OBJECT_V3/train_gate.json"
        model,weight=load_frozen_r2_head(gate,raw.lock,a.device)
    else:
        record=scope["final_weights"][a.variant]
        if sha256(record["train_gate"])!=record["train_gate_sha256"]:raise ValueError("FROZEN_TRAIN_GATE_CHANGED")
        model,weight=load_v5_membership_gate(record["train_gate"],cfg,a.device,record["checkpoint_sha256"])
    if a.variant=="B0":
        from worldmot.runtime.world_object_tracking_core import WorldObjectTrackingCore
        from worldmot.tracking.world_object_v4_variants import state_configs
        core=WorldObjectTrackingCore(state_configs(1024)["C1"],model,a.device)
    else:core=ObservationAnchoredWorldObjectCore(cfg,model,a.device)
    core.reset(a.scene);cache=WorldObjectCache(scope["splits"]["test"]["support_cache"],a.scene,raw_cache=raw)
    expected=verify_scene_scope(scope,"test",a.scene,cache);guard=install_no_gt_guard();counts=Counter();seen=[];timings=Counter();chain=__import__('hashlib').sha256()
    reference=None
    if a.variant=="B0":reference=(Path(scope["original_C1_reference_predictions"])/f"scene_{a.scene:02d}/tracks.jsonl").open()
    try:
        with (stage/"tracks.jsonl").open("x") as tracks,(stage/"timings.jsonl").open("x") as times,(stage/"birth_audit.jsonl").open("x") as birth_audit:
            it=iter(cache.frames())
            while True:
                start=time.perf_counter()
                try:frame=next(it)
                except StopIteration:break
                read_s=time.perf_counter()-start;t=frame["timestamp"]
                if seen and t!=seen[-1]+1:raise ValueError("V5_COMPLETE_CAUSAL_TIMESTAMP_ORDER")
                if a.variant=="B0":obs=replay_observations(frame);context=None
                else:obs,context=v5_observations(frame,cfg.use_geometry)
                before=observation_signature(obs);begin=time.perf_counter()
                row=core.step(obs,t) if a.variant=="B0" else core.step(obs,t,context)
                core_s=time.perf_counter()-begin
                if before!=observation_signature(obs):raise ValueError("FROZEN_BBOX_SCORE_ROI_INPUT_MUTATION")
                if a.variant=="B0":
                    old=json.loads(next(reference))
                    common_fields_equal({k:old[k] for k in COMMON_C1_FIELDS},{k:row[k] for k in COMMON_C1_FIELDS})
                    counts["exact_original_C1_timestamps"]+=1
                    original_decisions={(int(x["camera_id"]),int(x["detection_id"])):x for x in old["decisions"]}
                    if set(original_decisions)!={o.key for o in obs}:raise ValueError("B0_REFERENCE_CANDIDATES_CHANGED")
                output={(int(v),int(x["detection_id"])):x for v,items in row["tracks_2d_by_camera"].items() for x in items}
                decisions={(int(x["camera_id"]),int(x["detection_id"])):x for x in row["decisions"]}
                if set(decisions)!={o.key for o in obs}:raise ValueError("V5_MISSING_DETECTION_DECISION")
                for o in obs:
                    if o.key not in output:
                        if decisions[o.key].get("stage")!="TRACK_AWARE_DUPLICATE":raise ValueError("CURRENT_DETECTION_SILENTLY_DROPPED")
                        counts["audited_duplicate_suppression"]+=1;continue
                    x=output[o.key]
                    if x["bbox_xywh"]!=o.bbox_xywh.tolist() or x["score"]!=o.score:raise ValueError("V5_DETECTOR_OUTPUT_CHANGED")
                if a.variant!="B0":birth_audit.write(json.dumps({"timestamp":t,**core.bank.last_birth_audit},allow_nan=False)+"\n")
                row.update(experiment_variant=a.variant)
                write_start=time.perf_counter();tracks.write(json.dumps(row,allow_nan=False,separators=(",",":"))+"\n")
                timing={"timestamp":t,"cache_read_and_join_s":read_s,"tracking_core_s":core_s,**core.last_timings,
                        "output_serialization_s":time.perf_counter()-write_start,"complete_timestamp_s":time.perf_counter()-start}
                times.write(json.dumps(timing,allow_nan=False)+"\n");timings.update({k:v for k,v in timing.items() if k.endswith('_s')})
                chain.update(str(t).encode()+before.encode());seen.append(t);counts.update(candidates=len(obs),outputs=len(output))
                if len(seen)%256==0:
                    tracks.flush();times.flush();birth_audit.flush()
                    atomic_json(stage/"progress.json",{"variant":a.variant,"scene":a.scene,"timestamps":len(seen),"expected":expected["timestamps"],"candidates":counts["candidates"]})
        if reference is not None and reference.readline():raise ValueError("B0_REFERENCE_EXTRA_TIMESTAMPS")
        if len(seen)!=expected["timestamps"] or counts["candidates"]!=expected["candidates"]:raise ValueError("V5_FULL_SCENE_INPUT_COVERAGE")
        if guard["blocked_GT_reads"] or frozen_source!=v5_source_fingerprint():raise ValueError("GT_READ_OR_SOURCE_CHANGED_DURING_TEST")
        if a.variant!="B0" and core.scene_reset_count!=1:raise ValueError("V5_BANK_RESET_WITHIN_SCENE")
        manifest={"status":"COMPLETE","variant":a.variant,"scene":a.scene,"timestamps":len(seen),"first_timestamp":seen[0],"last_timestamp":seen[-1],
            "candidate_count":counts["candidates"],"output_count":counts["outputs"],"audited_duplicate_suppression":counts["audited_duplicate_suppression"],
            "exact_original_C1_timestamps":counts["exact_original_C1_timestamps"],"v5_source_sha256":frozen_source,"source_commit":source_commit(),
            "checkpoint":weight,"checkpoint_sha256":weight["checkpoint_sha256"],"config":cfg.to_dict() if cfg else core.config.to_dict(),
            "scope_sha256":sha256(a.scope_lock),"approval":approval,"input_chain_sha256":chain.hexdigest(),"candidate_changes":0,"ROI_mutations":0,
            "support_manifest_sha256":sha256(cache.path/"manifest.json"),"raw_manifest_sha256":expected["raw_manifest_sha256"],
            "output_sha256":{n:sha256(stage/n) for n in ['tracks.jsonl','timings.jsonl','birth_audit.jsonl']},
            "GT_in_inference":False,"causal":True,"scene_reset_count":1,"device":a.device,"precision":"float32",
            "timing_label":"CACHED_GEOMETRY_AND_DETECTOR_REPLAY","timing_sums":dict(timings),"worker_wall_s":time.perf_counter()-started,
            "peak_RSS_bytes":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            "CUDA_peak_allocated_bytes":torch.cuda.max_memory_allocated() if str(a.device).startswith('cuda') else 0}
        atomic_json(stage/"manifest.json",manifest)
        if destination.exists():raise FileExistsError("Never overwrite completed scene")
        os.rename(stage,destination)
        for file in destination.iterdir():
            if file.is_file():file.chmod(0o444)
        print(json.dumps({"status":"COMPLETE","variant":a.variant,"scene":a.scene,"timestamps":len(seen),"candidate_count":counts["candidates"],"tracks_sha256":manifest["output_sha256"]["tracks.jsonl"]}),flush=True)
    except Exception as error:
        atomic_json(stage/"failure.json",{"status":"FAILED_PRESERVED","variant":a.variant,"scene":a.scene,"type":type(error).__name__,"reason":str(error),"completed_timestamps":len(seen)})
        raise
    finally:
        if reference is not None:reference.close()


if __name__=="__main__":main()
