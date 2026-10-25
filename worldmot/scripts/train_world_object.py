#!/usr/bin/env python3
"""Full-scene causal V3.1 training with joint birth supervision and safe epoch resume."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import time
import numpy as np
from worldmot.runtime.experiment_authorization import require_experiment_authorization,source_commit,source_fingerprint
from worldmot.tracking.world_object_config import METHODS,METHOD_ALIASES,WorldObjectConfig

VAL_SCENES={4,8,12,16,20,24}


def atomic_json(path,data):
    path=Path(path);temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False));os.replace(temporary,path)


def run_epoch(model,config,args,scenes,optimizer=None,history_mode="predicted",*,epoch=0,phase="train"):
    import torch
    from worldmot.data.visiontrack import VisionTrack
    from worldmot.data.worldobject_cache import WorldObjectCache,sha256
    from worldmot.data.worldobject_scope import load_scope_lock,validate_scene_lock
    from worldmot.data.worldobject_observations import replay_observations
    from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
    from worldmot.runtime.world_object_tracking_core import WorldObjectTrackingCore
    from worldmot.training.world_object_causal_training import CausalWorldObjectTraining
    from worldmot.training.world_object_identity_targets import match_train_identities
    ds=VisionTrack(args.dataset_root,split="train",load_annotations=True)
    raw=ImmutableGMTDetectionCache(args.raw_cache);scope=load_scope_lock(args.scope_lock,"train",config.method_id)
    model.train(optimizer is not None)
    runner=CausalWorldObjectTraining(WorldObjectTrackingCore(config,model,args.device),history_mode)
    teacher=CausalWorldObjectTraining(WorldObjectTrackingCore(config,model,args.device),"teacher_forced")
    coverage=[];sums=Counter();diagnostics=Counter();count=0;started=time.perf_counter()
    logroot=Path(args.output)/"logs";logroot.mkdir(parents=True,exist_ok=True)
    logpath=logroot/f"{phase}_epoch_{epoch:02d}.jsonl"
    with logpath.open("w") as log:
        for scene in scenes:
            runner.reset(scene);teacher.reset(scene)
            cache=WorldObjectCache(args.cache,scene,raw_cache=raw,require_features=config.use_appearance,require_geometry=config.use_geometry)
            validate_scene_lock(scope,cache,"train")
            pairs={p.frame_id:p for p in ds.synchronized_pairs(scene)};times=[];audit=Counter();scene_diag=Counter()
            for index,frame in enumerate(cache.frames()):
                if index and index%args.clip_length==0:
                    log.write(json.dumps({"event":"CLIP_BOUNDARY","epoch":epoch,"scene":scene,
                        "clip":index//args.clip_length,**runner.clip_boundary()})+"\n")
                observations=replay_observations(frame);pair=pairs[frame["timestamp"]]
                labels,record=match_train_identities(observations,ds.ground_truth_pair(pair))
                _,losses=runner.step(observations,frame["timestamp"],labels,optimizer)
                sums.update(losses);audit.update(record);diagnostics.update(runner.last_diagnostics);scene_diag.update(runner.last_diagnostics)
                if optimizer is None and index<129:
                    teacher.step(observations,frame["timestamp"],labels)
                    for key in ("diagnostic_temporal_switches","same_person_split","false_cross_view_merge","first_appearance_split_count"):
                        diagnostics["teacher_prefix_"+key]+=teacher.last_diagnostics.get(key,0)
                        diagnostics["predicted_prefix_"+key]+=runner.last_diagnostics.get(key,0)
                count+=1;times.append(frame["timestamp"])
                if index%args.clip_length==0:
                    log.write(json.dumps({"event":"STEP","epoch":epoch,"scene":scene,"timestamp":frame["timestamp"],
                        "losses":losses,"diagnostics":runner.last_diagnostics,
                        "gpu_peak_bytes":torch.cuda.max_memory_allocated(args.device) if str(args.device).startswith("cuda") else 0})+"\n")
                    log.flush()
            if times!=list(pairs):raise ValueError("TRAIN_TIMESTAMP_METADATA_COVERAGE_MISMATCH")
            coverage.append({"scene":scene,"timestamps":len(times),"target_audit":dict(audit),"identity_diagnostics":dict(scene_diag),
                             "cache_manifest_sha256":sha256(cache.path/"manifest.json")})
            log.write(json.dumps({"event":"SCENE_COMPLETE","epoch":epoch,"phase":phase,**coverage[-1]})+"\n");log.flush()
    output={key:value/max(count,1) for key,value in sums.items()}
    output.update(frames=count,elapsed_s=time.perf_counter()-started,diagnostics=dict(diagnostics),
        birth_pair_precision=diagnostics["birth_pair_tp"]/max(1,diagnostics["birth_pair_tp"]+diagnostics["birth_pair_fp"]),
        birth_pair_recall=diagnostics["birth_pair_tp"]/max(1,diagnostics["birth_pair_tp"]+diagnostics["birth_pair_fn"]),
        birth_candidate_recall=diagnostics["birth_candidate_positive_retained"]/max(1,diagnostics["birth_candidate_positive_total"]),
        birth_proposal_recall=diagnostics["birth_proposal_positive_retained"]/max(1,diagnostics["birth_candidate_positive_total"]),
        average_history_objects=diagnostics["history_objects"]/max(count,1))
    print(json.dumps({"event":"EPOCH_COMPLETE","epoch":epoch,"phase":phase,"frames":count,
                      "identity_assignment":output.get("identity_assignment"),"joint_birth":output.get("joint_birth"),
                      "birth_candidate_recall":output["birth_candidate_recall"],"elapsed_s":output["elapsed_s"]}),flush=True)
    return output,coverage


def selection_score(row):
    metrics=row["validation"]
    return metrics["identity_assignment"]+metrics["joint_birth"]


def identity_error(metrics):
    d=metrics["diagnostics"]
    return d.get("same_person_split",0)/max(1,d.get("same_person_cross_view_pairs",0))+d.get("false_cross_view_merge",0)/max(1,d.get("different_person_cross_view_pairs",0))


def extension_allowed(selection):
    """Pre-registered TRAIN-only rule; never called after looking at TEST."""
    return len(selection)==4 and selection_score(selection[-1])<.98*selection_score(selection[-2]) and identity_error(selection[-1]["validation"])<identity_error(selection[0]["validation"])


def main():
    p=argparse.ArgumentParser()
    for key in ("cache","raw-cache","scope-lock","dataset-root","output"):p.add_argument("--"+key,required=True)
    p.add_argument("--authorization");p.add_argument("--method-id",choices=METHODS,default="FULL_WORLD_OBJECT_V3")
    p.add_argument("--epochs",type=int,default=4);p.add_argument("--max-epochs",type=int,default=6)
    p.add_argument("--seed",type=int,default=20261025);p.add_argument("--hidden-dim",type=int,default=128)
    p.add_argument("--clip-length",type=int,default=64);p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--device",default="cuda:0");p.add_argument("--resume",action="store_true")
    p.add_argument("--history-mode",choices=["predicted","teacher_forced"],default="predicted")
    a=p.parse_args();a.method_id=METHOD_ALIASES.get(a.method_id,a.method_id)
    approval=require_experiment_authorization(a.authorization,"train","train",a.method_id)
    if a.method_id=="NO_LEARNED_JOINT_BIRTH_V3":raise ValueError("INFERENCE_ABLATION_REUSES_FULL_CHECKPOINT_NO_DUPLICATE_TRAINING")
    if min(a.epochs,a.clip_length)<1 or a.max_epochs>6:raise ValueError("INVALID_REGISTERED_TRAINING_BUDGET")
    import torch
    from worldmot.models.world_object_update_head import WorldObjectUpdateHead,MODEL_SCHEMA
    from worldmot.models.world_object_checkpoint import GATE_SCHEMA,detector_contract
    from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
    from worldmot.data.worldobject_cache import sha256
    raw=ImmutableGMTDetectionCache(a.raw_cache)
    if raw.lock.get("split")!="train":raise ValueError("TRAIN_MUST_NOT_READ_TEST_CANDIDATES")
    config=WorldObjectConfig(raw.feature_dim,a.method_id);out=Path(a.output);out.mkdir(parents=True,exist_ok=a.resume)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(2)
    def new_model():return WorldObjectUpdateHead(config.feature_dim,a.hidden_dim,birth_pair_chunk=config.birth_pair_chunk).to(a.device)
    model=new_model();optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr)
    stage="selection";selection=[];final_history=[];selected=None;coverage=[]
    resume_path=out/"resume.pt";resume_meta=out/"resume.json"
    if a.resume and resume_meta.is_file():
        meta=json.loads(resume_meta.read_text())
        if meta["source_sha256"]!=source_fingerprint() or meta["method_id"]!=a.method_id or meta["sha256"]!=sha256(resume_path):
            raise ValueError("RESUME_SOURCE_OR_CHECKPOINT_MISMATCH")
        state=torch.load(resume_path,map_location=a.device,weights_only=False)
        model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"])
        stage,selection,final_history,selected,coverage=[state[k] for k in ("stage","selection","final_history","selected","coverage")]
        random.setstate(state["python_rng"]);np.random.set_state(state["numpy_rng"]);torch.set_rng_state(state["torch_rng"].cpu())
        if str(a.device).startswith("cuda"):torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
    def save_boundary():
        state={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":None,"stage":stage,
            "selection":selection,"final_history":final_history,"selected":selected,"coverage":coverage,
            "python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),
            "cuda_rng":torch.cuda.get_rng_state_all() if str(a.device).startswith("cuda") else []}
        temp=resume_path.with_suffix(".tmp");torch.save(state,temp);os.replace(temp,resume_path)
        atomic_json(resume_meta,{"source_sha256":source_fingerprint(),"method_id":a.method_id,"sha256":sha256(resume_path),
            "resume_boundary":"LAST_COMPLETE_EPOCH_WITH_SCENE_RESET_ONLY","scheduler_policy":"constant_lr"})
        atomic_json(out/"progress.json",{"stage":stage,"selection_history":selection,"final_history":final_history,"selected_epochs":selected})
    if stage=="selection":
        fit=sorted(set(range(1,25))-VAL_SCENES);budget=a.max_epochs if len(selection)>=4 and extension_allowed(selection[:4]) else a.epochs
        while len(selection)<budget:
            epoch=len(selection)+1
            train,tc=run_epoch(model,config,a,fit,optimizer,a.history_mode,epoch=epoch,phase="fit")
            validation,vc=run_epoch(model,config,a,sorted(VAL_SCENES),epoch=epoch,phase="validation")
            if len(tc)!=18 or len(vc)!=6:raise RuntimeError("INCOMPLETE_TRAIN_SELECTION_COVERAGE")
            selection.append({"epoch":epoch,"train":train,"validation":validation})
            if len(selection)==4 and extension_allowed(selection):budget=a.max_epochs
            save_boundary()
        selected=min(selection,key=selection_score)["epoch"]
        torch.manual_seed(a.seed);model=new_model();optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr)
        stage="final";save_boundary()
    if stage=="final":
        while len(final_history)<selected:
            epoch=len(final_history)+1
            metrics,coverage=run_epoch(model,config,a,list(range(1,25)),optimizer,a.history_mode,epoch=epoch,phase="final")
            final_history.append({"epoch":epoch,"train":metrics});save_boundary()
    if len(coverage)!=24 or sum(x["timestamps"] for x in coverage)!=28754:raise RuntimeError("FULL_TRAIN_GATE_COVERAGE_FAILED")
    supervised=sum(x["target_audit"].get("matched_identity_observations",0) for x in coverage)
    if supervised<=0 or final_history[-1]["train"]["diagnostics"].get("crossview_birth_positive_count",0)<=0:
        raise RuntimeError("FULL_TRAIN_GATE_NO_VALID_IDENTITY_OR_JOINT_BIRTH_SUPERVISION")
    weights=out/"identity_v3_joint_birth.pt"
    torch.save({"schema":MODEL_SCHEMA,"model_config":model.configuration(),"state_dict":model.state_dict()},weights)
    from worldmot.training.world_object_assignment_loss import LOSS_WEIGHTS
    report={"schema":GATE_SCHEMA,"status":"FULL_TRAIN_GATE_PASSED","checkpoint":weights.name,
        "checkpoint_sha256":sha256(weights),"model_config":model.configuration(),"feature_dim":config.feature_dim,
        "method_id":a.method_id,"tracking_config":config.to_dict(),"seed":a.seed,"source_commit":source_commit(),
        "source_sha256":source_fingerprint(),"training_budget":{"selection_epochs":len(selection),"selected_final_epochs":selected,
            "max_epochs":a.max_epochs,"lr":a.lr,"clip_length":a.clip_length,"clip_identity_reset":False,"history_mode":a.history_mode},
        "loss_weights":LOSS_WEIGHTS,"detector_contract":detector_contract(raw.lock),"approval":approval,"GT_in_model_inputs":False,
        "scope_lock_sha256":sha256(a.scope_lock),"final_scenes":list(range(1,25)),"final_timestamps":28754,"coverage":coverage,
        "identity_supervision_observations":supervised,"selection_history":selection,"final_history":final_history,
        "trainable_parameters":sum(p.numel() for p in model.parameters())}
    atomic_json(out/"train_gate.json",report)
    print(json.dumps({"event":"FULL_TRAIN_GATE_PASSED","method":a.method_id,"checkpoint_sha256":report["checkpoint_sha256"]}),flush=True)


if __name__=="__main__":main()
