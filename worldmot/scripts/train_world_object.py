#!/usr/bin/env python3
"""V3 identity-supervised causal TRAIN runner; disabled without explicit approval."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import numpy as np
from worldmot.runtime.experiment_authorization import require_experiment_authorization, source_commit,source_fingerprint
from worldmot.tracking.world_object_config import METHODS, WorldObjectConfig

VAL_SCENES={4,8,12,16,20,24}


def run_epoch(model,config,args,scenes,optimizer=None,history_mode="predicted"):
    from worldmot.data.visiontrack import VisionTrack
    from worldmot.data.worldobject_cache import WorldObjectCache
    from worldmot.data.worldobject_scope import load_scope_lock,validate_scene_lock
    from worldmot.data.worldobject_observations import replay_observations
    from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
    from worldmot.runtime.world_object_tracking_core import WorldObjectTrackingCore
    from worldmot.training.world_object_causal_training import CausalWorldObjectTraining
    from worldmot.training.world_object_identity_targets import match_train_identities
    ds=VisionTrack(args.dataset_root,split="train",load_annotations=True)
    raw=ImmutableGMTDetectionCache(args.raw_cache)
    scope=load_scope_lock(args.scope_lock,"train",config.method_id)
    model.train(optimizer is not None)
    runner=CausalWorldObjectTraining(WorldObjectTrackingCore(config,model,args.device),history_mode)
    coverage=[];sums={};count=0
    for scene in scenes:
        runner.reset(scene)
        cache=WorldObjectCache(args.cache,scene,raw_cache=raw,require_features=config.use_appearance,
                               require_geometry=config.use_geometry)
        validate_scene_lock(scope,cache,"train")
        pairs={p.frame_id:p for p in ds.synchronized_pairs(scene)}
        times=[];audit={}
        for index,frame in enumerate(cache.frames()):
            if index and index%args.clip_length==0:
                runner.reset(scene)
            observations=replay_observations(frame)
            pair=pairs[frame["timestamp"]]
            labels,record=match_train_identities(observations,ds.ground_truth_pair(pair))
            _,losses=runner.step(observations,frame["timestamp"],labels,optimizer)
            for key,value in losses.items():
                sums[key]=sums.get(key,0)+value
            for key,value in record.items():
                audit[key]=audit.get(key,0)+value
            count+=1;times.append(frame["timestamp"])
        if times!=list(pairs):
            raise ValueError("TRAIN_TIMESTAMP_METADATA_COVERAGE_MISMATCH")
        coverage.append({"scene":scene,"timestamps":len(times),"target_audit":audit,
                         "cache_manifest_sha256":__import__("worldmot.data.worldobject_cache",fromlist=["sha256"]).sha256(cache.path/"manifest.json")})
    return {k:v/max(count,1) for k,v in sums.items()},coverage


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--authorization");p.add_argument("--cache",required=True);p.add_argument("--raw-cache",required=True)
    p.add_argument("--scope-lock",required=True)
    p.add_argument("--dataset-root",required=True);p.add_argument("--output",required=True)
    p.add_argument("--method-id",choices=METHODS,default="FULL_WORLD_OBJECT_V3")
    p.add_argument("--epochs",type=int,default=4);p.add_argument("--seed",type=int,default=20261025)
    p.add_argument("--hidden-dim",type=int,default=128);p.add_argument("--clip-length",type=int,default=64)
    p.add_argument("--lr",type=float,default=2e-4);p.add_argument("--device",default="cuda:0")
    p.add_argument("--history-mode",choices=["predicted","teacher_forced"],default="predicted")
    a=p.parse_args()
    approval=require_experiment_authorization(a.authorization,"train","train",a.method_id)
    if min(a.epochs,a.clip_length)<1:
        raise ValueError("POSITIVE_TRAINING_BUDGET_REQUIRED")
    import torch
    from worldmot.models.world_object_update_head import WorldObjectUpdateHead,MODEL_SCHEMA
    from worldmot.models.world_object_checkpoint import GATE_SCHEMA,detector_contract
    from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
    from worldmot.data.worldobject_cache import sha256
    raw=ImmutableGMTDetectionCache(a.raw_cache)
    if raw.lock.get("split")!="train":
        raise ValueError("TRAIN_MUST_NOT_READ_TEST_CANDIDATES")
    config=WorldObjectConfig(raw.feature_dim,a.method_id)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    model=WorldObjectUpdateHead(config.feature_dim,a.hidden_dim).to(a.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr)
    fit=sorted(set(range(1,25))-VAL_SCENES);selection=[]
    for epoch in range(1,a.epochs+1):
        train,tc=run_epoch(model,config,a,fit,optimizer,a.history_mode)
        val,vc=run_epoch(model,config,a,sorted(VAL_SCENES))
        if len(tc)!=18 or len(vc)!=6:
            raise RuntimeError("INCOMPLETE_TRAIN_SELECTION_COVERAGE")
        selection.append({"epoch":epoch,"train":train,"validation":val})
    selected=min(selection,key=lambda row:row["validation"]["identity_assignment"])["epoch"]
    torch.manual_seed(a.seed)
    final=WorldObjectUpdateHead(config.feature_dim,a.hidden_dim).to(a.device)
    optimizer=torch.optim.AdamW(final.parameters(),lr=a.lr)
    history=[];coverage=[]
    for epoch in range(selected):
        loss,coverage=run_epoch(final,config,a,list(range(1,25)),optimizer,a.history_mode)
        history.append({"epoch":epoch+1,"train":loss})
    if len(coverage)!=24 or sum(x["timestamps"] for x in coverage)!=28754:
        raise RuntimeError("FULL_TRAIN_GATE_COVERAGE_FAILED")
    supervised=sum(x["target_audit"].get("matched_identity_observations",0) for x in coverage)
    if supervised<=0:
        raise RuntimeError("FULL_TRAIN_GATE_NO_VALID_IDENTITY_SUPERVISION")
    weights=output/"identity_v3.pt"
    torch.save({"schema":MODEL_SCHEMA,"model_config":final.configuration(),"state_dict":final.state_dict()},weights)
    report={"schema":GATE_SCHEMA,"status":"FULL_TRAIN_GATE_PASSED","checkpoint":weights.name,
        "checkpoint_sha256":sha256(weights),"model_config":final.configuration(),"feature_dim":config.feature_dim,
        "method_id":a.method_id,"tracking_config":config.to_dict(),"seed":a.seed,"source_commit":source_commit(),
        "source_sha256":source_fingerprint(),
        "training_budget":{"selection_epochs":a.epochs,"selected_final_epochs":selected,"lr":a.lr,
                           "clip_length":a.clip_length,"history_mode":a.history_mode},
        "detector_contract":detector_contract(raw.lock),"approval":approval,"GT_in_model_inputs":False,
        "scope_lock_sha256":sha256(a.scope_lock),
        "final_scenes":list(range(1,25)),"final_timestamps":28754,"coverage":coverage,
        "identity_supervision_observations":supervised,
        "selection_history":selection,"final_history":history,
        "trainable_parameters":sum(p.numel() for p in final.parameters())}
    (output/"train_gate.json").write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
