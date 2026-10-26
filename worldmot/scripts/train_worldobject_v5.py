#!/usr/bin/env python3
"""Full-scene membership training with scene-boundary resume and a full24 gate."""
from __future__ import annotations
import argparse,json,math,os,random,time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from worldmot.runtime.world_object_v5_contract import (require_v5_scope,atomic_json,registration_path,
    initial_membership_model,state_tensor_sha256,v5_source_fingerprint)
from worldmot.runtime.experiment_authorization import source_fingerprint,source_commit
from worldmot.tracking.world_object_v5_config import registered_config
from worldmot.runtime.world_object_v5_core import ObservationAnchoredWorldObjectCore
from worldmot.training.world_object_v5_causal_training import CausalMembershipTraining
from worldmot.training.world_object_membership_loss import LOSS_WEIGHTS
from worldmot.data.worldobject_cache import WorldObjectCache,sha256
from worldmot.data.visiontrack import VisionTrack
from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
from worldmot.models.world_object_v5_geometry import v5_observations
from worldmot.training.world_object_identity_targets import match_train_identities
from worldmot.runtime.world_object_v5_contract import verify_scene_scope


def learning_rate(reg,epoch,fraction):
    progress=epoch-1+fraction
    if progress<reg["warmup_epochs"]:
        return reg["learning_rate"]*max(.01,progress/reg["warmup_epochs"])
    decay=np.clip((progress-reg["warmup_epochs"])/(reg["base_epochs"]-reg["warmup_epochs"]),0,1)
    return float(reg["minimum_learning_rate"]+.5*(reg["learning_rate"]-reg["minimum_learning_rate"])*(1+math.cos(math.pi*decay)))


def identity_error(row):
    d=row["diagnostics"]
    return d.get("same_person_split",0)/max(1,d.get("same_person_cross_view_pairs",0))+d.get("false_cross_view_merge",0)/max(1,d.get("different_person_cross_view_pairs",0))


def extension_allowed(selection):
    if len(selection)<12:raise ValueError("TWELVE_COMPLETE_TRAIN_SELECTION_EPOCHS_REQUIRED")
    a,b,c=[selection[i]["validation"] for i in (9,10,11)]
    return c["losses"]["per_camera_membership"]<.98*(a["losses"]["per_camera_membership"]+b["losses"]["per_camera_membership"])/2 and identity_error(c)<identity_error(selection[8]["validation"])


def rng_state():
    return {"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),
            "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"]);np.random.set_state(state["numpy"]);torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def summarize(scenes):
    count=sum(x["timestamps"] for x in scenes);loss=Counter();diagnostics=Counter();audit=Counter()
    for scene in scenes:
        loss.update(scene["loss_sums"]);diagnostics.update(scene["diagnostics"]);audit.update(scene["target_audit"])
    return {"frames":count,"losses":{k:v/max(1,count) for k,v in loss.items()},"diagnostics":dict(diagnostics),
            "target_audit":dict(audit),"wall_s":sum(x["wall_s"] for x in scenes),"scene_count":len(scenes)}


def main():
    p=argparse.ArgumentParser()
    for k in ["scope-lock","authorization","calibration","output","common-budget"]:p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--variant",choices=["B2","B3","B4"],required=True);p.add_argument("--phase",choices=["selection","final"],required=True)
    p.add_argument("--device",default="cuda:0");p.add_argument("--resume",action="store_true")
    a=p.parse_args();scope,approval=require_v5_scope(a.scope_lock,a.authorization,"train","train",a.variant)
    reg=json.loads(registration_path().read_text());cal=json.loads(a.calibration.read_text());cfg=registered_config(a.variant,cal)
    if sha256(a.calibration)!=scope["calibration_sha256"]:raise ValueError("FROZEN_CALIBRATION_CHANGED")
    if str(a.device).startswith("cuda") and not torch.cuda.is_available():raise RuntimeError("REQUESTED_CUDA_NOT_AVAILABLE")
    torch.set_num_threads(1)
    model=initial_membership_model(cfg,reg["seed"],a.device);initial_sha=state_tensor_sha256(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=reg["learning_rate"],weight_decay=reg["weight_decay"])
    a.output.mkdir(parents=True,exist_ok=True);phase_root=a.output/a.phase;phase_root.mkdir(parents=True,exist_ok=True)
    state={"epoch":1,"section":"fit" if a.phase=="selection" else "train","epoch_scenes":[],"epoch_validation_scenes":[],
           "history":[],"complete":False,"optimizer_steps":0,"resume_sequence":0}
    pointer=phase_root/"RESUME.json"
    if pointer.exists():
        if not a.resume:raise FileExistsError("Existing run requires explicit verified --resume")
        meta=json.loads(pointer.read_text());path=phase_root/meta["checkpoint"]
        if meta["v5_source_sha256"]!=v5_source_fingerprint() or meta["scope_sha256"]!=sha256(a.scope_lock) or sha256(path)!=meta["sha256"]:
            raise ValueError("V5_RESUME_SOURCE_SCOPE_OR_WEIGHT_MISMATCH")
        payload=torch.load(path,map_location=a.device,weights_only=False)
        if payload["initial_state_tensor_sha256"]!=initial_sha or payload["tracking_config"]!=cfg.to_dict():raise ValueError("V5_RESUME_INITIALIZATION_MISMATCH")
        model.load_state_dict(payload["model"],strict=True);optimizer.load_state_dict(payload["optimizer"]);state=payload["progress"];restore_rng(payload["RNG"])
    ds=VisionTrack(scope["dataset_root"],split="train",load_annotations=True)
    raw=ImmutableGMTDetectionCache(scope["splits"]["train"]["raw_cache"])
    source=v5_source_fingerprint();start=time.perf_counter()

    def budget():
        if a.common_budget.exists():
            value=json.loads(a.common_budget.read_text())
            if value["registered_config_sha256"]!=sha256(registration_path()) or value["v5_source_sha256"]!=source or value["epochs"] not in {12,16}:
                raise ValueError("V5_COMMON_TRAIN_BUDGET_MISMATCH")
            return value["epochs"]
        if a.phase=="final":raise RuntimeError("FINAL_TRAIN_REQUIRES_FROZEN_TRAIN_ONLY_COMMON_BUDGET")
        return reg["base_epochs"]

    def save_boundary():
        if v5_source_fingerprint()!=source:raise ValueError("V5_SOURCE_CHANGED_DURING_TRAINING")
        state["resume_sequence"]+=1;filename=f"resume_{state['resume_sequence']:05d}.pt";target=phase_root/filename
        temporary=target.with_suffix(".partial")
        torch.save({"model":model.state_dict(),"optimizer":optimizer.state_dict(),"progress":state,"RNG":rng_state(),
            "tracking_config":cfg.to_dict(),"initial_state_tensor_sha256":initial_sha},temporary)
        os.replace(temporary,target)
        atomic_json(pointer,{"checkpoint":filename,"sha256":sha256(target),"v5_source_sha256":source,
            "scope_sha256":sha256(a.scope_lock),"resume_boundary":"COMPLETE_SCENE_ONLY","scheduler":"REGISTERED_WARMUP_COSINE12_THEN_MINIMUM_LR","optimizer_restored":True})
        atomic_json(phase_root/"progress.json",{"variant":a.variant,"phase":a.phase,"epoch":state["epoch"],"section":state["section"],
            "complete":state["complete"],"completed_epochs":len(state["history"]),"current_epoch_complete_scenes":len(state["epoch_scenes"]),
            "current_validation_complete_scenes":len(state["epoch_validation_scenes"]),"optimizer_steps":state["optimizer_steps"],
            "resume_checkpoint_sha256":sha256(target),"budget":budget(),"history_mode":"predicted_from_first_timestamp"})
        # Only our rolling recovery artifacts are removed; frozen and final
        # checkpoints plus all scene/epoch coverage reports remain untouched.
        for old in sorted(phase_root.glob("resume_*.pt"))[:-2]:old.unlink()

    def run_scene(sid,epoch,training,section):
        model.train(training)
        runner=CausalMembershipTraining(ObservationAnchoredWorldObjectCore(cfg,model,a.device));runner.reset(sid)
        cache=WorldObjectCache(scope["splits"]["train"]["support_cache"],sid,raw_cache=raw)
        expected=verify_scene_scope(scope,"train",sid,cache)
        pairs={pair.frame_id:pair for pair in ds.synchronized_pairs(sid)}
        directory=phase_root/"logs";directory.mkdir(exist_ok=True)
        logpath=directory/f"{section}_epoch{epoch:02d}_scene{sid:02d}.jsonl"
        if logpath.exists():logpath.rename(logpath.with_name(logpath.name+f".interrupted.{time.time_ns()}"))
        sums=Counter();diagnostic=Counter();audit=Counter();times=[];gradients=[];pending=0;started=time.perf_counter()
        epoch_scenes=reg["fit_scenes"] if a.phase=="selection" else reg["final_train_scenes"]
        epoch_times=sum(scope["splits"]["train"]["scenes"][str(s)]["timestamps"] for s in epoch_scenes)
        epoch_offset=sum(x["timestamps"] for x in state["epoch_scenes"])
        optimizer.zero_grad(set_to_none=True)
        with logpath.open("x") as log:
            for index,frame in enumerate(cache.frames()):
                obs,context=v5_observations(frame,cfg.use_geometry)
                labels,target_audit=match_train_identities(obs,ds.ground_truth_pair(pairs[frame["timestamp"]]))
                before_next_id=runner.core.bank.next_id
                result,losses=runner.forward_loss_then_step(obs,frame["timestamp"],context,labels,training=training)
                if training:
                    (losses["total"]/reg["gradient_accumulation_timestamps"]).backward();pending+=1
                    if pending==reg["gradient_accumulation_timestamps"]:
                        lr=learning_rate(reg,epoch,(epoch_offset+index+1)/epoch_times)
                        for group in optimizer.param_groups:group["lr"]=lr
                        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),reg["gradient_clip_norm"])
                        if not bool(torch.isfinite(norm)):raise ValueError("NONFINITE_V5_GRADIENT")
                        gradients.append(float(norm));optimizer.step();optimizer.zero_grad(set_to_none=True);state["optimizer_steps"]+=1;pending=0
                keys=list(losses);values=torch.stack([losses[k].detach() for k in keys]).cpu().tolist();sums.update(dict(zip(keys,values)))
                diagnostic.update(runner.last_diagnostics);audit.update(target_audit);times.append(frame["timestamp"])
                diagnostic.update(verified_world_camera_timestamps=int(bool(context.cameras)),missing_world_camera_timestamps=int(context.reprojection_status=="WORLD_FROM_OMEGA_NOT_RECORDED"))
                if index%64==0:
                    log.write(json.dumps({"event":"STEP","scene":sid,"timestamp":frame["timestamp"],"losses":dict(zip(keys,values)),
                        "diagnostics":runner.last_diagnostics,"bank":{k:v for k,v in result["bank"].items() if k not in ["stages"]},
                        "clip_boundary":runner.clip_boundary(),"previous_next_id":before_next_id,"GT_in_model":False},allow_nan=False)+"\n");log.flush()
            if training and pending:
                for param in model.parameters():
                    if param.grad is not None:param.grad.mul_(reg["gradient_accumulation_timestamps"]/pending)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),reg["gradient_clip_norm"])
                if not bool(torch.isfinite(norm)):raise ValueError("NONFINITE_V5_GRADIENT")
                for group in optimizer.param_groups:group["lr"]=learning_rate(reg,epoch,(epoch_offset+len(times))/epoch_times)
                gradients.append(float(norm));optimizer.step();optimizer.zero_grad(set_to_none=True);state["optimizer_steps"]+=1
        if times!=list(pairs) or len(times)!=expected["timestamps"]:raise RuntimeError("INCOMPLETE_V5_SCENE_TRAINING")
        if runner.core.scene_reset_count!=1:raise RuntimeError("V5_HISTORY_RESET_INSIDE_SCENE")
        return {"scene":sid,"timestamps":len(times),"first_timestamp":times[0],"last_timestamp":times[-1],
            "support_manifest_sha256":sha256(cache.path/"manifest.json"),"raw_manifest_sha256":expected["raw_manifest_sha256"],
            "loss_sums":dict(sums),"diagnostics":dict(diagnostic),"target_audit":dict(audit),"scene_reset_count":1,
            "gradient_norm_max":max(gradients,default=None),"gradient_finite":True,"wall_s":time.perf_counter()-started,
            "log_sha256":sha256(logpath),"final_next_id":runner.core.bank.next_id,"GT_repaired_model_IDs":False}

    if a.phase=="final":
        selected=json.loads((a.output/"selection/selection_gate.json").read_text())
        if (selected["epochs"]!=budget() or selected["initial_state_tensor_sha256"]!=initial_sha
                or selected["v5_source_sha256"]!=source or selected["scope_sha256"]!=sha256(a.scope_lock)):
            raise ValueError("FINAL_TRAIN_REQUIRES_COMPLETED_SAME_INITIALIZATION_SELECTION")
    if state["epoch"]<=budget():state["complete"]=False
    while state["epoch"]<=budget():
        epoch=state["epoch"]
        section="fit" if a.phase=="selection" else "train"
        wanted=reg["fit_scenes"] if a.phase=="selection" else reg["final_train_scenes"]
        order=np.random.default_rng(reg["seed"]+epoch).permutation(wanted).tolist()
        completed={x["scene"] for x in state["epoch_scenes"]}
        for sid in order:
            if sid in completed:continue
            record=run_scene(sid,epoch,True,section);state["epoch_scenes"].append(record);save_boundary()
            print(json.dumps({"event":"SCENE_COMPLETE","variant":a.variant,"phase":a.phase,"section":section,"epoch":epoch,"scene":sid,"timestamps":record["timestamps"],"wall_s":record["wall_s"]}),flush=True)
        training=summarize(state["epoch_scenes"])
        if a.phase=="selection":
            done={x["scene"] for x in state["epoch_validation_scenes"]}
            for sid in reg["validation_scenes"]:
                if sid in done:continue
                record=run_scene(sid,epoch,False,"validation");state["epoch_validation_scenes"].append(record);save_boundary()
            validation=summarize(state["epoch_validation_scenes"])
            if training["scene_count"]!=18 or validation["scene_count"]!=6 or training["frames"]+validation["frames"]!=28754:raise RuntimeError("V5_SELECTION_COVERAGE")
            row={"epoch":epoch,"train":training,"validation":validation,"fit_coverage":state["epoch_scenes"],"validation_coverage":state["epoch_validation_scenes"]}
        else:
            if training["scene_count"]!=24 or training["frames"]!=28754:raise RuntimeError("V5_FINAL_EPOCH_FULL24_COVERAGE")
            row={"epoch":epoch,"train":training,"coverage":state["epoch_scenes"]}
        state["history"].append(row);state["epoch"]+=1;state["epoch_scenes"]=[];state["epoch_validation_scenes"]=[]
        if a.phase=="selection" and a.variant=="B3" and epoch==reg["base_epochs"] and not a.common_budget.exists():
            extended=extension_allowed(state["history"])
            atomic_json(a.common_budget,{"status":"FROZEN_TRAIN_ONLY_BUDGET","epochs":reg["max_epochs"] if extended else reg["base_epochs"],
                "rule":reg["extension_rule"],"extension_allowed":extended,"decision_variant":"B3","TEST_read":False,
                "v5_source_sha256":source,"registered_config_sha256":sha256(registration_path()),"epoch12_validation":row["validation"]})
        atomic_json(phase_root/"history.json",state["history"]);save_boundary()
        print(json.dumps({"event":"EPOCH_COMPLETE","variant":a.variant,"phase":a.phase,"epoch":epoch,"train":training,
                          "validation":row.get("validation"),"current_budget":budget()}),flush=True)
    state["complete"]=True;save_boundary()
    if a.phase=="selection":
        atomic_json(phase_root/"selection_gate.json",{"status":"SELECTION_COMPLETE","variant":a.variant,"epochs":len(state["history"]),
            "initial_state_tensor_sha256":initial_sha,"v5_source_sha256":source,"registered_config_sha256":sha256(registration_path()),
            "scope_sha256":sha256(a.scope_lock),"full24_checkpoint":False,"TEST_read":False,"history":state["history"]})
        return
    epochs=budget()
    if len(state["history"])!=epochs or any(r["train"]["frames"]!=28754 for r in state["history"]):raise RuntimeError("V5_FULL_FINAL_BUDGET_REQUIRED")
    if any(sorted(x["scene"] for x in r["coverage"])!=list(range(1,25)) for r in state["history"]):raise RuntimeError("V5_EVERY_FINAL_EPOCH_MUST_COVER24")
    if any(r["train"]["target_audit"].get("matched_identity_observations",0)<=0 for r in state["history"]):raise RuntimeError("NO_VALID_IDENTITY_SUPERVISION")
    weights=a.output/"membership_v5.pt";temporary=weights.with_suffix(".partial")
    torch.save({"schema":model.configuration()["schema"],"model_config":model.configuration(),"state_dict":model.state_dict()},temporary);os.replace(temporary,weights)
    gate={"status":"FULL_TRAIN_GATE_PASSED","schema":"worldobject_membership_full24_gate_v5_v1","variant":a.variant,
        "checkpoint":weights.name,"checkpoint_sha256":sha256(weights),"model_config":model.configuration(),"tracking_config":cfg.to_dict(),
        "prediction_heads":model.configuration()["prediction_heads"],"seed":reg["seed"],"initial_state_tensor_sha256":initial_sha,
        "source_commit":source_commit(),"source_sha256":source_fingerprint(),"v5_source_sha256":source,
        "registered_config_sha256":sha256(registration_path()),"scope_sha256":sha256(a.scope_lock),"calibration_sha256":sha256(a.calibration),
        "common_budget_sha256":sha256(a.common_budget),"final_epochs":epochs,"final_scenes":list(range(1,25)),"final_timestamps":28754,
        "final_total_training_timestamps":28754*epochs,"final_epoch_coverage":[r["coverage"] for r in state["history"]],
        "history_mode":"predicted_from_first_timestamp","GT_in_model_inputs":False,"GT_repair_of_predicted_history":False,
        "clip_identity_reset":False,"optimizer":"AdamW","optimizer_steps":state["optimizer_steps"],"loss_weights":LOSS_WEIGHTS,
        "training_history":state["history"],"trainable_parameters":sum(p.numel() for p in model.parameters()),"approval":approval,
        "wall_s_this_process":time.perf_counter()-start}
    atomic_json(a.output/"train_gate.json",gate)
    print(json.dumps({"event":"FULL_TRAIN_GATE_PASSED","variant":a.variant,"checkpoint_sha256":gate["checkpoint_sha256"],"final_epochs":epochs}),flush=True)


if __name__=="__main__":main()
