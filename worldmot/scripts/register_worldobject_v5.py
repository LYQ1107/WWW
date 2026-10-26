#!/usr/bin/env python3
"""Register private V5 TRAIN/TEST scopes bound to the current human instruction."""
import argparse,json,os
from pathlib import Path
from worldmot.runtime.world_object_v5_contract import (SCHEMA,BASE,FULL_SHA,registration_path,v5_source_fingerprint,atomic_json)
from worldmot.runtime.experiment_authorization import APPROVAL_SCHEMA,source_fingerprint,source_commit
from worldmot.tracking.world_object_v5_config import registered_config
from worldmot.data.worldobject_cache import sha256


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=["train","test"],required=True)
    for name in ["output-root","mirror-root","calibration","request-file","mirror-receipt"]:p.add_argument("--"+name,required=True,type=Path)
    a=p.parse_args();a.output_root.mkdir(parents=True,exist_ok=True);private=a.output_root/"provenance";private.mkdir(exist_ok=True)
    # Never write large assets into the local directory hidden below an absent
    # NAS mount. A caller can explicitly use another independent mounted root.
    nas=Path('/data1/liuyeqiang/nas/A100')
    if str(a.output_root).startswith(str(nas)) and not os.path.ismount(nas):raise RuntimeError("NAS_MOUNT_REQUIRED")
    calibration=json.loads(a.calibration.read_text());reg=json.loads(registration_path().read_text());receipt=json.loads(a.mirror_receipt.read_text())
    if receipt["status"]!="TRAIN24_FROZEN_INPUTS_VERIFIED" or receipt["timestamps"]!=28754:raise ValueError("VERIFIED_FULL_TRAIN_INPUTS_REQUIRED")
    if calibration["calibration_scenes"]!=reg["fit_scenes"] or calibration["TEST_read"]:raise ValueError("TRAIN_ONLY_CALIBRATION_REQUIRED")
    scope={"schema":SCHEMA,"base_commit":BASE,"source_commit":source_commit(),"source_sha256":source_fingerprint(),
        "v5_source_sha256":v5_source_fingerprint(),"registered_config_sha256":sha256(registration_path()),
        "calibration_sha256":sha256(a.calibration),"input_mirror_receipt_sha256":sha256(a.mirror_receipt),
        "dataset_root":"/data/DATASETS/TRACKING/JDE/VisionTrack","variants":reg["variants"],"splits":{},
        "variant_configs":{v:registered_config(v,calibration).to_dict() for v in ["B1","B2","B3","B4"]},
        "TEST_parameter_selection":False,"GT_in_model_inputs":False,"original_R2_scopes_modified":False,
        "human_instruction_sha256":sha256(a.request_file),"frozen_C1_checkpoint_sha256":FULL_SHA}
    names=["train"] if a.stage=="train" else ["train","test"]
    for split in names:
        support=a.mirror_root/f"worldobject_v3_birthfix_r2_20261025/cache_{split}_v2"
        raw=a.mirror_root/f"gmt_detector_comparison_20261023/raw_{split}_v1";scenes={}
        count=24 if split=="train" else 22
        for sid in range(1,count+1):
            s=support/f"scene_{sid:02d}/manifest.json";r=raw/f"scene_{sid:02d}/manifest.json"
            sm=json.loads(s.read_text());rm=json.loads(r.read_text())
            if sm["status"]!="COMPLETE" or rm["status"]!="COMPLETE" or sm["timestamps"]!=rm["frames"] or sm["counts"]["observations"]!=rm["observations"]:raise ValueError("INCOMPLETE_REGISTERED_SCENE")
            scenes[str(sid)]={"support_manifest_sha256":sha256(s),"raw_manifest_sha256":sha256(r),"timestamps":sm["timestamps"],"candidates":sm["counts"]["observations"]}
        expected=28754 if split=="train" else 29019
        if sum(v["timestamps"] for v in scenes.values())!=expected:raise ValueError("REGISTERED_TIMESTAMP_COUNT")
        scope["splits"][split]={"support_cache":str(support),"raw_cache":str(raw),"raw_lock_sha256":sha256(raw/"DETECTOR_LOCK.json"),"scenes":scenes,"timestamps":expected,"candidates":sum(x["candidates"] for x in scenes.values())}
    if a.stage=="test":
        if scope["splits"]["test"]["candidates"]!=548583:raise ValueError("FROZEN_TEST_CANDIDATE_COUNT")
        budget=json.loads((private/"common_train_budget.json").read_text());gates={};initial=set()
        for variant in ["B2","B3","B4"]:
            gatepath=a.output_root/f"train/{variant}/train_gate.json";g=json.loads(gatepath.read_text())
            if (g["status"]!="FULL_TRAIN_GATE_PASSED" or g["final_epochs"]!=budget["epochs"] or g["final_timestamps"]!=28754
                    or g["v5_source_sha256"]!=v5_source_fingerprint() or g["final_scenes"]!=list(range(1,25))):raise ValueError("ALL_THREE_FULL24_GATES_REQUIRED_BEFORE_TEST")
            if sha256(gatepath.parent/g["checkpoint"])!=g["checkpoint_sha256"]:raise ValueError("FINAL_WEIGHT_CHANGED")
            initial.add(g["initial_state_tensor_sha256"]);gates[variant]={"train_gate":str(gatepath),"train_gate_sha256":sha256(gatepath),"checkpoint_sha256":g["checkpoint_sha256"],"epochs":g["final_epochs"]}
        if len(initial)!=1:raise ValueError("TRAINED_MODELS_MUST_SHARE_INITIALIZATION")
        scope["final_weights"]=gates;scope["common_train_budget_sha256"]=sha256(private/"common_train_budget.json")
        scope["original_C1_reference_predictions"]="/data1/liuyeqiang/worldmot_v4_c1_fulltest_outputs_20261026/predictions/V4_C1"
    lock=private/f"scope_v5_{a.stage}.json"
    if lock.exists():raise FileExistsError("Never overwrite a registered V5 scope")
    atomic_json(lock,scope)
    approval={"schema":APPROVAL_SCHEMA,"approved":True,"user_authorization_reference":str(a.request_file)+" SHA256 "+sha256(a.request_file),
        "source_commit":source_commit(),"source_sha256":source_fingerprint(),"v5_source_sha256":v5_source_fingerprint(),"scope_lock_sha256":sha256(lock),
        "operations":["train","replay","evaluate","diagnostics","preflight","live"] if a.stage=="train" else ["replay","evaluate","diagnostics","live"],
        "splits":names,"method_ids":["V5_"+v for v in reg["variants"]],"TRAIN_membership_only":True,"TEST_parameter_selection":False,
        "finite_variants":reg["variants"],"max_epochs":16,"no_multiseed_search":True,"original_GT_modified":False}
    auth=private/f"authorization_v5_{a.stage}.json";atomic_json(auth,approval)
    print(json.dumps({"status":"V5_SCOPE_REGISTERED","stage":a.stage,"source":v5_source_fingerprint(),"scope_sha256":sha256(lock),"authorization_sha256":sha256(auth)}),flush=True)


if __name__=="__main__":main()
