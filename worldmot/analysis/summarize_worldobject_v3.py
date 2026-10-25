#!/usr/bin/env python3
"""Build honest progress/final tables from immutable measured artifacts only.

This posthoc program is outside the model/input source fingerprint. It does not
launch inference, open GT, select parameters, or modify any experiment asset.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

METHODS=("FULL_WORLD_OBJECT_V3","FULL_NO_GEOMETRY_V3","HEAD_ONLY_NO_GEOMETRY_V3",
         "NO_LEARNED_JOINT_BIRTH_V3","RULE_BASED_WORLD_OBJECT_V3",
         "LEGACY_PAIR_THEN_MERGE_ROI_RESTORED_CONTROL","LEGACY_PAIR_THEN_MERGE_FEATURE_OFF_CONTROL",
         "GEOMETRY_ONLY_ABLATION_V3")
REFERENCES={
    "GMT_NATIVE_OFFICIAL":"fulltrain_20261022/test_evaluation/gmt_native/metrics.json",
    "GMTDET_V2_PRIMARY":"gmt_detector_comparison_20261023/test_gmt_v2_evaluation/joint_seed20261022_legal14_complete/metrics.json",
    "GMTDET_V1":"gmt_detector_comparison_20261023/evaluation_gmt_v1_v2/metrics.json",
}
REFERENCE_PREDICTIONS={
    "GMT_NATIVE_OFFICIAL":"fulltrain_20261022/test/gmt_native",
    "GMTDET_V2_PRIMARY":"gmt_detector_comparison_20261023/test_gmt_v2/joint_seed20261022",
    "GMTDET_V1":"gmt_detector_comparison_20261023/test_gmt_v1_v2",
}
METRICS=("HOTA","AssA","DetA","IDF1","CVIDF1","CVMA","MOTA","IDSW","Frag","TP","FP","FN")
LEGAL_NAMES={"00001garden","00003garden","00005garden","00012night","00019court2","00022gate",
             "00023court4","00027square","00028path","00029path","00117wood","00120park","00144road","00146bridge"}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")


def write_csv(path,rows,columns):
    with Path(path).open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction="ignore")
        writer.writeheader();writer.writerows(rows)


def official(path):
    data=read(path)
    if data.get("custom_metrics_used") is not False or data.get("raw_GT_modified") is not False:
        raise ValueError("UNVERIFIED_OFFICIAL_PROTOCOL: "+str(path))
    if set(data.get("evaluated_scenes",[]))!=LEGAL_NAMES:
        raise ValueError("OFFICIAL_14_SCENE_SCOPE_MISMATCH: "+str(path))
    track=data.get("TrackEval",{}).get("combined",{})
    native=data.get("VisionTrack_MATLAB",{}).get("combined",{})
    values={k:track[k] for k in METRICS if k in track}
    values.update({k:native[k] for k in ("CVIDF1","CVMA") if k in native})
    values.update({k:track[v] for k,v in (("TP","CLR_TP"),("FP","CLR_FP"),("FN","CLR_FN")) if v in track})
    if all(k in values for k in ("TP","FP","FN")):
        values["CLEAR_Precision"]=100*values["TP"]/max(1,values["TP"]+values["FP"])
        values["CLEAR_Recall"]=100*values["TP"]/max(1,values["TP"]+values["FN"])
    return {"status":data["status"],"official_scenes":len(LEGAL_NAMES),**values,
            "metrics_source":str(path),"metrics_sha256":sha(path)}


def coverage(root):
    scenes=[]
    for directory in sorted(Path(root).glob("scene_*")):
        manifest=directory/"manifest.json"
        if not manifest.is_file():continue
        data=read(manifest)
        scenes.append({"scene":int(directory.name.split("_")[1]),"status":data.get("status"),
            "timestamps":data.get("timestamps",data.get("frames",0)),"manifest_sha256":sha(manifest)})
    done=[s for s in scenes if s["status"]=="COMPLETE"]
    return {"completed_scenes":len(done),"completed_timestamps":sum(s["timestamps"] for s in done),"scenes":scenes}


def reference_speed(root,method):
    directory=root.parent/REFERENCE_PREDICTIONS[method]
    data=[read(directory/f"scene_{sid:02d}/manifest.json") for sid in range(1,23)]
    if any(x.get("status")!="COMPLETE" or x.get("full_video") is not True or x.get("GT_in_inference") is not False for x in data):
        raise ValueError("HISTORICAL_SPEED_NOT_FULL_CAUSAL_INFERENCE")
    frames=sum(x["frames"] for x in data);seconds=sum(x["real_e2e_seconds"] for x in data)
    if frames!=29019 or seconds<=0:raise ValueError("HISTORICAL_SPEED_COVERAGE_MISMATCH")
    scope="HISTORICAL_ALL_22_TEST_FULL_VIDEO_WALL_THROUGHPUT"
    return {"historical_E2E_FPS":frames/seconds,"historical_timing_scope":scope}, {
        "method":method,"measurement":scope,"stage":"full_video_delivery","paired_timestamps":frames,"total_s":seconds,
        "serial_E2E_FPS":frames/seconds,"scope":scope,
        "notes":"Historical complete-video throughput. Native GMT delivers final IDs at video return; no per-timestamp latency distribution is claimed. Current methods use the registered 300-pair live sample."}


def sampled_memory(root,method):
    job=root/"jobs"/("live_"+method+".status.json");path=root/"logs/resource_samples.csv"
    if not job.is_file() or not path.is_file():return {}
    status=read(job)
    if status.get("status")!="COMPLETE" or time.time()-status["finished_at"]<10:return {}
    artifact=root/"diagnostics"/("sampled_memory_"+method+".json")
    if artifact.is_file():return read(artifact)
    peak=0;count=0
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if row["gpu_uuid"]!=status["gpu_uuid"] or not row["device_used_bytes"]:continue
            if status["started_at"]<=float(row["time_unix"])<=status["finished_at"]:
                peak=max(peak,int(row["device_used_bytes"]));count+=1
    result={"sampled_peak_device_bytes":peak if count else None,"memory_samples":count,"gpu_uuid":status["gpu_uuid"],
        "memory_definition":"Observed total device allocation, including GMT child context; target 0.2-second sampling, not an exact allocation-event peak."}
    write_json(artifact,result);return result


def live_timing(root,method):
    values={};manifests=[]
    for sid in (1,8,22):
        directory=Path(root)/"live"/method/f"scene_{sid:02d}"
        if not (directory/"manifest.json").is_file():return {},[]
        manifest=read(directory/"manifest.json")
        if manifest.get("detector_is_frozen_replay") is not False:
            raise ValueError("CACHE_DETECTOR_IS_NOT_TRUE_LIVE_FPS")
        if manifest["timestamps"]!=100 or sha(directory/"tracks.jsonl")!=manifest["tracks_sha256"]:
            raise ValueError("LIVE_TIMING_SCOPE_OR_OUTPUT_SHA_MISMATCH")
        rows=[json.loads(line) for line in (directory/"timings.jsonl").read_text().splitlines()]
        if len(rows)!=100:raise ValueError("INCOMPLETE_LIVE_TIMING")
        manifests.append(manifest)
        for row in rows:
            stages={k:row[k] for k in ("rgb_read_s","output_s","live_rgb_to_id_s") if k in row}
            stages.update(row["stages"])
            detector=row.get("detector_stages",{})
            forward=[value for key,value in detector.items() if key.startswith("camera_") and key.endswith("_detector_s")]
            if forward:stages["GMT_detector_model_and_native_postprocess_s"]=sum(forward)
            for key,value in stages.items():
                if not np.isfinite(value) or value<0:raise ValueError("INVALID_LIVE_LATENCY")
                values.setdefault(key,[]).append(float(value))
    duration=values["live_rgb_to_id_s"]
    result={"live_E2E_FPS":len(duration)/sum(duration),"live_timing_pairs":len(duration),
            "live_timing_scope":"TEST_SCENES_1_8_22_FIRST_100_EACH_INCLUDING_FIRST_FRAME",
            "live_model_initialization_included":False}
    output=[]
    for stage,times in sorted(values.items()):
        array=np.asarray(times,dtype=float)
        output.append({"method":method,"measurement":"TRUE_RGB_GMT_TO_DELIVERED_ID",
            "stage":stage,"paired_timestamps":len(times),"total_s":float(array.sum()),
            "p50_s":float(np.quantile(array,.5)),"p95_s":float(np.quantile(array,.95)),
            "mean_s":float(array.mean()),"equivalent_stage_FPS":len(times)/array.sum() if array.sum()>0 else None,
            "serial_E2E_FPS":result["live_E2E_FPS"] if stage=="live_rgb_to_id_s" else None,
            "scope":result["live_timing_scope"],"notes":"Nested components overlap; do not sum Ω model/preprocess with Ω wall, or solver/update with combined core."})
    memory=sampled_memory(Path(root),method);result.update(memory)
    for row in output:row.update(memory)
    return result,output


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run-root",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);root=args.run_root
    rows=[];runtime=[];train={};test={}
    for method,relative in REFERENCES.items():
        path=root.parent/relative
        speed,stage=reference_speed(root,method);runtime.append(stage)
        rows.append({"method":method,"result_type":"HISTORICAL_OFFICIAL_REFERENCE",**official(path),**speed})
    preflight=read(root/"diagnostics/preflight_attempt01/report.json")
    for method in METHODS:
        row={"method":method,"result_type":"REGISTERED_CURRENT_RUN","status":"PENDING_OFFICIAL_TEST"}
        path=root/"evaluation"/method/"metrics.json"
        if path.is_file():row.update(official(path))
        measured=coverage(root/"predictions"/method);test[method]=measured
        row.update(test_scenes_inferred=measured["completed_scenes"],test_timestamps=measured["completed_timestamps"])
        if path.is_file() and measured["completed_scenes"]!=22:
            raise ValueError("FORMAL_CURRENT_METHOD_REQUIRES_ALL_22_INFERRED_SCENES")
        row["association_parameters"]=preflight["identity_model_parameters"] if method in METHODS[:4] else 0
        row["frozen_omega_parameters"]=0 if method=="FULL_NO_GEOMETRY_V3" else preflight["frozen_omega_parameters"]
        gate=root/"train"/method/"train_gate.json"
        progress=root/"train"/method/"progress.json"
        if gate.is_file():
            data=read(gate)
            train[method]={"status":data["status"],"actual_final_timestamps":data["final_timestamps"],
                "actual_final_scenes":data["final_scenes"],"coverage":data["coverage"],
                "checkpoint_sha256":data["checkpoint_sha256"],"train_gate_sha256":sha(gate),
                "selected_final_epochs":data["training_budget"]["selected_final_epochs"],
                "source_sha256":data["source_sha256"],"TRAIN_only_selection":True}
        elif progress.is_file():train[method]={"status":"TRAINING_NOT_FULL_GATE_PASSED",**read(progress)}
        elif method in METHODS[:3]:train[method]={"status":"NOT_YET_COMPLETE","actual_final_timestamps":0}
        elif method=="NO_LEARNED_JOINT_BIRTH_V3":train[method]={"status":"INFERENCE_ABLATION_REUSES_FULL_TRAINED_CHECKPOINT","separate_training":False}
        else:train[method]={"status":"PARAMETER_FREE_CONTROL","neural_optimizer_training_required":False,
            "actual_full_TRAIN_causal_replay":coverage(root/"train_replay"/method)}
        speed,stages=live_timing(root,method);row.update(speed);runtime.extend(stages);rows.append(row)
    columns=["method","result_type","status","official_scenes","test_scenes_inferred","test_timestamps",*METRICS,
        "CLEAR_Precision","CLEAR_Recall","live_E2E_FPS","live_timing_pairs","live_timing_scope",
        "historical_E2E_FPS","historical_timing_scope","sampled_peak_device_bytes","gpu_uuid",
        "association_parameters","frozen_omega_parameters","metrics_source","metrics_sha256"]
    write_csv(args.output/"worldobject_v3_final_comparison.csv",rows,columns)
    write_csv(args.output/"worldobject_v3_runtime_breakdown.csv",runtime,["method","measurement","stage","paired_timestamps",
        "total_s","p50_s","p95_s","mean_s","equivalent_stage_FPS","serial_E2E_FPS","scope","notes",
        "sampled_peak_device_bytes","gpu_uuid","memory_samples","memory_definition"])
    write_json(args.output/"worldobject_v3_fulltrain_coverage.json",{"schema":"worldobject_v3_actual_training_coverage_v1",
        "expected_scenes":24,"expected_timestamps_per_final_epoch":28754,"expected_RGB_images":57508,
        "important":"A manifest/cache scan or a completed selection epoch is not full final training.","methods":train})
    write_json(args.output/"worldobject_v3_fulltest_coverage.json",{"schema":"worldobject_v3_actual_TEST_coverage_v1",
        "expected_scenes":22,"expected_timestamps":29019,"expected_RGB_images":58038,
        "official_legal_scenes":14,"GT_blocked_scenes":[5,12,13,14,17,20,21,22],"methods":test})
    print(json.dumps({"rows":len(rows),"current_official_completed":sum(r["result_type"]=="REGISTERED_CURRENT_RUN" and r["status"]=="COMPLETE_VALID_SCOPE" for r in rows),"live_stage_rows":len(runtime)}))


if __name__=="__main__":main()
