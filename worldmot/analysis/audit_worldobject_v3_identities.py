#!/usr/bin/env python3
"""Full-video posthoc fixed-candidate attribution; never an inference input.

All delivered methods must have complete immutable 22-scene predictions before
this program opens annotations. Pair counts are diagnostic (IoU 0.5 mapping),
and are never labelled official CVIDF1 or used to choose thresholds.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import numpy as np

from drive_worldobject_v3 import FULL,NG,HEAD,NOBIRTH,RULE,ROI,OFF,GEO,METHODS
from summarize_worldobject_v3 import sha,read,write_json

CONTRASTS=((FULL,NG),(FULL,NOBIRTH),(RULE,ROI),(FULL,RULE),(ROI,OFF),(FULL,HEAD))


def contrast(counts,proposed,baseline,kind):
    counts[kind+"_events"]+=1
    counts[kind+"_corrected" if proposed and not baseline else
           kind+"_new_errors" if baseline and not proposed else
           kind+"_unchanged_correct" if proposed else kind+"_unchanged_wrong"]+=1


def immutable_predictions(root):
    proofs={}
    for method in METHODS:
        scenes=[]
        for sid in range(1,23):
            directory=root/"predictions"/method/f"scene_{sid:02d}"
            manifest=read(directory/"manifest.json");path=directory/"tracks.jsonl"
            if manifest.get("status")!="COMPLETE" or manifest.get("method_id")!=method or sha(path)!=manifest["tracks_sha256"]:
                raise ValueError("COMPLETE_IMMUTABLE_METHOD_REQUIRED_BEFORE_POSTHOC_GT: "+str(path))
            scenes.append({"scene":sid,"timestamps":manifest["timestamps"],"tracks_sha256":manifest["tracks_sha256"]})
        if sum(s["timestamps"] for s in scenes)!=29019:raise ValueError("NOT_FULL_TEST")
        proofs[method]=scenes
    return proofs


def main():
    p=argparse.ArgumentParser();p.add_argument("--run-root",required=True,type=Path);p.add_argument("--output",required=True,type=Path)
    p.add_argument("--dataset-root",default="/data/DATASETS/TRACKING/JDE/VisionTrack");a=p.parse_args();root=a.run_root
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from worldmot.runtime.experiment_authorization import require_experiment_authorization
    from worldmot.data.visiontrack import VisionTrack
    from worldmot.data.worldobject_cache import WorldObjectCache
    from worldmot.data.worldobject_observations import replay_observations
    from worldmot.training.world_object_identity_targets import match_evaluation_identities,update_history_labels
    for method in METHODS:
        require_experiment_authorization(root/("authorization_v4.json" if method==GEO else "authorization_v3.json"),"diagnostics","test",method)
    proofs=immutable_predictions(root)  # No GT annotation access precedes this check.
    audit=read(root.parent/"fulltrain_20261022/audit/data_audit.json")["test"]
    legal={s["scene"] for s in audit["scenes"] if s["official_gt_legal"]}
    ds=VisionTrack(a.dataset_root,split="test",load_annotations=True)
    per_scene={};primary={m:Counter() for m in METHODS};blocked={m:Counter() for m in METHODS}
    effects={f"{x}_VS_{y}":Counter() for x,y in CONTRASTS};examples=defaultdict(list)
    matched_audit=Counter()
    for sid in range(1,23):
        counts={m:Counter() for m in METHODS};seen=set()
        states={m:{"active":set(),"history_labels":{},"previous":{},"last_id_time":{}} for m in METHODS}
        cache=WorldObjectCache(root/"cache_test_v2",sid,require_features=False,require_geometry=True)
        pairs={pair.frame_id:pair for pair in ds.synchronized_pairs(sid)}
        with ExitStack() as stack:
            streams={m:stack.enter_context((root/"predictions"/m/f"scene_{sid:02d}/tracks.jsonl").open()) for m in METHODS}
            for frame in cache.frames():
                timestamp=frame["timestamp"];obs=replay_observations(frame)
                labels,label_audit=match_evaluation_identities(obs,ds.ground_truth_pair(pairs[timestamp]))
                if sid in legal:matched_audit.update(label_audit)
                outputs={m:json.loads(next(streams[m])) for m in METHODS}
                assignments={};decisions={};known={}
                for method,row in outputs.items():
                    if row["timestamp"]!=timestamp:raise ValueError("NONCAUSAL_POSTHOC_JOIN")
                    objects={(int(v),o["detection_id"]):o for v,items in row["tracks_2d_by_camera"].items() for o in items}
                    if set(objects)!={o.key for o in obs}:raise ValueError("FIXED_CANDIDATE_COVERAGE_DIFFERENCE")
                    for o in obs:
                        if not np.array_equal(np.asarray(objects[o.key]["bbox_xywh"]),o.bbox_xywh):raise ValueError("FIXED_BOX_CHANGED")
                    assignments[method]={k:v["global_id"] for k,v in objects.items()}
                    decisions[method]={(x["camera_id"],x["detection_id"]):x for x in row.get("decisions",[])}
                    state=states[method]
                    known[method]={state["history_labels"].get(g) for g in state["active"]}-{None}
                    counts[method]["paired_timestamps"]+=1;counts[method]["current_detections"]+=len(obs)
                for left in range(len(obs)):
                    if labels[left] is None:continue
                    for right in range(left+1,len(obs)):
                        if labels[right] is None or obs[left].camera_id==obs[right].camera_id:continue
                        same=labels[left]==labels[right];first=same and labels[left] not in seen
                        outcomes={}
                        for method in METHODS:
                            ids=assignments[method];merged=ids[obs[left].key]==ids[obs[right].key]
                            result=merged==same;outcomes[method]=result;c=counts[method]
                            c["same_person_cross_view_pairs" if same else "different_person_cross_view_pairs"]+=1
                            c["correct_shared_identity" if same and merged else "same_person_split" if same else "false_cross_view_merge" if merged else "correctly_separate"]+=1
                            if first:
                                c["first_clean_detection_birth_pairs"]+=1;c["first_clean_detection_birth_splits"]+=int(not merged)
                            if same and labels[left] not in known[method]:
                                c["model_history_absent_same_person_pairs"]+=1;c["model_history_absent_pair_splits"]+=int(not merged)
                            if merged and not same:
                                reason=decisions[method].get(obs[left].key,{}).get("decision","")
                                c["false_merge_by_new_birth" if "BIRTH" in reason else "false_merge_by_history_or_legacy"]+=1
                            if not result:
                                kind="FIRST_CLEAN_BIRTH_SPLIT" if first else "SAME_PERSON_SPLIT" if same else "DIFFERENT_PERSON_MERGE"
                                key=(method,kind,sid in legal)
                                if len(examples[key])<6:
                                    examples[key].append({"scene":sid,"timestamp":timestamp,"method":method,"error":kind,
                                        "GT_scope_legal":sid in legal,"actors":[{"camera":obs[j].camera_id,"detection_id":obs[j].detection_id,
                                            "gt_identity":labels[j],"global_id":ids[obs[j].key],"bbox_xywh":obs[j].bbox_xywh.tolist(),
                                            "geometry_valid":obs[j].geometry_valid,"fusion_eligible":obs[j].cross_view_fusion_eligible,
                                            "reliability":obs[j].geometry_reliability,"decision":decisions[method].get(obs[j].key,{})} for j in (left,right)]})
                        if sid in legal:
                            for proposed,baseline in CONTRASTS:
                                c=effects[f"{proposed}_VS_{baseline}"]
                                contrast(c,outcomes[proposed],outcomes[baseline],"same_person" if same else "different_person")
                                if first:contrast(c,outcomes[proposed],outcomes[baseline],"first_clean_birth")
                for j,o in enumerate(obs):
                    label=labels[j]
                    if label is None:continue
                    outcomes={};event_key=(o.camera_id,label)
                    for method in METHODS:
                        state=states[method];gid=assignments[method][o.key]
                        prior=state["previous"].get(event_key)
                        if prior:
                            old_gid,old_time=prior;changed=old_gid!=gid;gap=timestamp-old_time
                            c=counts[method];c["clean_temporal_comparisons"]+=1;c["diagnostic_temporal_id_changes"]+=int(changed)
                            if gap>1:
                                suffix="within_30" if gap<=30 else "over_30"
                                c["occlusion_returns_"+suffix]+=1;c["occlusion_id_changes_"+suffix]+=int(changed)
                            outcomes[method]=not changed
                        state["previous"][event_key]=(gid,timestamp)
                    if sid in legal and len(outcomes)==len(METHODS):
                        for proposed,baseline in CONTRASTS:contrast(effects[f"{proposed}_VS_{baseline}"],outcomes[proposed],outcomes[baseline],"temporal_continuity")
                for method,row in outputs.items():
                    state=states[method];ids=assignments[method]
                    update_history_labels(state["history_labels"],ids,obs,labels)
                    for gid in ids.values():state["last_id_time"][gid]=timestamp
                    state["active"]={x["global_id"] for x in row["tracks_3d"]} if "tracks_3d" in row else {
                        gid for gid,t in state["last_id_time"].items() if timestamp-t<=30}
                seen.update(x for x in labels if x is not None)
            if any(stream.readline() for stream in streams.values()):raise ValueError("EXTRA_PREDICTION_FRAMES")
        per_scene[str(sid)]={"official_GT_legal":sid in legal,"methods":{m:dict(v) for m,v in counts.items()}}
        for method in METHODS:(primary if sid in legal else blocked)[method].update(counts[method])
        print(json.dumps({"posthoc_scene_complete":sid,"GT_legal":sid in legal}),flush=True)
    for c in effects.values():
        for kind in ("same_person","different_person","first_clean_birth","temporal_continuity"):
            c[kind+"_net_corrections"]=c[kind+"_corrected"]-c[kind+"_new_errors"]
    common={"source_analysis_sha256":sha(__file__),"GT_role":"POSTHOC_ONLY_AFTER_IMMUTABLE_PREDICTIONS",
        "official_metric":False,"posthoc_bbox_label_mapping_IoU":.5,"threshold_selection_on_TEST":False,
        "primary_scope":"14 original legal full TEST videos","blocked_scope":"8 full-inference videos; diagnostic only, excluded from primary counts",
        "prediction_proofs":proofs,"fixed_candidate_boxes_verified":True}
    report={**common,"schema":"worldobject_v3_fixed_candidate_identity_attribution_v1",
        "primary_legal14":{m:dict(v) for m,v in primary.items()},"blocked8_diagnostic_only":{m:dict(v) for m,v in blocked.items()},
        "matched_label_audit_legal14":dict(matched_audit),"per_scene":per_scene,
        "paired_counterfactuals_legal14":{k:dict(v) for k,v in effects.items()},
        "failure_cases":[case for cases in examples.values() for case in cases],
        "causal_limit":"Geometry flags in error examples are correlations. Geometry net corrections come from the Full-vs-NoGeometry fixed-candidate contrast.",
        "ROI_limit":"ROI restoration effect is new legacy ROI_ON minus new legacy FEATURE_OFF on the same corrected geometry; old W3 absolute deltas additionally include geometry/input repairs."}
    births={**common,"schema":"worldobject_v3_joint_birth_contribution_v1",
        "first_birth_definition":"First timestamp with any clean matched detection of this GT identity in the scene. Shared across methods; not a claim about physical first GT visibility.",
        "model_absent_definition":"Identity absent from cleanly labelled prior active predicted objects; legacy active set approximates the documented 30-step retention.",
        "TEST_legal14":{m:{k:v for k,v in counts.items() if "birth" in k or "false_merge" in k} for m,counts in primary.items()},
        "frozen_weight_birth_ablation":dict(effects[f"{FULL}_VS_{NOBIRTH}"]),"TRAIN_epoch_metrics":{}}
    for method in (FULL,NG,HEAD):
        gate=read(root/"train"/method/"train_gate.json")
        births["TRAIN_epoch_metrics"][method]={"gate_sha256":sha(root/"train"/method/"train_gate.json"),
            "selection_history":gate["selection_history"],"final_history":gate["final_history"]}
    a.output.mkdir(parents=True,exist_ok=True)
    write_json(a.output/"worldobject_v3_identity_error_attribution.json",report)
    write_json(a.output/"worldobject_v3_birth_pair_analysis.json",births)


if __name__=="__main__":main()
