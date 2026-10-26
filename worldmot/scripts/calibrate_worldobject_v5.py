#!/usr/bin/env python3
"""One fit18-only detector-quality calibration and cache/gauge availability audit."""
import argparse,json,time
from pathlib import Path
from collections import Counter
import numpy as np
from worldmot.data.visiontrack import VisionTrack
from worldmot.data.worldobject_cache import WorldObjectCache,sha256
from worldmot.models.gmt_detection_cache import ImmutableGMTDetectionCache
from worldmot.models.world_object_v5_geometry import v5_observations
from worldmot.training.world_object_identity_targets import match_train_identities
from worldmot.runtime.world_object_v5_contract import atomic_json,registration_path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mirror-root",required=True,type=Path);p.add_argument("--dataset-root",default="/data/DATASETS/TRACKING/JDE/VisionTrack",type=Path)
    p.add_argument("--input-mirror-receipt",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path);a=p.parse_args();reg=json.loads(registration_path().read_text())
    if a.output.exists():raise FileExistsError("Calibration is fixed once; inspect existing record")
    receipt=json.loads(a.input_mirror_receipt.read_text())
    if (receipt.get("status")!="TRAIN24_FROZEN_INPUTS_VERIFIED" or receipt.get("timestamps")!=28754
            or receipt.get("TRAIN_scenes")!=24 or Path(receipt["output"])!=a.mirror_root):
        raise ValueError("COMPLETE_IMMUTABLE_INPUT_MIRROR_REQUIRED_BEFORE_CALIBRATION")
    ds=VisionTrack(a.dataset_root,split="train",load_annotations=True)
    raw=ImmutableGMTDetectionCache(a.mirror_root/"gmt_detector_comparison_20261023/raw_train_v1")
    if raw.lock["split"]!="train":raise ValueError("NO_TEST_CALIBRATION")
    scores=[];per_scene=[];counts=Counter();started=time.perf_counter()
    for sid in reg["fit_scenes"]:
        cache=WorldObjectCache(a.mirror_root/"worldobject_v3_birthfix_r2_20261025/cache_train_v2",sid,raw_cache=raw)
        pairs={p.frame_id:p for p in ds.synchronized_pairs(sid)};seen=[];audit=Counter()
        for frame in cache.frames():
            obs,context=v5_observations(frame);labels,gt=match_train_identities(obs,ds.ground_truth_pair(pairs[frame["timestamp"]]))
            scores.extend(o.score for o,label in zip(obs,labels) if label is not None)
            audit.update(gt);audit.update(timestamps=1,candidates=len(obs),geometry_valid=sum(o.geometry_valid for o in obs),
                fusion_eligible=sum(o.cross_view_fusion_eligible for o in obs),verified_camera_timestamps=int(bool(context.cameras)),
                missing_world_camera_transform=int(context.reprojection_status=="WORLD_FROM_OMEGA_NOT_RECORDED"))
            seen.append(frame["timestamp"])
        if seen!=list(pairs):raise ValueError("CALIBRATION_FULL_SCENE_COVERAGE")
        per_scene.append({"scene":sid,**dict(audit),"support_manifest_sha256":sha256(cache.path/"manifest.json")});counts.update(audit)
        print(json.dumps({"calibration_scene_complete":sid,"timestamps":len(seen)}),flush=True)
    high=float(np.clip(np.percentile(scores,5),raw.lock["threshold"],.9))
    atomic_json(a.output,{"status":"FROZEN_TRAIN_ONLY_CALIBRATION","split":"train","calibration_scenes":reg["fit_scenes"],
        "birth_high_score":high,"rule":reg["birth_high_score_calibration"],"matched_score_quantiles":{str(q):float(np.percentile(scores,q)) for q in [0,5,25,50,75,95,100]},
        "counts":dict(counts),"per_scene":per_scene,"registered_config_sha256":sha256(registration_path()),"raw_detector_lock_sha256":sha256(raw.lock_path),
        "TRAIN_annotations_sha256":sha256(a.dataset_root/"annotations/train.json"),"input_mirror_receipt_sha256":sha256(a.input_mirror_receipt),
        "calibration_script_sha256":sha256(__file__),"TEST_read":False,"GT_3D_labels_used":False,"wall_s":time.perf_counter()-started})
    print(json.dumps({"status":"FROZEN_TRAIN_ONLY_CALIBRATION","birth_high_score":high,"counts":dict(counts)}),flush=True)


if __name__=="__main__":main()
