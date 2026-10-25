"""Posthoc V3 error attribution on immutable, already delivered predictions."""
from __future__ import annotations
from collections import Counter
import argparse
import json
from pathlib import Path
from ..runtime.experiment_authorization import require_experiment_authorization


def identity_diagnostics(observations,labels,assignments,previous):
    counts=Counter();failures=[]
    groups={}
    for o,label in zip(observations,labels):
        counts["current_detections"]+=1
        if label is None:
            counts["unmatched_or_ambiguous_detections"]+=1
            continue
        gid=assignments[o.key]
        groups.setdefault(gid,set()).add(label)
        key=(o.camera_id,label)
        if key in previous and previous[key]!=gid:
            counts["diagnostic_temporal_switches"]+=1
        previous[key]=gid
        counts["identity_observations"]+=1
    for gid,identities in groups.items():
        if len(identities)>1:
            counts["wrong_world_object_merges"]+=1
            failures.append({"global_id":gid,"reason":"DIFFERENT_PEOPLE_SAME_WORLD_OBJECT"})
    for a,left in enumerate(observations):
        for b in range(a+1,len(observations)):
            right=observations[b]
            if left.camera_id==right.camera_id or labels[a] is None or labels[b] is None:
                continue
            same=labels[a]==labels[b]
            merged=assignments[left.key]==assignments[right.key]
            counts["same_person_cross_view_pairs" if same else "different_person_cross_view_pairs"]+=1
            counts["correct_shared_identity" if same and merged else
                   "same_person_split" if same else "false_cross_view_merge" if merged else "correctly_separate"]+=1
    return dict(counts),failures


def main(kind="all"):
    p=argparse.ArgumentParser(description="V3 posthoc diagnostics; no inference/official metrics")
    p.add_argument("--authorization");p.add_argument("--cache",required=True)
    p.add_argument("--predictions",required=True);p.add_argument("--dataset-root",required=True)
    p.add_argument("--output",required=True);p.add_argument("--split",choices=["train","test"],required=True)
    p.add_argument("--method-id",required=True)
    a=p.parse_args();require_experiment_authorization(a.authorization,"diagnostics",a.split,a.method_id)
    from ..data.visiontrack import VisionTrack
    from ..data.worldobject_cache import WorldObjectCache,sha256
    from ..data.worldobject_observations import replay_observations
    from ..training.world_object_identity_targets import match_evaluation_identities
    ds=VisionTrack(a.dataset_root,split=a.split,load_annotations=True)
    total=Counter();failures=[];per_scene={}
    for sid in ds.scene_ids():
        directory=Path(a.predictions)/f"scene_{sid:02d}"
        manifest=json.loads((directory/"manifest.json").read_text())
        track_path=directory/"tracks.jsonl"
        if manifest.get("status")!="COMPLETE" or sha256(track_path)!=manifest["tracks_sha256"]:
            raise ValueError("DIAGNOSTICS_REQUIRE_IMMUTABLE_COMPLETE_PREDICTIONS")
        if manifest.get("method_id")!=a.method_id:
            raise ValueError("DIAGNOSTIC_METHOD_MISMATCH")
        cache=WorldObjectCache(a.cache,sid,require_features=False)
        pairs={pair.frame_id:pair for pair in ds.synchronized_pairs(sid)}
        previous={};summary=Counter()
        with track_path.open() as stream:
            for frame in cache.frames():
                row=json.loads(next(stream))
                if row["timestamp"]!=frame["timestamp"]:
                    raise ValueError("DIAGNOSTIC_TIMESTAMP_MISMATCH")
                observations=replay_observations(frame)
                labels,_=match_evaluation_identities(observations,ds.ground_truth_pair(pairs[frame["timestamp"]]))
                assignments={(int(v),obj["detection_id"]):obj["global_id"]
                             for v,objects in row["tracks_2d_by_camera"].items() for obj in objects}
                if set(assignments)!={o.key for o in observations}:
                    raise ValueError("DIAGNOSTIC_CANDIDATE_COVERAGE_MISMATCH")
                counts,errors=identity_diagnostics(observations,labels,assignments,previous)
                summary.update(counts)
                failures.extend({"scene":sid,"timestamp":frame["timestamp"],**e} for e in errors)
            if stream.readline():
                raise ValueError("EXTRA_PREDICTION_TIMESTAMPS")
        total.update(summary);per_scene[str(sid)]=dict(summary)
    with Path(a.output).open("x") as stream:
        json.dump({"schema":"worldobject_posthoc_diagnostics_v3","kind":kind,"split":a.split,
            "method_id":a.method_id,"summary":dict(total),"per_scene":per_scene,"failures":failures,
            "official_metric":False,"GT_role":"POSTHOC_ONLY","threshold_selection_on_TEST":False},stream,indent=2)
