"""TRAIN-only target construction. Identity labels never enter model inputs."""
from __future__ import annotations
from collections import Counter
import numpy as np


def identity_key(annotation):
    # COCO annotation id is NOT a person identity.
    value = annotation.get("instance_id", annotation.get("track_id"))
    return None if value is None else str(value)


def box_iou(a,b):
    a,b = np.asarray(a,float),np.asarray(b,float)
    lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[:2]+a[2:],b[:2]+b[2:])
    inter=float(np.prod(np.maximum(hi-lo,0)))
    return inter/max(1e-12,float(np.prod(a[2:])+np.prod(b[2:])-inter))


def match_train_identities(observations, gt_by_camera, *, split="train", iou_threshold=0.5):
    if split != "train":
        raise ValueError("IDENTITY_TARGETS_REQUIRE_TRAIN_SPLIT")
    return _match_identity_annotations(observations,gt_by_camera,iou_threshold)


def match_evaluation_identities(observations,gt_by_camera,iou_threshold=0.5):
    """Posthoc labels only; callers must never feed these into a TEST core."""
    return _match_identity_annotations(observations,gt_by_camera,iou_threshold)


def _match_identity_annotations(observations,gt_by_camera,iou_threshold):
    labels={};audit=Counter()
    for camera in sorted({o.camera_id for o in observations}):
        anns=list(gt_by_camera.get(camera,[]))
        counts=Counter(identity_key(a) for a in anns)
        clean=[];ambiguous=[]
        for a in anns:
            key=identity_key(a)
            valid=(key is not None and counts[key]==1 and not a.get("ignore",False)
                   and not a.get("iscrowd",False) and a.get("category_id",1)==1
                   and np.isfinite(a.get("bbox",[])).all() and len(a.get("bbox",[]))==4
                   and min(a["bbox"][2:])>0)
            (clean if valid else ambiguous).append(a)
            audit["valid_annotations" if valid else "ambiguous_or_invalid_annotations"]+=1
        obs=[o for o in observations if o.camera_id==camera]
        # Mask detections overlapping ambiguous labels before matching clean GT.
        masked={o.key for o in obs if any(len(a.get("bbox",[]))==4 and
                box_iou(o.bbox_xywh,a["bbox"])>=iou_threshold for a in ambiguous)}
        candidates=sorted(((box_iou(o.bbox_xywh,a["bbox"]),o.key,identity_key(a))
                           for o in obs if o.key not in masked for a in clean),key=lambda x:(-x[0],x[1],x[2]))
        used_o,used_id=set(),set()
        for overlap,key,ident in candidates:
            if overlap<iou_threshold:
                break
            if key in used_o or ident in used_id:
                continue
            labels[key]=ident;used_o.add(key);used_id.add(ident)
        audit["masked_detections"]+=len(masked)
    result=[labels.get(o.key) for o in observations]
    audit["matched_identity_observations"]=sum(label is not None for label in result)
    return result,dict(audit)


def build_identity_targets(history, observations, labels, history_labels, *, split="train", retention_steps=30):
    if split!="train":
        raise ValueError("TRAIN_TARGETS_FORBIDDEN_FOR_TEST")
    m,n=len(history),len(observations)
    qlabels=[history_labels.get(q.global_id) for q in history]
    # A corrupted predicted bank may have several IDs for one person. Mask
    # ambiguous history supervision rather than selecting one using future GT.
    counts=Counter(x for x in qlabels if x is not None)
    qvalid=np.array([x is not None and counts[x]==1 for x in qlabels],bool)
    mapping={x:i for i,x in enumerate(qlabels) if qvalid[i]}
    target=np.full(n,-1,np.int64)
    for j,label in enumerate(labels):
        if label is not None and (label not in counts or counts[label]==1):
            target[j]=mapping.get(label,m)
    same=[];different=[]
    for a in range(n):
        for b in range(a+1,n):
            if labels[a] is None or labels[b] is None or observations[a].camera_id==observations[b].camera_id:
                continue
            (same if labels[a]==labels[b] else different).append((a,b))
    positives=np.zeros((m,n),bool)
    for j,i in enumerate(target):
        if 0<=i<m:
            positives[i,j]=True
    return {"assignment":target,"history_valid":qvalid,"positive_pairs":positives,
            "birth":np.array([float(i==m) for i in target],np.float32),
            "observation_valid":target>=0,"null":~positives.any(axis=1),
            "existence":np.array([float(positives[i].any() or q.missed+1<=retention_steps) for i,q in enumerate(history)],np.float32),
            "same_pairs":same,"different_pairs":different}


def update_history_labels(history_labels, assignments, observations, labels):
    """Post-decision diagnostic sidecar; never used as a feature or assignment input."""
    grouped={}
    for o,label in zip(observations,labels):
        if label is not None:
            grouped.setdefault(assignments[o.key],set()).add(label)
    for gid,values in grouped.items():
        old=history_labels.get(gid)
        if old is not None:
            values.add(old)
        history_labels[gid]=next(iter(values)) if len(values)==1 else None
