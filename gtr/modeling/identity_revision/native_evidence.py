"""Shared GT-free native evidence extraction for offline audit and live guard.

Inputs contain complete native candidates, descriptors and physical history.
Boxes must share the same image coordinate system within a camera.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from .conflict_detector import ConflictEvidence


def box_iou(a,b):
    upper=torch.maximum(a[:,None,:2],b[None,:,:2]);lower=torch.minimum(a[:,None,2:],b[None,:,2:])
    intersection=(lower-upper).clamp_min(0).prod(-1)
    areas_a=(a[:,2:]-a[:,:2]).clamp_min(0).prod(-1)
    areas_b=(b[:,2:]-b[:,:2]).clamp_min(0).prod(-1)
    return intersection/(areas_a[:,None]+areas_b[None,:]-intersection).clamp_min(1e-12)


def native_conflicts(query,query_boxes,frame,camera,history,history_boxes,history_ids,
                     history_frames,history_cameras,candidates,scores,accepted,
                     history_is_physical=None):
    if type(frame)!=int or frame<1 or type(camera)!=int or camera<1:raise ValueError('Physical camera/frame required')
    n,m=len(query),len(candidates);r=len(history)
    if query.shape!=(n,1152) or history.shape!=(r,1152) or scores.shape!=(n,m) or accepted.shape!=(n,):
        raise ValueError('Native complete query/candidate/1152 evidence contract')
    if query_boxes.shape!=(n,4) or history_boxes.shape!=(r,4) or any(x.shape!=(r,) for x in (history_ids,history_frames,history_cameras)):
        raise ValueError('Physical box/source metadata contract')
    if (history_frames>frame).any():raise ValueError('Future evidence rejected')
    if any(not torch.isfinite(x).all() for x in (query,history,query_boxes,history_boxes,scores)):
        raise ValueError('Nonfinite risk evidence')
    if len(candidates)!=len(torch.unique(candidates)) or (len(history_ids) and not (history_ids[:,None]==candidates[None,:]).any(1).all()):
        raise ValueError('Illegal native candidate/ID provenance')
    physical=torch.ones(r,dtype=torch.bool,device=history.device) if history_is_physical is None else history_is_physical
    if physical.shape!=(r,):raise ValueError('Descriptor-version provenance shape')
    q=F.normalize(query.float(),dim=-1);h=F.normalize(history.float(),dim=-1)
    past=history_frames<frame
    owners=(history_ids[past,None]==candidates[None,:]).float()
    prototypes=F.normalize(owners.T@h[past],dim=-1);cosines=q@prototypes.T
    counts=(history_ids[:,None]==candidates[None,:]).sum(0).clamp_min(1)
    normalized=scores/counts[None,:]
    previous=torch.nonzero((history_frames==frame-1)&(history_cameras==camera)&physical,as_tuple=True)[0]
    local={};overlap=box_iou(query_boxes,history_boxes[previous])
    if overlap.numel():
        rr,cc=linear_sum_assignment(-overlap.detach().cpu())
        local={int(i):int(previous[j]) for i,j in zip(rr,cc)}
    features=[];evidence=[]
    for row in range(n):
        identity=int(accepted[row]);column=torch.nonzero(candidates==identity,as_tuple=True)[0]
        selected=int(column[0]) if len(column) else None
        best_local=local.get(row)
        global_features=[0.]*12
        if selected is not None:
            competing=normalized[row].clone();competing[selected]=-float('inf')
            margin=float(normalized[row,selected]-competing.max()) if m>1 else 1.
            selected_cos=float(cosines[row,selected])
            selected_refs=(history_ids==identity)&past;same=(history_cameras==camera)&selected_refs
            global_features[0]=float(normalized[row,selected]);global_features[1]=margin;global_features[2]=selected_cos
            global_features[8]=float(same.sum()/selected_refs.sum().clamp_min(1));global_features[9]=float(torch.log1p(counts[selected])/5)
        if selected is None or best_local is None:
            evidence.append(ConflictEvidence(identity,-1,False,0,float('nan'),float('nan'),float('nan'),float('nan'),float('nan')))
            features.append(global_features);continue
        local_id=int(history_ids[best_local]);local_column=torch.nonzero(candidates==local_id,as_tuple=True)[0]
        local_cos=float(q[row]@h[best_local]);local_position=int(torch.nonzero(previous==best_local,as_tuple=True)[0][0])
        geometric=float(overlap[row,local_position])
        other_overlap=overlap[row].clone();other_overlap[local_position]=-1
        iou_margin=geometric-max(0.,float(other_overlap.max())) if other_overlap.numel()>1 else geometric
        consecutive=0
        for lag in range(1,4):
            if ((history_frames==frame-lag)&(history_cameras==camera)&(history_ids==local_id)&physical).any():consecutive+=1
            else:break
        competing=normalized[row].clone();competing[selected]=-float('inf')
        margin=float(normalized[row,selected]-competing.max()) if m>1 else 1.
        selected_cos=float(cosines[row,selected]);legal=bool(len(local_column))
        evidence.append(ConflictEvidence(identity,local_id,legal,consecutive,geometric,iou_margin,local_cos,selected_cos,margin))
        local_col=int(local_column[0]) if legal else selected
        selected_refs=(history_ids==identity)&past
        same=(history_cameras==camera)&selected_refs
        features.append([float(normalized[row,selected]),margin,selected_cos,local_cos,geometric,iou_margin,
            min(consecutive,3)/3,float(local_id==identity),float(same.sum()/selected_refs.sum().clamp_min(1)),
            float(torch.log1p(counts[selected])/5),float(normalized[row,local_col]-normalized[row,selected]),float(legal)])
    return evidence,torch.tensor(features,dtype=torch.float32,device=query.device).reshape(n,12)
