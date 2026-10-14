"""Read-only cache admission diagnostics; never changes selected features."""
import math
import torch
from torch.nn import functional as F


def match_gates(cache,query):
    if cache is None or not len(query.reid_features):return dict(status='NO_CACHE_OR_QUERY')
    cameras=query.geo_camera.unique();frames=query.geo_frame.unique()
    if len(cameras)!=1 or len(frames)!=1:return dict(status='MIXED_QUERY_OBSERVATIONS')
    entry=cache.entries.get(int(cameras[0]));frame=int(frames[0])
    if entry is None:return dict(status='NO_RETAINED_SOURCE')
    age=frame-entry['frame']
    if age<=0 or age>cache.max_age:
        return dict(status='NOT_A_CURRENT_REUSE_EVENT',source_frame=entry['frame'],query_frame=frame)
    native=query.reid_features;boxes=query.geo_xyxy;n=len(native)
    cosine=F.normalize(torch.nan_to_num(native.float()),dim=1)@entry['native'].T
    finite=torch.isfinite(native).all(1)&torch.isfinite(boxes).all(1)&(native.norm(dim=1)>0)
    left=torch.maximum(boxes[:,None,:2],entry['boxes'][None,:,:2]);right=torch.minimum(boxes[:,None,2:],entry['boxes'][None,:,2:])
    intersection=(right-left).clamp_min(0).prod(2)
    area=(boxes[:,2:]-boxes[:,:2]).clamp_min(0).prod(1)
    source_area=(entry['boxes'][:,2:]-entry['boxes'][:,:2]).clamp_min(0).prod(1)
    iou=intersection/(area[:,None]+source_area[None,:]-intersection).clamp_min(1e-8)
    spatial=(iou>=cache.min_iou)&finite[:,None]
    def inspect(scores,filtered):
        top=scores.topk(min(2,scores.shape[1]),dim=1);indices=top.indices[:,0];best=top.values[:,0]
        margin=(best-top.values[:,1])/best.abs().clamp_min(1e-8) if top.values.shape[1]>1 else torch.ones_like(best)
        if filtered and top.values.shape[1]>1:margin=torch.where(torch.isfinite(top.values[:,1]),margin,torch.ones_like(margin))
        mutual=scores.argmax(0)[indices]==torch.arange(n,device=scores.device)
        candidate_iou=iou[torch.arange(n,device=scores.device),indices]
        masks=dict(finite_nonzero=finite,cosine=torch.isfinite(best)&(best>=cache.min_cosine),
                   relative_margin=margin>=cache.min_margin,mutual=mutual,iou=candidate_iou>=cache.min_iou)
        accepted=torch.stack(list(masks.values())).all(0)
        def values(tensor):return [float(value) if math.isfinite(float(value)) else None for value in tensor.detach().cpu()]
        return dict(source_indices=indices.detach().cpu().tolist(),cosine=values(best),relative_margin=values(margin),
                    bbox_iou=values(candidate_iou),gate_pass={key:value.detach().cpu().tolist() for key,value in masks.items()},
                    accepted=accepted.detach().cpu().tolist())
    original=inspect(cosine,False)
    filtered=inspect(cosine.masked_fill(~spatial,float('-inf')),True)
    actual=query.geo_valid.detach().cpu().tolist()
    return dict(status='CURRENT_REUSE_MATCH_RECOMPUTED',source_frame=entry['frame'],query_frame=frame,source_age=age,
                original=original,actual_query_valid=actual,original_matches_actual_valid=original['accepted']==actual,
                spatially_admissible_counts=spatial.sum(1).detach().cpu().tolist(),
                spatial_filter_counterfactual=filtered,
                scope='native descriptors and source/current boxes only; spatial-filter result is an admission counterfactual, not features used by tracker or tracking accuracy; no GT and no cache state mutation')
