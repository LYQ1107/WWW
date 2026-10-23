"""Actual point support -> exclusive per-camera assignment with NULL."""
from __future__ import annotations
from dataclasses import asdict,dataclass
import numpy as np


@dataclass
class PointAssociationSettings:
    minimum_visibility: float = .5
    minimum_confidence: float = .5
    minimum_valid_points: int = 3
    minimum_inside_points: int = 3
    minimum_support: float = .20
    maximum_reference_gap: int = 3
    lost_retention: int = 30
    points_per_person: int = 16

    def manifest(self):return asdict(self)


def point_support(xy,visibility,confidence,valid,reference_weights,detection,settings,
                  source_camera,target_camera,temporal_gap):
    xy=np.asarray(xy,np.float64).reshape(-1,2)
    vis=np.asarray(visibility,np.float64);conf=np.asarray(confidence,np.float64)
    valid=np.asarray(valid,bool);weights=np.asarray(reference_weights,np.float64)
    n=len(weights)
    if any(len(v)!=n for v in [xy,vis,conf,valid]):raise ValueError('Point evidence lengths differ')
    finite=np.isfinite(xy).all(1)&np.isfinite(vis)&np.isfinite(conf)&np.isfinite(weights)
    eligible=finite&valid&(vis>=settings.minimum_visibility)&(conf>=settings.minimum_confidence)&(weights>0)
    x,y,w,h=map(float,detection.bbox_xywh)
    inside=eligible&(xy[:,0]>=x)&(xy[:,0]<x+w)&(xy[:,1]>=y)&(xy[:,1]<y+h)
    denominator=float(np.where(np.isfinite(weights)&(weights>0),weights,0).sum())
    # The denominator includes all valid reference weights, not only points
    # surviving in this target view. Few surviving points cannot inflate votes.
    numerator=float(np.where(inside,weights*np.clip(vis,0,1)*np.clip(conf,0,1),0).sum())
    ratio=numerator/denominator if denominator>0 else 0.
    if temporal_gap>settings.maximum_reference_gap:reason='REFERENCE_OUTSIDE_HISTORY_WINDOW'
    elif int(eligible.sum())<settings.minimum_valid_points:reason='INSUFFICIENT_VALID_POINTS'
    elif int(inside.sum())<settings.minimum_inside_points:reason='NO_OVERLAP_OR_NO_POINT_SUPPORT'
    elif ratio<settings.minimum_support:reason='LOW_POINT_SUPPORT'
    else:reason=None
    return {'valid_point_count':int(eligible.sum()),'inside_box_count':int(inside.sum()),
        'point_support_ratio':ratio,'reference_weight_sum':denominator,
        'visibility_mean':float(vis[finite].mean()) if finite.any() else 0.,
        'confidence_mean':float(conf[finite].mean()) if finite.any() else 0.,
        'source_camera':int(source_camera),'target_camera':int(target_camera),
        'temporal_gap':int(temporal_gap),'association_score':ratio if reason is None else 0.,
        'rejection_reason':reason,'detection_id':int(detection.detection_id)}


def exclusive_assignment(scores,minimum):
    """One real candidate per row, one row per candidate; explicit NULL columns."""
    from scipy.optimize import linear_sum_assignment
    values=np.asarray(scores,np.float64)
    if values.ndim!=2:raise ValueError('Association scores must be a matrix')
    rows,cols=values.shape
    if not rows or not cols:return []
    good=np.isfinite(values)&(values>=minimum)
    cost=np.zeros((rows,cols+rows),np.float64)
    cost[:,:cols]=np.where(good,-values,1.)
    rr,cc=linear_sum_assignment(cost)
    return [(int(r),int(c)) for r,c in zip(rr,cc) if c<cols and good[r,c]]


def bidirectional_matches(forward,reverse,minimum):
    forward=np.asarray(forward,np.float64);reverse=np.asarray(reverse,np.float64)
    if forward.shape!=reverse.T.shape:raise ValueError('Directional camera matrices do not align')
    combined=np.where((forward>=minimum)&(reverse.T>=minimum),
                      np.sqrt(np.maximum(forward,0)*np.maximum(reverse.T,0)),0.)
    return [(a,b,float(combined[a,b])) for a,b in exclusive_assignment(combined,minimum)]
