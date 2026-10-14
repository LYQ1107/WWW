"""Masked neural patch matches with explicit sampling precision and validity.

Patch-centre correspondences fit a candidate F/H only. Independent pixel-level
background matches must validate it before it can be used as geometry evidence.
"""
import cv2, numpy as np, torch
from .geometry_alignment import symmetric_epipolar_distance, symmetric_transfer_distance

def background_descriptors(feature_map, mapping, boxes):
    _, channels, height, width = feature_map.shape
    device = feature_map.device
    ys, xs = torch.meshgrid((torch.arange(height, device=device)+.5)*mapping.patch,
                           (torch.arange(width, device=device)+.5)*mapping.patch, indexing='ij')
    points = torch.stack([xs.flatten(), ys.flatten()], dim=1)
    rw, rh = mapping.resized
    ox, oy = (mapping.target-rw)//2, (mapping.target-rh)//2
    # Require the full descriptor footprint to lie in image content, not padding.
    half = mapping.patch/2
    legal = (points[:,0]-half>=ox)&(points[:,0]+half<=ox+rw)&(points[:,1]-half>=oy)&(points[:,1]+half<=oy+rh)
    # Exclude every patch touching a detector foreground rectangle, plus 4px
    # in the processed image. No identity labels are used.
    for box in mapping.boxes(boxes.to(device)):
        touches=(points[:,0]+half>=box[0]-4)&(points[:,0]-half<=box[2]+4)&(points[:,1]+half>=box[1]-4)&(points[:,1]-half<=box[3]+4)
        legal &= ~touches
    desc=feature_map[0].permute(1,2,0).reshape(-1,channels).float()
    legal &= torch.isfinite(desc).all(1)&(desc.norm(dim=1)>0)
    desc=torch.nn.functional.normalize(desc[legal],dim=1)
    points=points[legal]
    original=(points-points.new_tensor([ox,oy]))/points.new_tensor([rw/mapping.width,rh/mapping.height])
    return desc, original

def mutual_ratio_matches(desc1, desc2, points1, points2, ratio=.8):
    if min(len(desc1),len(desc2))<2:
        return np.empty((0,2)),np.empty((0,2))
    cosine=(desc1@desc2.T).clamp(-1,1)
    # Euclidean distance between unit descriptors; require the ratio in both
    # directions as well as a mutual nearest neighbour.
    distance=(2-2*cosine).clamp_min(0).sqrt()
    a, ai=distance.topk(2,dim=1,largest=False)
    b, bi=distance.topk(2,dim=0,largest=False)
    rows=torch.arange(len(desc1),device=desc1.device);cols=ai[:,0]
    legal=(bi[0,cols]==rows)&(a[:,0]<ratio*a[:,1])&(b[0,cols]<ratio*b[1,cols])
    return points1[legal].detach().cpu().numpy(),points2[cols[legal]].detach().cpu().numpy()

def validate_candidate(matrix, kind, points1, points2, shapes):
    result=dict(valid=False,validation_matches=len(points1),median_residual=None,inlier_ratio=0.,coverage=0.)
    if matrix is None or matrix.shape!=(3,3) or not np.isfinite(matrix).all() or len(points1)<20:
        return result
    norm=np.linalg.norm(matrix)
    if norm<=1e-12:return result
    matrix=matrix/norm
    if np.linalg.matrix_rank(matrix)!=(2 if kind=='F' else 3):return result
    try:
        residual=(symmetric_epipolar_distance if kind=='F' else symmetric_transfer_distance)(matrix,points1,points2)
    except (ValueError,np.linalg.LinAlgError):
        return result
    if not np.isfinite(residual).all():return result
    inside=residual<=12.;ratio=float(inside.mean());median=float(np.median(residual))
    a,b=points1[inside].astype(np.float32),points2[inside].astype(np.float32)
    coverage=min(cv2.contourArea(cv2.convexHull(a))/np.prod(shapes[0]),cv2.contourArea(cv2.convexHull(b))/np.prod(shapes[1])) if len(a)>=3 else 0.
    result.update(valid=bool(len(a)>=20 and ratio>=.25 and median<=12. and coverage>=.01),median_residual=median,inlier_ratio=ratio,coverage=float(coverage))
    return result
