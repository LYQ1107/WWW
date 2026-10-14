"""Same joint-call VGGT pose to original-image F, with independent validity."""
import numpy as np
from .geometry_alignment import fundamental_from_cameras, symmetric_epipolar_distance

def validate_camera_fundamental(extrinsics, intrinsics, mappings, background):
    result=dict(F=None,valid=False,inlier_ratio=0.,median_residual=None,
                validation_matches=len(background['points1']),coverage=background['coverage'])
    try:
        f= fundamental_from_cameras(extrinsics[0],extrinsics[1],intrinsics[0],intrinsics[1])
        transforms=[]
        for m in mappings:
            w,h=m.resized
            transforms.append(np.array([[w/m.width,0,(m.target-w)//2],[0,h/m.height,(m.target-h)//2],[0,0,1.]]))
        f=transforms[1].T@f@transforms[0]
        norm=np.linalg.norm(f)
        if not np.isfinite(f).all() or norm<=1e-12:return result
        f=f/norm
        residual=symmetric_epipolar_distance(f,background['points1'],background['points2'])
        finite=len(residual)>=20 and np.isfinite(residual).all()
        median=float(np.median(residual)) if finite else None
        ratio=float((residual<=12.).mean()) if finite else 0.
        valid=finite and background['coverage']>=.01 and ratio>=.25 and median<=12.
        result.update(F=f,valid=bool(valid),inlier_ratio=ratio,median_residual=median)
    except (ValueError,np.linalg.LinAlgError,FloatingPointError):
        pass
    return result
