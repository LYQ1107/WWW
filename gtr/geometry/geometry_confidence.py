"""Background-only ORB/SIFT RANSAC reference with observed validity diagnostics."""
import cv2, numpy as np
from .geometry_alignment import symmetric_epipolar_distance

def background_mask(image,boxes):
    mask=np.full(image.shape[:2],255,dtype=np.uint8)
    for box in boxes:
        x1,y1,x2,y2=np.round(box).astype(int);cv2.rectangle(mask,(max(0,x1-4),max(0,y1-4)),(min(mask.shape[1]-1,x2+4),min(mask.shape[0]-1,y2+4)),0,-1)
    return mask

def estimate_background(image1,image2,boxes1=(),boxes2=(),method='orb',mask_targets=True):
    detector=cv2.ORB_create(nfeatures=3000) if method=='orb' else cv2.SIFT_create(nfeatures=3000)
    mask1=background_mask(image1,boxes1) if mask_targets else None;mask2=background_mask(image2,boxes2) if mask_targets else None
    k1,d1=detector.detectAndCompute(image1,mask1);k2,d2=detector.detectAndCompute(image2,mask2)
    result=dict(valid=False,keypoints1=len(k1),keypoints2=len(k2),matches=0,inliers=0,inlier_ratio=0.,median_residual=None,coverage=0.,F=None,H=None,points1=np.empty((0,2)),points2=np.empty((0,2)))
    if d1 is None or d2 is None or len(d1)<8 or len(d2)<8:return result
    norm=cv2.NORM_HAMMING if method=='orb' else cv2.NORM_L2
    matches=cv2.BFMatcher(norm).knnMatch(d1,d2,k=2)
    good=[m for pair in matches if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance]
    result['matches']=len(good)
    if len(good)<8:return result
    p1=np.float32([k1[m.queryIdx].pt for m in good]);p2=np.float32([k2[m.trainIdx].pt for m in good])
    F,mask=cv2.findFundamentalMat(p1,p2,cv2.FM_RANSAC,3.,.99)
    H,_=cv2.findHomography(p1,p2,cv2.RANSAC,3.)
    result.update(H=H,points1=p1,points2=p2)
    if F is None or F.shape!=(3,3) or mask is None or not np.isfinite(F).all():return result
    inside=mask.ravel().astype(bool);p1i=p1[inside];p2i=p2[inside]
    ratio=float(inside.mean());res=symmetric_epipolar_distance(F,p1i,p2i)
    coverage=min(cv2.contourArea(cv2.convexHull(p1i))/(image1.shape[0]*image1.shape[1]),cv2.contourArea(cv2.convexHull(p2i))/(image2.shape[0]*image2.shape[1])) if len(p1i)>=3 else 0.
    median=float(np.median(res)) if len(res) and np.isfinite(res).all() else None
    valid=int(inside.sum())>=20 and ratio>=.25 and median is not None and median<=3 and coverage>=.01
    result.update(F=F/np.linalg.norm(F),inliers=int(inside.sum()),inlier_ratio=ratio,median_residual=median,coverage=float(coverage),valid=bool(valid))
    return result
