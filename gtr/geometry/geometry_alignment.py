"""Camera-from-world conventions: x2.T F21 x1=0; no cross-call world comparison."""
import numpy as np

def skew(t):
    x,y,z=np.asarray(t).reshape(3);return np.array([[0,-z,y],[z,0,-x],[-y,x,0]],dtype=float)

def fundamental_from_cameras(extrinsic1,extrinsic2,K1,K2):
    R1,t1=extrinsic1[:,:3],extrinsic1[:,3];R2,t2=extrinsic2[:,:3],extrinsic2[:,3]
    R21=R2@R1.T;t21=t2-R21@t1
    F=np.linalg.inv(K2).T@skew(t21)@R21@np.linalg.inv(K1)
    norm=np.linalg.norm(F)
    if not np.isfinite(F).all() or norm<1e-12:raise ValueError('degenerate relative pose')
    return F/norm

def symmetric_epipolar_distance(F,points1,points2):
    a=np.c_[points1,np.ones(len(points1))];b=np.c_[points2,np.ones(len(points2))]
    l2=a@F.T;l1=b@F;r=(l2*b).sum(-1)**2
    den1=(l1[:,:2]**2).sum(-1);den2=(l2[:,:2]**2).sum(-1)
    with np.errstate(divide='ignore',invalid='ignore'):return np.sqrt(.5*r*(1/den1+1/den2))

def propagate_fundamental(F,H1_current_from_old,H2_current_from_old):
    Fnew=np.linalg.inv(H2_current_from_old).T@F@np.linalg.inv(H1_current_from_old)
    if not np.isfinite(Fnew).all() or np.linalg.norm(Fnew)<1e-12:raise ValueError('invalid propagation')
    return Fnew/np.linalg.norm(Fnew)

def coordinate_fundamental(F,A1_new_from_old,A2_new_from_old):
    return propagate_fundamental(F,A1_new_from_old,A2_new_from_old)

def soft_box_evidence(F,boxes1,boxes2,valid,confidence=1.,tolerance_pixels=12.):
    """Neutral on invalid. Box centers approximate evidence, never exact surface points."""
    shape=(len(boxes2),len(boxes1))
    if not valid or F is None or not np.isfinite(F).all():return np.zeros(shape)
    p1=(boxes1[:,:2]+boxes1[:,2:])/2;p2=(boxes2[:,:2]+boxes2[:,2:])/2
    if not p1.size or not p2.size:return np.zeros(shape)
    a=np.tile(p1,(len(p2),1));b=np.repeat(p2,len(p1),axis=0)
    distances=symmetric_epipolar_distance(F,a,b).reshape(shape)
    # Enlarged uncertainty accounts for differing visible box surfaces.
    radius1=np.linalg.norm(boxes1[:,2:]-boxes1[:,:2],axis=1)/2
    radius2=np.linalg.norm(boxes2[:,2:]-boxes2[:,:2],axis=1)/2
    scale=tolerance_pixels+radius2[:,None]+radius1[None,:]
    score=np.clip(confidence,0,1)*np.tanh((1-distances/scale)/2)
    return np.where(np.isfinite(score),score,0.)
