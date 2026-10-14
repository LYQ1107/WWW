"""Current/past background motion and native GMT identity ambiguity; no labels."""
import cv2,numpy as np,torch
from .geometry_confidence import background_mask

class CausalBackgroundMotion:
    def __init__(self):
        self.previous={}

    def observe(self,records,boxes):
        values=[];details=[]
        for record,foreground in zip(records,boxes):
            camera=int(record['view_id'])
            image=cv2.imread(record['file_name'],cv2.IMREAD_GRAYSCALE)
            if image is None:raise OSError('cannot read current motion image')
            h,w=image.shape;small=cv2.resize(image,(320,180))
            scaled=foreground.detach().cpu().numpy()*np.array([320/w,180/h]*2)
            mask=background_mask(small,scaled)
            previous=self.previous.get(camera);self.previous[camera]=(small,mask)
            row=dict(camera=camera,valid=False,inliers=0,normalized_motion=0.,median_residual_processed_px=None)
            if previous is not None:
                old,old_mask=previous
                points=cv2.goodFeaturesToTrack(old,maxCorners=128,qualityLevel=.01,minDistance=5,mask=old_mask)
                if points is not None and len(points)>=12:
                    forward,status,_=cv2.calcOpticalFlowPyrLK(old,small,points,None)
                    if forward is not None:
                        backward,reverse_status,_=cv2.calcOpticalFlowPyrLK(small,old,forward,None)
                        if backward is not None:
                            a=points[:,0];b=forward[:,0]
                            finite=np.isfinite(b).all(1)&np.isfinite(backward[:,0]).all(1)
                            hit=finite&(status[:,0]>0)&(reverse_status[:,0]>0)&(np.linalg.norm(backward[:,0]-a,axis=1)<=1.)
                            pixel=np.rint(np.where(np.isfinite(b),b,0)).astype(int)
                            hit&=(pixel[:,0]>=0)&(pixel[:,0]<320)&(pixel[:,1]>=0)&(pixel[:,1]<180)
                            hit&=mask[pixel[:,1].clip(0,179),pixel[:,0].clip(0,319)]>0
                            a,b=a[hit],b[hit]
                            if len(a)>=12:
                                cv2.setRNGSeed(20261014)
                                H,inliers=cv2.findHomography(a,b,cv2.RANSAC,2.)
                                if H is not None and inliers is not None and np.isfinite(H).all():
                                    inside=inliers[:,0]>0
                                    if inside.sum()>=12 and inside.mean()>=.5:
                                        projected=cv2.perspectiveTransform(a[None],H)[0]
                                        residual=np.linalg.norm(projected[inside]-b[inside],axis=1)
                                        coverage=cv2.contourArea(cv2.convexHull(a[inside]))/(320*180)
                                        if np.isfinite(residual).all() and np.median(residual)<=2. and coverage>=.01:
                                            motion=float(np.median(np.linalg.norm(b[inside]-a[inside],axis=1))/np.hypot(320,180))
                                            row.update(valid=True,inliers=int(inside.sum()),normalized_motion=motion,median_residual_processed_px=float(np.median(residual)))
                                            values.append(motion)
            details.append(row)
        return max(values,default=0.),details

def identity_update_signal(logits,instances,k,activate):
    refs=[x for i,x in enumerate(instances) if i!=k];query=instances[k]
    if not len(query) or not refs or any(not x.has('geo_camera') for x in refs):return 1.,False
    ids=torch.cat([x.track_ids for x in refs]);cams=torch.cat([x.geo_camera for x in refs])
    unique=ids.unique()
    if not len(unique):return 1.,False
    incidence=(ids[:,None]==unique[None,:]).float()
    probabilities=torch.cat(activate(logits.split([len(x) for x in refs],dim=1)),dim=1)
    score=probabilities@incidence
    opposite=(query.geo_camera[:,None]!=cams[None,:]).float()@incidence>0
    same=(query.geo_camera[:,None]==cams[None,:]).float()@incidence>0
    top=score.topk(min(2,len(unique)),dim=1)
    best=top.indices[:,0];rows=torch.arange(len(query),device=logits.device)
    handoff=bool((opposite[rows,best]&~same[rows,best]).any())
    if len(unique)<2:return 1.,handoff
    # Only rows with an opposite-camera candidate among the competing tracks.
    legal=opposite.gather(1,top.indices).any(1)
    margins=(top.values[:,0]-top.values[:,1])/top.values[:,0].abs().clamp_min(1e-8)
    legal&=torch.isfinite(margins)
    return (float(margins[legal].min()) if legal.any() else 1.),handoff
