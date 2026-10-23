"""Deterministic, spatially spread person-box interior query points."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class PersonPointReference:
    xy: np.ndarray
    weights: np.ndarray
    source_camera: int
    timestamp: int
    detection_id: int
    global_id: int | None = None


class PersonPointSampler:
    def __init__(self,points_per_person=16,interior_fraction=.15):
        if points_per_person not in [8,16,32]:raise ValueError('Registered point counts: 8/16/32')
        self.points_per_person=int(points_per_person)
        self.interior_fraction=float(interior_fraction)

    def sample(self,image,detection,timestamp,global_id=None):
        image=np.asarray(image)
        h,w=image.shape[:2]
        x,y,bw,bh=map(float,detection.bbox_xywh)
        f=self.interior_fraction
        lo=np.maximum([0,0],[x+f*bw,y+f*bh])
        hi=np.minimum([w-1,h-1],[x+(1-f)*bw,y+(1-f)*bh])
        empty=lambda:PersonPointReference(np.empty((0,2),np.float32),np.empty(0,np.float32),
            detection.view_id,int(timestamp),detection.detection_id,global_id)
        if np.any(hi<lo) or min(bw,bh)<=0:return empty()
        xx=np.linspace(lo[0],hi[0],min(12,max(1,int(hi[0]-lo[0])+1)))
        yy=np.linspace(lo[1],hi[1],min(12,max(1,int(hi[1]-lo[1])+1)))
        candidates=np.unique(np.array([(a,b) for b in yy for a in xx],np.float32),axis=0)
        pixel=np.rint(candidates).astype(int)
        if detection.mask is not None:
            mask=np.asarray(detection.mask,bool)
            if mask.shape!=(h,w):raise ValueError('Person mask must be in original RGB coordinates')
            keep=mask[pixel[:,1],pixel[:,0]];candidates,pixel=candidates[keep],pixel[keep]
        if not len(candidates):return empty()
        # Compute local gradients only at candidate locations, avoiding a
        # repeated full-image gradient allocation for each person.
        ix,iy=pixel[:,0],pixel[:,1]
        grey=lambda yy,xx:image[yy,xx].astype(np.float32).mean(-1)
        gradient=np.abs(grey(iy,np.minimum(ix+1,w-1))-grey(iy,np.maximum(ix-1,0)))
        gradient+=np.abs(grey(np.minimum(iy+1,h-1),ix)-grey(np.maximum(iy-1,0),ix))
        texture=np.clip(gradient/64.,0,1)
        weights=.25+.75*texture
        chosen=[];available=np.ones(len(candidates),bool)
        normalized=(candidates-lo)/np.maximum(hi-lo,1)
        nearest=np.full(len(candidates),1.,np.float32)
        for _ in range(min(self.points_per_person,len(candidates))):
            rank=(.1+.9*nearest)*weights
            rank[~available]=-np.inf
            best=int(np.argmax(rank));chosen.append(best);available[best]=False
            distance=np.linalg.norm(normalized-normalized[best],axis=1)
            nearest=np.minimum(nearest,distance)
        return PersonPointReference(candidates[chosen],weights[chosen].astype(np.float32),
            int(detection.view_id),int(timestamp),int(detection.detection_id),global_id)
