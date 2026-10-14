"""Offline target IoU labels; no model dependencies or inference inputs."""
import numpy as np

def identity(box,annotations):
    if not annotations:return -1
    gt=np.asarray([a['bbox'] for a in annotations],dtype=float);gt[:,2:]+=gt[:,:2]
    b=np.asarray(box);lo=np.maximum(b[:2],gt[:,:2]);hi=np.minimum(b[2:],gt[:,2:]);inter=np.clip(hi-lo,0,None).prod(-1)
    area=np.clip(b[2:]-b[:2],0,None).prod();ga=np.clip(gt[:,2:]-gt[:,:2],0,None).prod(-1)
    iou=inter/np.maximum(area+ga-inter,1e-12);j=int(iou.argmax())
    return int(annotations[j]['instance_id']) if iou[j]>=.5 else -1

