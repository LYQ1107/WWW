"""Vectorized causal history/current and current/current evidence; no identity labels."""
from __future__ import annotations
import numpy as np

GEOMETRY_DIM=16
BIRTH_GEOMETRY_DIM=16


def model_inputs(history,observations,config,timestamp,device="cpu"):
    # V4 G4 is independent of sync G1, temporal G2 and state-update G3.
    # Zeroing G4 retains the trained 16-D geometry tensor shapes and weights.
    import torch
    m,n,d=len(history),len(observations),config.feature_dim
    q=np.zeros((m,d),np.float32);obs=np.zeros((n,d),np.float32)
    qc=np.zeros((m,4),np.float32);dc=np.zeros((n,4),np.float32)
    geo=np.zeros((m,n,GEOMETRY_DIM),np.float32)
    birth_geo=np.zeros((n,n,BIRTH_GEOMETRY_DIM),np.float32)
    cameras=np.array([o.camera_id for o in observations],int)
    visible=np.array([o.visible for o in observations],bool)
    # Store one orientation per physical cross-camera edge. No GT pruning.
    birth_mask=(cameras[:,None]<cameras[None,:])&visible[:,None]&visible[None,:]
    if int(birth_mask.sum())>config.max_birth_pair_edges:
        raise ValueError("BIRTH_PAIR_EDGE_LIMIT: no labelled samples silently removed")
    for i,x in enumerate(history):
        if config.use_appearance and x.identity_embedding is not None:
            memory=getattr(x,'identity_state',None)
            q[i]=(memory.context_embedding(cameras[visible],timestamp,d,config.min_memory_score)
                  if config.enable_multiview_coordination and memory is not None else x.identity_embedding)
        qc[i]=[min(x.age,1000)/1000,min(x.missed,100)/100,x.identity_confidence,float(x.status=="CONFIRMED")]
    for j,o in enumerate(observations):
        if config.use_appearance and o.feature_valid:obs[j]=o.roi_feature
        dc[j]=[o.score,float(o.visible),np.sin(float(o.camera_id)),np.cos(float(o.camera_id))]
    if config.head_geometry and n:
        centers=np.nan_to_num(np.array([o.world_center for o in observations]),nan=0.,posinf=0.,neginf=0.)
        cov=np.nan_to_num(np.array([np.diag(o.world_covariance) for o in observations]),nan=1e6,posinf=1e6,neginf=1e6)
        rel=np.array([o.geometry_reliability for o in observations])
        valid=np.array([o.geometry_valid for o in observations],float)
        fusion=np.array([o.cross_view_fusion_eligible for o in observations],float)
        update=np.array([o.world_state_update_eligible for o in observations],float)
        available=np.array([o.point_support_available for o in observations],float)
        birth_geo[:,:,:3]=np.clip(np.abs(centers[:,None]-centers[None,:]),0,100)
        birth_geo[:,:,3:6]=np.log1p(np.clip(cov[:,None]+cov[None,:],0,1e6))
        for offset,values in ((6,rel),(8,valid),(10,fusion),(12,update),(14,available)):
            birth_geo[:,:,offset]=values[:,None];birth_geo[:,:,offset+1]=values[None,:]
        birth_geo*=((valid[:,None]>0)&(valid[None,:]>0))[:,:,None]
        if m:
            qcenters=np.nan_to_num(np.array([x.center for x in history]),nan=0.,posinf=0.,neginf=0.)
            qcov=np.nan_to_num(np.array([np.diag(x.covariance) for x in history]),nan=1e6,posinf=1e6,neginf=1e6)
            qvalid=np.array([x.geometry_valid for x in history],float)
            geo[:,:,:3]=np.clip(centers[None]-qcenters[:,None],-100,100)
            geo[:,:,3:6]=np.log1p(np.clip(cov[None]+qcov[:,None],0,1e6))
            geo[:,:,6]=rel[None];geo[:,:,7]=qvalid[:,None];geo[:,:,8]=valid[None]
            geo[:,:,9]=np.array([min(timestamp-x.last_timestamp,100)/100 for x in history])[:,None]
            geo[:,:,10]=[[float(x.visibility_by_camera.get(o.camera_id,False)) for o in observations] for x in history]
            geo[:,:,11]=fusion[None];geo[:,:,12]=update[None];geo[:,:,13]=available[None]
            geo[:,:,14]=np.log1p(np.clip(cov.sum(1)[None]+qcov.sum(1)[:,None],0,1e6))
            geo[:,:,15]=qc[:,1,None]
            geo*=((qvalid[:,None]>0)&(valid[None,:]>0))[:,:,None]
    arrays=(q,obs,geo,dc,qc,birth_geo,birth_mask)
    if any(not np.isfinite(x).all() for x in arrays):raise ValueError("NONFINITE_MODEL_INPUT")
    return tuple(torch.as_tensor(x,device=device) for x in arrays)
