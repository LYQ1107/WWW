"""One history/observation input builder for inference and supervised replay."""
from __future__ import annotations
import numpy as np


def model_inputs(history, observations, config, timestamp, device="cpu"):
    import torch
    m,n,d = len(history),len(observations),config.feature_dim
    q = np.zeros((m,d), np.float32)
    obs = np.zeros((n,d), np.float32)
    qc,dc = np.zeros((m,4),np.float32), np.zeros((n,4),np.float32)
    geo = np.zeros((m,n,12),np.float32)
    for i,x in enumerate(history):
        if config.use_appearance and x.identity_embedding is not None:
            q[i] = x.identity_embedding
        qc[i] = [min(x.age,1000)/1000, min(x.missed,100)/100, x.identity_confidence,
                 float(x.status == "CONFIRMED")]
    for j,o in enumerate(observations):
        if config.use_appearance and o.feature_valid:
            obs[j] = o.roi_feature
        # Camera ID denotes source only; it is never a target or identity key.
        dc[j] = [o.score, float(o.visible), np.sin(float(o.camera_id)), np.cos(float(o.camera_id))]
        for i,x in enumerate(history):
            if config.head_geometry and x.geometry_valid and o.geometry_valid:
                delta = o.world_center-x.center
                cov = np.diag(o.world_covariance+x.covariance)
                geo[i,j] = np.r_[np.clip(delta,-100,100), np.clip(cov,0,100),
                                 o.geometry_reliability, float(x.geometry_valid), 1.0,
                                 min(timestamp-x.last_timestamp,100)/100,
                                 float(x.visibility_by_camera.get(o.camera_id,False)), 1.0]
    return tuple(torch.as_tensor(x,device=device) for x in (q,obs,geo,dc,qc))
