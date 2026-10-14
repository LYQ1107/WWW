"""Completed geometry propagated only through validated image-plane warps.

For real camera motion these homographies are a background-plane approximation,
not an exact general3D motion model. Current independent pixel validation is
mandatory and evidence stays a soft prior, never a hard identity restriction.
"""
import time,numpy as np
from .geometry_alignment import propagate_fundamental

class TemporalGeometryConsumer:
    def __init__(self,max_age=20):
        self.max_age=max_age;self.edges={};self.state=None
        self.stats=dict(received_valid=0,received_invalid=0,expired=0,missing_warp=0,unreliable_warp=0,propagated=0)

    def add_motion(self,frame,details):
        self.edges[frame]={row['camera']:row for row in details}
        self.edges={t:rows for t,rows in self.edges.items() if t>frame-self.max_age}

    def receive(self,message,current_frame,now=None):
        now=time.monotonic() if now is None else now
        result=message['result'];source=message['source_frame']
        if source>current_frame or message['completed_at']>now or not result.get('valid',False):
            self.state=None;self.stats['received_invalid']+=1;return False
        F=result.get('F')
        if F is None or np.shape(F)!=(3,3) or not np.isfinite(F).all():
            self.state=None;self.stats['received_invalid']+=1;return False
        self.state=message;self.stats['received_valid']+=1;return True

    def candidate(self,current_frame,now=None):
        if self.state is None:return None
        now=time.monotonic() if now is None else now
        source=self.state['source_frame'];result=self.state['result']
        if self.state['completed_at']>now or not 0<=current_frame-source<=self.max_age:
            self.stats['expired']+=1;return None
        cameras=tuple(result['camera_order']);transforms={camera:np.eye(3) for camera in cameras};error_sq=0.
        for frame in range(source+1,current_frame+1):
            rows=self.edges.get(frame,{})
            for camera in cameras:
                row=rows.get(camera)
                if row is None or not row.get('valid') or row.get('temporal_H_original') is None:
                    self.stats['missing_warp']+=1;return None
                H=np.asarray(row['temporal_H_original'],dtype=float);residual=row['median_residual_processed_px']
                if not np.isfinite(H).all() or np.linalg.matrix_rank(H)!=3 or np.linalg.cond(H)>1e8 or residual is None or not np.isfinite(residual):
                    self.stats['unreliable_warp']+=1;return None
                transforms[camera]=H@transforms[camera];error_sq+=residual**2
        residual=error_sq**.5
        if residual>3.:
            self.stats['unreliable_warp']+=1;return None
        try:F=propagate_fundamental(result['F'],transforms[cameras[0]],transforms[cameras[1]])
        except (ValueError,np.linalg.LinAlgError):return None
        self.stats['propagated']+=int(current_frame>source)
        return dict(F=F,camera_order=cameras,source_frame=source,completed_at=self.state['completed_at'],
            propagation_residual_processed_px=residual,plane_assumption=True,confidence_decay=float(np.exp(-residual/3-(current_frame-source)/self.max_age)),valid=False)
