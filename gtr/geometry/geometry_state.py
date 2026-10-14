"""Completion-gated geometry state: expired, future or invalid result is neutral."""
from dataclasses import dataclass
import numpy as np

@dataclass
class GeometryState:
    source_frame:int
    completed_at:float
    F:object
    valid:bool
    confidence:float
    source_camera_ids:tuple=(1,2)
    propagation_residual:float=0.

    def usable(self,current_frame,now,max_age=20):
        return bool(self.valid and self.F is not None and np.isfinite(self.F).all() and self.completed_at<=now and 0<=current_frame-self.source_frame<=max_age and np.isfinite(self.propagation_residual) and self.propagation_residual<=3.)
