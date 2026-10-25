"""Six-dimensional world motion, independent of image boxes and identity.

CV/white-acceleration and Kalman equations follow Stone Soup's ConstantVelocity,
KalmanPredictor and KalmanUpdater (MIT; provenance in V4_SOURCE_ADAPTATION.md).
This is a small NumPy implementation, not an imported tracking/association stack.
State order is [x,y,z,vx,vy,vz]; units are relative-world units per frame_id tick
unless a verified timestamp_ticks_per_second is explicitly supplied.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def positive_covariance(value, size):
    c = np.asarray(value, dtype=float).reshape(size,size)
    if not np.isfinite(c).all() or not np.allclose(c,c.T,rtol=1e-7,atol=1e-10):
        raise ValueError("INVALID_SPATIAL_COVARIANCE")
    eigen = np.linalg.eigvalsh(c)
    if eigen.min() < -1e-9 * max(float(np.max(np.abs(eigen))),1e-12):
        raise ValueError("NON_POSITIVE_SPATIAL_COVARIANCE")
    # Only numerical regularisation, proportional to the data's own scale.
    floor = max(float(np.trace(c))/size*1e-9, np.finfo(float).tiny)
    return (c+c.T)/2 + np.eye(size)*max(0.,floor-float(eigen.min()))


def measurement_covariance(observation):
    """Support scatter, not scatter/N, inflated for unreliable measurements.

    Existing quality is derived from depth separation/point confidence. When
    additional aligner/visibility confidence is absent, it is not fabricated.
    Its absence is explicit and the existing eligibility remains mandatory.
    """
    if (not observation.world_state_update_eligible or not observation.geometry_valid
            or not np.isfinite(observation.world_center).all()):
        raise ValueError("SPATIAL_MEASUREMENT_ABSTAIN")
    quality = float(observation.geometry_reliability)
    alignment = observation.source.get('alignment_confidence', 1.)
    visibility = observation.source.get('geometry_visible_fraction', 1.)
    reliability = float(np.clip(quality*alignment*visibility, 0.,1.))
    if not np.isfinite(reliability) or reliability <= 0:
        raise ValueError("UNTRUSTED_GEOMETRY_ALIGNMENT")
    scatter=positive_covariance(observation.world_covariance,3)
    residual=observation.source.get('alignment_residual_p95')
    if residual is not None:
        residual=float(residual)
        if not np.isfinite(residual) or residual<0:raise ValueError("INVALID_ALIGNMENT_RESIDUAL")
        scatter=scatter+np.eye(3)*residual**2
    # Keep point-quality eligibility separate from uncertainty inflation:
    # inlier ratio is not itself a calibrated probability of a correct person.
    return scatter/max(reliability,.01), quality


def conservative_measurement(observations):
    """One measurement per Ω call; correlated views do not double precision."""
    usable=[]
    for o in observations:
        try: cov, reliability=measurement_covariance(o)
        except (ValueError,np.linalg.LinAlgError): continue
        usable.append((o,cov,reliability))
    if not usable: return None
    weights=np.asarray([o.score*r for o,_,r in usable],float)
    if weights.sum() <= 0:return None
    weights/=weights.sum()
    center=sum(w*o.world_center for w,(o,_,_) in zip(weights,usable))
    cov=sum(w*(c+np.outer(o.world_center-center,o.world_center-center))
            for w,(o,c,_) in zip(weights,usable))
    reliability=float(sum(w*r for w,(_,_,r) in zip(weights,usable)))
    # Disagreement also reduces evidence strength; uncertainty is not merely
    # inflated to make any pair look compatible.
    disagreement=sum(w*float((o.world_center-center)@np.linalg.solve(c,o.world_center-center))
                     for w,(o,c,_) in zip(weights,usable))
    reliability *= float(np.exp(-min(disagreement,100.)/6.))
    return np.asarray(center),positive_covariance(cov,3),reliability


@dataclass
class PersistentWorldMotionState:
    state: np.ndarray
    covariance: np.ndarray
    acceleration_density: np.ndarray
    last_prediction_timestamp: float
    last_trusted_geometry_timestamp: float
    motion_valid: bool=True
    geometry_confidence: float=1.
    geometry_frame_id: str="PERSISTENT_WORLD"
    uncertainty_status: str="TRUSTED_MEASUREMENT"
    timestamp_ticks_per_second: float | None=None
    alignment_confidence: float=1.

    @classmethod
    def initialize(cls,position,covariance,timestamp,*,confidence=1.,acceleration_ratio=.01,
                   ticks_per_second=None,frame_id="PERSISTENT_WORLD"):
        pos=np.asarray(position,float).reshape(3)
        if not np.isfinite(pos).all():raise ValueError("NONFINITE_POSITION")
        c=positive_covariance(covariance,3)
        if acceleration_ratio < 0 or (ticks_per_second is not None and ticks_per_second<=0):
            raise ValueError("INVALID_TIME_OR_PROCESS_NOISE")
        p=np.zeros((6,6));p[:3,:3]=c;p[3:,3:]=c
        return cls(np.r_[pos,np.zeros(3)],p,c*acceleration_ratio,float(timestamp),float(timestamp),
                   geometry_confidence=float(confidence),geometry_frame_id=str(frame_id),
                   timestamp_ticks_per_second=ticks_per_second)

    @property
    def position(self):return self.state[:3]

    @property
    def velocity(self):return self.state[3:]

    @property
    def time_unit(self):return "seconds" if self.timestamp_ticks_per_second else "frame_id_tick"

    def predict(self,timestamp):
        raw=float(timestamp)-self.last_prediction_timestamp
        if not np.isfinite(raw) or raw<0:raise ValueError("NONCAUSAL_MOTION_TIMESTAMP")
        if raw==0:return
        dt=raw/(self.timestamp_ticks_per_second or 1.)
        f=np.eye(6);f[:3,3:]=np.eye(3)*dt
        a=self.acceleration_density
        try:
            with np.errstate(over='raise',invalid='raise'):
                q=np.block([[a*dt**3/3,a*dt**2/2],[a*dt**2/2,a*dt]])
                predicted=f@self.state
                covariance=positive_covariance(f@self.covariance@f.T+q,6)
            if not np.isfinite(predicted).all():raise ValueError('NONFINITE_PREDICTION')
        except (ValueError,np.linalg.LinAlgError,FloatingPointError,OverflowError):
            self.motion_valid=False;self.geometry_confidence=0.;self.uncertainty_status="INVALID_PREDICTION_ABSTAIN"
            self.last_prediction_timestamp=float(timestamp)
            return
        self.state=predicted;self.covariance=covariance
        self.last_prediction_timestamp=float(timestamp)
        self.uncertainty_status="PREDICTED_NO_CURRENT_MEASUREMENT"

    def innovation(self,position,covariance):
        z=np.asarray(position,float).reshape(3)
        if not np.isfinite(z).all():raise ValueError("NONFINITE_POSITION")
        r=z-self.position
        s=positive_covariance(self.covariance[:3,:3]+positive_covariance(covariance,3),3)
        d2=max(0.,float(r@np.linalg.solve(s,r)))
        return r,s,d2

    def update(self,position,covariance,timestamp,*,confidence=1.,frame_id="PERSISTENT_WORLD"):
        if str(frame_id)!=self.geometry_frame_id:
            self.uncertainty_status="WORLD_ALIGNMENT_FAILED";return False
        self.predict(timestamp)
        try:
            measurement_cov=positive_covariance(covariance,3)
            innovation,s,_=self.innovation(position,measurement_cov)
            if not np.isfinite(confidence) or confidence<=0:raise ValueError("INVALID_CONFIDENCE")
        except (ValueError,np.linalg.LinAlgError):
            self.uncertainty_status="INVALID_MEASUREMENT_ABSTAIN";return False
        k=np.linalg.solve(s,self.covariance[:3,:]).T
        self.state=self.state+k@innovation
        kh=np.zeros((6,6));kh[:,:3]=k;i=np.eye(6)-kh
        # Joseph form preserves PSD under finite precision.
        self.covariance=positive_covariance(i@self.covariance@i.T+k@measurement_cov@k.T,6)
        self.last_trusted_geometry_timestamp=float(timestamp)
        self.geometry_confidence=float(np.clip(confidence,0,1));self.motion_valid=True
        self.uncertainty_status="TRUSTED_MEASUREMENT";return True

    def invalidate(self,status="SPATIAL_UNCERTAIN"):
        self.motion_valid=False;self.geometry_confidence=0.;self.uncertainty_status=status

    def transform(self,scale,rotation,translation,new_frame_id=None):
        scale=float(scale);r=np.asarray(rotation,float).reshape(3,3);t=np.asarray(translation,float).reshape(3)
        if (not np.isfinite(scale) or scale<=0 or not np.isfinite(r).all() or not np.isfinite(t).all()
                or not np.allclose(r@r.T,np.eye(3),atol=1e-6) or np.linalg.det(r)<0):
            raise ValueError("INVALID_SIM3")
        a=scale*r;j=np.zeros((6,6));j[:3,:3]=a;j[3:,3:]=a
        self.state=j@self.state;self.state[:3]+=t
        self.covariance=positive_covariance(j@self.covariance@j.T,6)
        self.acceleration_density=a@self.acceleration_density@a.T
        if new_frame_id is not None:self.geometry_frame_id=str(new_frame_id)


def temporal_geometry_evidence(q,observation,config):
    state=getattr(q,'spatial_state',None)
    if state is None or not state.motion_valid:return 0.,False
    try:
        r,reliability=measurement_covariance(observation)
        _,_,d2=state.innovation(observation.world_center,r)
    except (ValueError,np.linalg.LinAlgError):return 0.,False
    elapsed=max(0.,float(observation.timestamp)-state.last_trusted_geometry_timestamp)
    age_weight=np.exp(-elapsed/config.geometry_stale_ticks)
    # Continuous bounded log-likelihood surrogate. Neither high cosine nor
    # d_M alone bypasses the joint visual evidence and competing-ID solver.
    weight=reliability*state.geometry_confidence*age_weight
    weight*=float(observation.source.get('alignment_confidence',1.))
    value=config.temporal_geometry_weight*weight*(2*np.exp(-d2/6.)-1.)
    return float(value), bool(d2>config.geometry_gate**2 and weight>=.25)
