"""Auditable Ω support evidence. Missing world/camera transforms never become I."""
from dataclasses import dataclass, field
from copy import copy
import numpy as np
from ..geometry.camera_model import CameraModel
from ..data.worldobject_observations import replay_observations


@dataclass
class WorldGeometryContext:
    timestamp: int
    cameras: dict = field(default_factory=dict)
    alignment_success: bool = False
    alignment_confidence: float = 0.
    residual_p95: float | None = None
    reprojection_status: str = "WORLD_FROM_OMEGA_NOT_RECORDED"
    pair_evidence_cache: dict = field(default_factory=dict,repr=False)

    @classmethod
    def from_record(cls, timestamp, record):
        a=record.get("alignment",{})
        success=bool(a.get("success",False))
        ratio=1. if a.get("reason")=="initial_reference" else a.get("inlier_ratio")
        confidence=float(np.clip(ratio,0,1)) if ratio is not None and np.isfinite(ratio) and success else 0.
        residual=a.get("residual_p95")
        if residual is not None and (not np.isfinite(residual) or residual<0):residual=None
        context=cls(int(timestamp),alignment_success=success,alignment_confidence=confidence,residual_p95=residual)
        transform=a.get("world_from_omega")
        if transform is None and success and a.get("reason")=="initial_reference":transform=np.eye(4)
        if transform is None:return context
        t=np.asarray(transform,float)
        if (not success or t.shape!=(4,4) or not np.isfinite(t).all()
                or not np.allclose(t[3],[0,0,0,1]) or np.linalg.det(t[:3,:3])<=0):
            context.reprojection_status="INVALID_WORLD_FROM_OMEGA_ABSTAIN";return context
        linear=t[:3,:3];scale=np.cbrt(np.linalg.det(linear))
        if not np.allclose((linear/scale)@(linear/scale).T,np.eye(3),atol=1e-5):
            context.reprojection_status="NON_SIM3_CAMERA_TRANSFORM_ABSTAIN";return context
        try:
            for camera,value in record.get("camera_models",{}).items():
                args=dict(value);args["world_from_camera"]=t@np.asarray(value["world_from_camera"],float)
                context.cameras[int(camera)]=CameraModel(**args)
        except (ValueError,np.linalg.LinAlgError):
            context.cameras={};context.reprojection_status="INVALID_CAMERA_ABSTAIN";return context
        context.reprojection_status="VERIFIED_CURRENT_WORLD_CAMERA" if context.cameras else "CAMERA_NOT_RECORDED"
        return context


def v5_observations(frame, use_geometry=True):
    context=WorldGeometryContext.from_record(frame["timestamp"],frame.get("record",{}))
    result=[]
    for original in replay_observations(frame):
        o=copy(original)
        o.source={**original.source,"alignment_confidence":context.alignment_confidence,
                  "alignment_evidence_source":"CURRENT_CACHE_RECORD",
                  "reprojection_status":context.reprojection_status}
        if context.residual_p95 is not None:o.source["alignment_residual_p95"]=context.residual_p95
        if not use_geometry:
            o.geometry_valid=False;o.geometry_reliability=0.;o.world_center=np.full(3,np.nan)
            o.world_covariance=np.eye(3)*1e6;o.spatial_support=None
            o.point_support_available=False;o.cross_view_fusion_eligible=False;o.world_state_update_eligible=False
            o.geometry_status="REGISTERED_NO_OMEGA_B4"
        result.append(o)
    if not use_geometry:context=WorldGeometryContext(int(frame["timestamp"]),reprojection_status="REGISTERED_NO_OMEGA_B4")
    return result,context


def projection_coverage(points, observation, context):
    """Coverage against a real current detection; never a visibility claim."""
    camera=context.cameras.get(observation.camera_id)
    p=np.asarray(points,float).reshape(-1,3)
    if camera is None or not len(p):return None
    if observation.timestamp!=context.timestamp:raise ValueError("STALE_CAMERA_EVIDENCE")
    uv,valid=camera.project(p,clip=False)
    x,y,w,h=observation.bbox_xywh
    inside=valid&(uv[:,0]>=x)&(uv[:,0]<=x+w)&(uv[:,1]>=y)&(uv[:,1]<=y+h)
    return {"coverage":float(inside.mean()),"positive_depth_fraction":float(valid.mean()),
            "point_count":len(p),"current_detection_supported":True}


def _support_points(item):
    if hasattr(item,"spatial_support"):
        s=item.spatial_support
        return np.asarray(s.foreground_points,float).reshape(-1,3) if s is not None else np.empty((0,3))
    p=np.asarray(item.support_points,float).reshape(-1,3)
    old=getattr(item,"support_center",None)
    return p+(item.center-old) if old is not None and item.geometry_valid else p


def _spatial_pair_evidence(a,b,context,config):
    """Partial surfaces are allowed; centres alone never certify a shared person."""
    out={"available":False,"reliable":False,"mahalanobis":None,"support_gap":None,
         "reprojection_available":False,"coverage_a_to_b":None,"coverage_b_to_a":None,
         "reason":"GEOMETRY_UNRELIABLE"}
    if not config.use_geometry or not (a.cross_view_fusion_eligible and b.cross_view_fusion_eligible):return out
    if (context.timestamp!=a.timestamp or a.timestamp!=b.timestamp):raise ValueError("NONCURRENT_PAIR_GEOMETRY")
    pa,pb=_support_points(a),_support_points(b)
    if len(pa)<8 or len(pb)<8 or context.alignment_confidence<config.alignment_min_inlier_ratio:return out
    covariance=a.world_covariance+b.world_covariance
    floor=max(float(np.trace(covariance))/3*1e-4,1e-10)
    if context.residual_p95 is not None:floor+=context.residual_p95**2
    covariance=covariance+np.eye(3)*floor
    delta=a.world_center-b.world_center
    try:d2=max(0.,float(delta@np.linalg.solve(covariance,delta)))
    except np.linalg.LinAlgError:return out
    lo_a,hi_a=np.percentile(pa,[10,90],axis=0);lo_b,hi_b=np.percentile(pb,[10,90],axis=0)
    gap=np.maximum(0.,np.maximum(lo_a,lo_b)-np.minimum(hi_a,hi_b))
    gap=float(np.sqrt(np.sum(gap**2/np.maximum(np.diag(covariance),floor))))
    p_ab=projection_coverage(pa,b,context);p_ba=projection_coverage(pb,a,context)
    if p_ab is not None and p_ba is not None:
        out.update(reprojection_available=True,coverage_a_to_b=p_ab["coverage"],coverage_b_to_a=p_ba["coverage"])
    quality=min(a.geometry_reliability,b.geometry_reliability)
    reliable=quality>=config.reliable_geometry_min_quality
    out.update(available=True,reliable=bool(reliable),mahalanobis=float(np.sqrt(d2)),support_gap=gap,
               quality=float(quality),alignment_confidence=context.alignment_confidence,reason="CURRENT_WORLD_SUPPORT_DISTRIBUTIONS")
    return out


def spatial_pair_evidence(a,b,context,config):
    key=(a.key,b.key,config.use_geometry)
    if key not in context.pair_evidence_cache:
        context.pair_evidence_cache[key]=_spatial_pair_evidence(a,b,context,config)
    return context.pair_evidence_cache[key]


def historical_geometry(history,observations,context,config):
    """32 bounded fields with explicit availability, quality and gauge masks."""
    m,n=len(history),len(observations);g=np.zeros((m,n,32),np.float32)
    if not config.use_geometry or not m or not n:return g
    if not config.rich_geometry:
        from .world_object_model_inputs import model_inputs
        from ..tracking.world_object_v4_variants import state_configs
        # No second implementation of the registered old16-D representation.
        old=model_inputs(history,observations,state_configs(1024)["C1"],context.timestamp,"cpu")[2].numpy()
        g[:,:,:16]=old;return g
    hc=np.asarray([q.center if q.geometry_valid else np.zeros(3) for q in history])
    oc=np.asarray([o.world_center if o.geometry_valid else np.zeros(3) for o in observations])
    hp=np.asarray([q.covariance if q.geometry_valid else np.eye(3)*1e6 for q in history])
    op=np.asarray([o.world_covariance if o.geometry_valid else np.eye(3)*1e6 for o in observations])
    age=np.asarray([max(0,context.timestamp-(q.spatial_state.last_trusted_geometry_timestamp
        if q.spatial_state is not None else q.last_timestamp)) for q in history])
    hquality=np.asarray([q.spatial_state.geometry_confidence if q.spatial_state is not None and q.spatial_state.motion_valid else 0. for q in history])
    oquality=np.asarray([o.geometry_reliability if o.geometry_valid else 0. for o in observations])
    hvalid=np.asarray([q.geometry_valid for q in history]);ovalid=np.asarray([o.geometry_valid for o in observations])
    valid=hvalid[:,None]&ovalid[None,:]&context.alignment_success
    covariance=hp[:,None]+op[None,:]
    diag=np.diagonal(covariance,axis1=-2,axis2=-1).copy()
    floor=np.maximum(diag.mean(-1)*1e-4,1e-10)
    if context.residual_p95 is not None:floor+=context.residual_p95**2
    covariance=covariance+np.eye(3)*floor[:,:,None,None]
    delta=oc[None]-hc[:,None]
    diag=np.maximum(np.diagonal(covariance,axis1=-2,axis2=-1),1e-10)
    weight=hquality[:,None]*oquality[None]*context.alignment_confidence*np.exp(-age[:,None]/config.geometry_stale_ticks)
    z=np.clip(delta/np.sqrt(diag),-8,8)
    g[:,:,:3]=z*weight[:,:,None]
    g[:,:,3:6]=np.clip(np.log1p(diag),0,20)/20
    g[:,:,6]=weight;g[:,:,7]=hquality[:,None];g[:,:,8]=oquality[None]
    g[:,:,9]=np.minimum(age[:,None]/100,1);g[:,:,10]=context.alignment_confidence
    g[:,:,11]=np.clip(np.linalg.norm(z,axis=-1)/8,0,1)*weight
    g[:,:,12]=np.asarray([o.cross_view_fusion_eligible for o in observations],float)[None]
    g[:,:,13]=np.asarray([o.world_state_update_eligible for o in observations],float)[None]
    g[:,:,14]=np.asarray([o.spatial_support.foreground_fraction if o.spatial_support else 0. for o in observations])[None]
    g[:,:,15]=np.asarray([min(len(_support_points(o)),64)/64 for o in observations])[None]
    g[:,:,16]=np.asarray([[float(q.visibility_by_camera.get(o.camera_id,False)) for o in observations] for q in history])
    g[:,:,17]=np.clip(np.log1p(np.trace(hp,axis1=-2,axis2=-1)),0,20)[:,None]/20
    g[:,:,18]=np.clip(np.log1p(np.trace(op,axis1=-2,axis2=-1)),0,20)[None]/20
    g[:,:,19]=np.asarray([o.point_support_available for o in observations],float)[None]
    g[:,:,20]=np.asarray([q.spatial_state is not None and q.spatial_state.motion_valid for q in history],float)[:,None]
    if context.residual_p95 is not None:g[:,:,21]=np.clip(context.residual_p95/np.sqrt(diag.mean(-1)),0,8)/8
    # Quantile support regions retain partial surface overlap without asserting
    # that the two cameras sampled the same anatomical points.
    hbounds=[np.percentile(_support_points(q),[10,90],axis=0) if len(_support_points(q)) else np.zeros((2,3)) for q in history]
    obounds=[np.percentile(_support_points(o),[10,90],axis=0) if len(_support_points(o)) else np.zeros((2,3)) for o in observations]
    ha,hb=np.asarray(hbounds),np.asarray(obounds)
    gap=np.maximum(0,np.maximum(ha[:,None,0],hb[None,:,0])-np.minimum(ha[:,None,1],hb[None,:,1]))
    g[:,:,22:25]=np.clip(gap/np.sqrt(diag),0,8)*weight[:,:,None]/8
    g[:,:,25]=np.clip(np.linalg.norm((ha[:,None,1]-ha[:,None,0])-(hb[None,:,1]-hb[None,:,0]),axis=-1)/np.sqrt(diag.sum(-1)),0,8)/8
    # A sparse compute proposal is based only on current visual evidence. All
    # membership edges still exist; uncomputed projection fields stay masked.
    if context.cameras:
        raw=np.asarray([q.identity_embedding if q.identity_embedding is not None else np.zeros(config.feature_dim) for q in history])
        current=np.asarray([o.roi_feature for o in observations]);app=raw@current.T
        for j,o in enumerate(observations):
            for i in np.argsort(-app[:,j],kind="stable")[:config.top_k]:
                if not valid[i,j]:continue
                projected=projection_coverage(_support_points(history[i]),o,context)
                if projected is not None:
                    g[i,j,26]=projected["coverage"];g[i,j,27]=projected["positive_depth_fraction"];g[i,j,28]=1.
    g[:,:,29]=valid;g[:,:,30]=float(context.cameras!={});g[:,:,31]=np.asarray([q.missed>0 for q in history],float)[:,None]
    g*=valid[:,:,None]
    if not np.isfinite(g).all():raise ValueError("NONFINITE_V5_GEOMETRY")
    return g
