"""Fixed camera/time proposal-logit prior, with no new learned parameters."""
import time
from types import MethodType
import torch


def observe(self,instances,records,frame):
    if not self.active:return
    started=time.perf_counter()
    for inst,record in zip(instances,records):
        native=inst.reid_features;n=len(inst);height,width=inst.image_size
        inst.geo_camera=torch.full((n,),int(record['view_id']),dtype=torch.long,device=native.device)
        inst.geo_frame=torch.full((n,),frame,dtype=torch.long,device=native.device)
        raw=native.float()
        inst.geo_valid=torch.isfinite(raw).all(1)&(raw.norm(dim=1)>0)
        inst.geo_xyxy=inst.pred_boxes.tensor.detach().clone()*inst.pred_boxes.tensor.new_tensor([record['width']/width,record['height']/height]*2)
        # Uniform metadata supports existing diagnostic Instances operations;
        # there is no copied native/foundation representation in this field.
        inst.geo_vector=native.new_empty((n,0))
        self.native_prior_stats['observed_targets']+=n
        self.native_prior_stats['maximum_extra_vector_columns']=max(self.native_prior_stats['maximum_extra_vector_columns'],inst.geo_vector.shape[1])
    self.native_prior_stats['observed_groups']+=1
    self.observed_frame=frame;self.calls+=1;self.seconds+=time.perf_counter()-started


def adapt(self,outputs,instances,k):
    if not self.active or not self.score.enabled:return outputs
    query=instances[k];refs=[inst for index,inst in enumerate(instances) if index!=k]
    if not len(query) or not refs:return outputs
    if not query.has('geo_frame') or any(not inst.has('geo_frame') for inst in refs):return outputs
    cameras=torch.cat([inst.geo_camera for inst in refs]);frames=torch.cat([inst.geo_frame for inst in refs])
    valid=torch.cat([inst.geo_valid for inst in refs])
    cross=query.geo_camera[:,None]!=cameras[None,:]
    legal=cross&query.geo_valid[:,None]&valid[None,:]&((query.geo_frame[:,None]-frames[None,:]).abs()<=self.cfg.MAX_GEOMETRY_AGE)
    feature=legal.to(dtype=outputs[0].dtype)
    return [self.score.apply(output,feature,None,cross) for output in outputs]


def install(runtime):
    cfg=runtime.cfg
    if cfg.FEATURE_BACKBONE!='gmt' or cfg.HEAD_PATH or cfg.GEOMETRY_WEIGHT or cfg.ASYNC_WORKER or cfg.REUSE_TARGET_FEATURES or cfg.SPARSE_UPDATE_ENABLED:
        raise ValueError('native fixed prior requires no head, geometry, async, cache or sparse scheduler')
    runtime.native_prior_stats=dict(observed_groups=0,observed_targets=0,maximum_extra_vector_columns=0)
    runtime.observe=MethodType(observe,runtime);runtime.adapt=MethodType(adapt,runtime)
