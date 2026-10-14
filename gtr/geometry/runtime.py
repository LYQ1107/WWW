"""Minimal optional evidence hook; original GTA, assignment and memory stay intact."""
import json,time,random
from pathlib import Path
import cv2,numpy as np,torch
from .pair_score_adapter import PairScoreAdapter,calibrated_cosine
from .roi_feature_extractor import PadMapping
from .geometry_confidence import estimate_background
from .geometry_alignment import soft_box_evidence

class GeometryRuntime:
    def __init__(self,cfg):
        self.cfg=cfg;self.vggt=None;self.geometry={};self.calls=0;self.vggt_calls=0;self.dinov2_calls=0;self.failed=0;self.seconds=0.;self.calibration=None
        self.score=PairScoreAdapter(cfg.ENABLED,cfg.FEATURE_WEIGHT if cfg.FEATURE_ENABLED else 0.,cfg.GEOMETRY_WEIGHT if cfg.GEOMETRY_ENABLED else 0.,cfg.MAX_RESIDUAL)
        self.active=cfg.ENABLED and (self.score.feature_weight!=0 or self.score.geometry_weight!=0)
        if self.active and self.score.feature_weight:
            if not cfg.CALIBRATION_PATH:raise ValueError('feature residual requires association-TRAIN scalar calibration')
            self.calibration=json.loads(Path(cfg.CALIBRATION_PATH).read_text())
            if self.calibration.get('source_split')!='association_train':raise ValueError('invalid calibration split')
            if self.calibration.get('symmetric',False)!=cfg.SYMMETRIZE_VIEW_ORDER:raise ValueError('calibration order mode mismatch')
            if cfg.FEATURE_LAYERS!=[self.calibration['layer']] or cfg.POOLING!=self.calibration['pooling']:raise ValueError('calibration feature mismatch')
            if self.calibration.get('backbone','vggt')!=cfg.FEATURE_BACKBONE:raise ValueError('calibration backbone mismatch')

    def reset(self):self.geometry={};self.calls=0;self.vggt_calls=0;self.dinov2_calls=0;self.failed=0;self.seconds=0.

    def observe(self,instances,records,frame):
        if not self.active:return
        # Pure GMT also jitters association ROIs at inference. New modules must
        # preserve Python, NumPy and CPU/CUDA RNG streams for paired validity.
        py_state=random.getstate();np_state=np.random.get_state()
        device=instances[0].reid_features.device
        index=device.index if device.index is not None else torch.cuda.current_device()
        with torch.random.fork_rng(devices=[index]):
            try:return self._observe_impl(instances,records,frame)
            finally:random.setstate(py_state);np.random.set_state(np_state)

    def _observe_impl(self,instances,records,frame):
        if not self.active:return
        levels=self.cfg.FEATURE_LAYERS;dim=self.cfg.FEATURE_DIM*len(levels)*(2 if self.cfg.SYMMETRIZE_VIEW_ORDER else 1)
        original_boxes=[]
        for v,(inst,r) in enumerate(zip(instances,records)):
            device=inst.reid_features.device;n=len(inst);h,w=inst.image_size
            boxes=inst.pred_boxes.tensor.detach().clone()*inst.pred_boxes.tensor.new_tensor([r['width']/w,r['height']/h]*2)
            original_boxes.append(boxes)
            inst.geo_vector=torch.zeros(n,dim,device=device)
            inst.geo_valid=torch.zeros(n,dtype=torch.bool,device=device)
            inst.geo_camera=torch.full((n,),int(r.get('view_id',v+1)),device=device,dtype=torch.long)
            inst.geo_frame=torch.full((n,),frame,device=device,dtype=torch.long)
            inst.geo_xyxy=boxes
        self.geometry={t:x for t,x in self.geometry.items() if frame-t<=self.cfg.MAX_GEOMETRY_AGE}
        if frame%self.cfg.UPDATE_INTERVAL:return
        start=time.perf_counter()
        try:
            if self.score.feature_weight:
                if self.vggt is None:
                    device=str(instances[0].reid_features.device)
                    if self.cfg.FEATURE_BACKBONE=='dinov2':
                        from .dinov2_adapter import FrozenDINOv2
                        self.vggt=FrozenDINOv2(self.cfg.DINOV2_CHECKPOINT,device=device)
                    else:
                        from .vggt_adapter import FrozenVGGT
                        self.vggt=FrozenVGGT(self.cfg.VGGT_CHECKPOINT,device=device,layers=levels)
                from vggt.utils.load_fn import load_and_preprocess_images
                images=load_and_preprocess_images([r['file_name'] for r in records],mode='pad')
                mapped=[PadMapping(r['width'],r['height']).boxes(b) for r,b in zip(records,original_boxes)]
                result,_,_=self.vggt.extract(images,mapped,modes=(self.cfg.POOLING,),dim=self.cfg.FEATURE_DIM)
                if self.cfg.FEATURE_BACKBONE=='dinov2':self.dinov2_calls+=1
                else:self.vggt_calls+=1
                reverse=None
                if self.cfg.SYMMETRIZE_VIEW_ORDER:
                    if len(instances)!=2:raise ValueError('symmetric mode currently supports exactly two cameras')
                    reverse,_,_=self.vggt.extract(images.flip(0),mapped[::-1],modes=(self.cfg.POOLING,),dim=self.cfg.FEATURE_DIM)
                    self.vggt_calls+=1
                for v,inst in enumerate(instances):
                    chunks=[result[f'layer_{l}/{self.cfg.POOLING}'][v].to(inst.reid_features.device) for l in levels]
                    if reverse is not None:chunks += [reverse[f'layer_{l}/{self.cfg.POOLING}'][1-v].to(inst.reid_features.device) for l in levels]
                    inst.geo_vector=torch.cat(chunks,dim=-1)/len(chunks)**.5
                    inst.geo_valid=torch.isfinite(inst.geo_vector).all(-1)&(inst.geo_vector.norm(dim=-1)>0)
            if self.score.geometry_weight and len(instances)==2:
                # Fundamental is only used for same-time coordinates. No stale-position projection.
                images=[cv2.imread(r['file_name'],cv2.IMREAD_GRAYSCALE) for r in records]
                result=estimate_background(*images,*[b.cpu().numpy() for b in original_boxes],method=self.cfg.GEOMETRY_METHOD,mask_targets=True)
                result['camera_order']=tuple(int(r.get('view_id',v+1)) for v,r in enumerate(records))
                self.geometry[frame]=result
            self.calls+=1
        except (RuntimeError,ValueError,OSError,cv2.error):
            self.failed+=1
            if not self.cfg.FALLBACK_TO_GMT:raise
            for inst in instances:inst.geo_valid.zero_()
        finally:self.seconds+=time.perf_counter()-start

    def adapt(self,outputs,instances,k):
        if not self.active:return outputs
        query=instances[k];refs=[x for i,x in enumerate(instances) if i!=k]
        if not query.has('geo_camera') or any(not x.has('geo_camera') for x in refs):return outputs
        cams=torch.cat([x.geo_camera for x in refs]);frames=torch.cat([x.geo_frame for x in refs]);boxes=torch.cat([x.geo_xyxy for x in refs]);vecs=torch.cat([x.geo_vector for x in refs]);valid=torch.cat([x.geo_valid for x in refs])
        cross=query.geo_camera[:,None]!=cams[None,:]
        feature=None;geo=None
        if self.score.feature_weight:
            feature=calibrated_cosine(vecs,query.geo_vector,self.calibration['slope'],self.calibration['intercept'])
            feature*=cross & query.geo_valid[:,None] & valid[None,:] & ((query.geo_frame[:,None]-frames[None,:]).abs()<=self.cfg.MAX_GEOMETRY_AGE)
        if self.score.geometry_weight:
            geo=outputs[0].new_zeros(outputs[0].shape)
            if len(query):
                t=int(query.geo_frame[0]);result=self.geometry.get(t)
                if result and result['valid']:
                    # F defined input view 1 -> 2; invert for reverse query.
                    same=frames==t
                    for camera in query.geo_camera.unique().tolist():
                        qi=query.geo_camera==camera;ri=same&(cams!=camera)
                        if qi.any() and ri.any():
                            F=result['F'] if camera==result['camera_order'][1] else result['F'].T
                            scores=soft_box_evidence(F,boxes[ri].cpu().numpy(),query.geo_xyxy[qi].cpu().numpy(),True,result['inlier_ratio'])
                            rows=qi.nonzero(as_tuple=True)[0];cols=ri.nonzero(as_tuple=True)[0]
                            geo[rows[:,None],cols[None,:]]=torch.as_tensor(scores,device=geo.device,dtype=geo.dtype)
        return [self.score.apply(output,feature,geo,cross) for output in outputs]
