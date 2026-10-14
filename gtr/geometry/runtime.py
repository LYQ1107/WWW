"""Minimal optional evidence hook; original GTA, assignment and memory stay intact."""
import json,time,random
from pathlib import Path
import cv2,numpy as np,torch
from .pair_score_adapter import PairScoreAdapter,calibrated_cosine,normalize_candidate_scores
from .roi_feature_extractor import PadMapping
from .geometry_confidence import estimate_background
from .geometry_alignment import soft_box_evidence,soft_plane_evidence

class GeometryRuntime:
    def __init__(self,cfg):
        self.cfg=cfg;self.vggt=None;self.geometry={};self.calls=0;self.vggt_calls=0;self.dinov2_calls=0;self.geometry_valid_calls=0;self.geometry_invalid_calls=0;self.failed=0;self.seconds=0.;self.calibration=None
        self.score=PairScoreAdapter(cfg.ENABLED,cfg.FEATURE_WEIGHT if cfg.FEATURE_ENABLED else 0.,cfg.GEOMETRY_WEIGHT if cfg.GEOMETRY_ENABLED else 0.,cfg.MAX_RESIDUAL)
        self.active=cfg.ENABLED and (self.score.feature_weight!=0 or self.score.geometry_weight!=0)
        self.head=None;self.head_checkpoint=None
        if self.active and cfg.UPDATE_INTERVAL<1:raise ValueError('update interval must be positive')
        if self.active and cfg.TOTAL_CALL_BUDGET>=0 and not cfg.SPARSE_UPDATE_ENABLED:
            raise ValueError('total call budget requires an explicit scheduling policy')
        if self.active and self.score.geometry_weight and cfg.GEOMETRY_METHOD not in {'orb','sift','orb_homography','sift_homography','vggt_camera'}:
            raise ValueError('unsupported online geometry method')
        if self.active and self.score.geometry_weight and cfg.GEOMETRY_METHOD=='vggt_camera' and cfg.FEATURE_BACKBONE!='vggt' and not cfg.ASYNC_WORKER:
            raise ValueError('VGGT camera geometry requires the VGGT encoder')
        if self.active and cfg.ASYNC_WORKER:
            if not self.score.geometry_weight or cfg.GEOMETRY_METHOD!='vggt_camera' or not cfg.ASYNC_GPU:
                raise ValueError('async mode requires explicit separate GPU and VGGT camera geometry')
        if self.active and self.score.feature_weight:
            if cfg.FEATURE_BACKBONE not in {'gmt','vggt','dinov2'}:raise ValueError('unsupported feature backbone')
            if cfg.FEATURE_BACKBONE=='gmt' and (cfg.SYMMETRIZE_VIEW_ORDER or cfg.FEATURE_LAYERS!=[-1] or cfg.PROJECTION_DIM!=0 or cfg.POOLING!='reid'):
                raise ValueError('GMT capacity control requires native ReID, one layer, no view symmetrization')
            if cfg.HEAD_PATH:
                self.head_checkpoint=torch.load(cfg.HEAD_PATH,map_location='cpu')
                h=self.head_checkpoint
                if h.get('source_split')!='association_train' or h['backbone']!=cfg.FEATURE_BACKBONE:raise ValueError('head source/backbone mismatch')
                projection_dim=cfg.FEATURE_DIM if cfg.PROJECTION_DIM<0 else cfg.PROJECTION_DIM
                if h['projection_dim']!=projection_dim or h['layers']!=cfg.FEATURE_LAYERS or h['pooling']!=cfg.POOLING:raise ValueError('head feature configuration mismatch')
                if h['settings']['symmetric']!=cfg.SYMMETRIZE_VIEW_ORDER:raise ValueError('head view-order mode mismatch')
                if h['feature_weight']!=cfg.FEATURE_WEIGHT or h['max_residual']!=cfg.MAX_RESIDUAL:raise ValueError('head residual configuration mismatch')
            else:
                self._load_scalar_calibration(cfg)
        self.reset()

    def _load_scalar_calibration(self,cfg):
        if not cfg.CALIBRATION_PATH:raise ValueError('feature residual requires association-TRAIN scalar calibration')
        self.calibration=json.loads(Path(cfg.CALIBRATION_PATH).read_text())
        if self.calibration.get('source_split')!='association_train':raise ValueError('invalid calibration split')
        if self.calibration.get('symmetric',False)!=cfg.SYMMETRIZE_VIEW_ORDER:raise ValueError('calibration order mode mismatch')
        if cfg.FEATURE_LAYERS!=self.calibration.get('layers',[self.calibration['layer']]) or cfg.POOLING!=self.calibration['pooling']:raise ValueError('calibration feature mismatch')
        if self.calibration.get('backbone','vggt')!=cfg.FEATURE_BACKBONE:raise ValueError('calibration backbone mismatch')
        if self.calibration.get('normalization','absolute') not in {'absolute','candidate_row_zscore'}:raise ValueError('unsupported score normalization')
        projection_dim=cfg.FEATURE_DIM if cfg.PROJECTION_DIM<0 else cfg.PROJECTION_DIM
        if self.calibration.get('dim',256)!=projection_dim:raise ValueError('calibration projection dimension mismatch')

    def reset(self):
        if getattr(self,'async_worker',None) is not None:self.async_worker.close()
        self.async_worker=None;self.temporal=None;self.async_trace=[];self.async_peak_gpu_bytes=0;self.async_shutdown_seconds=0.
        self.geometry={};self.calls=0;self.vggt_calls=0;self.dinov2_calls=0;self.geometry_valid_calls=0;self.geometry_invalid_calls=0;self.failed=0;self.seconds=0.
        self.scheduler=None;self.motion=None;self.schedule_trace=[];self.motion_failures=0
        self.pending_margin=1.;self.pending_handoff=False;self.identity_source_frame=None;self.last_success_frame=None;self.observed_frame=None
        self.update_cost=2 if self.score.feature_weight and self.cfg.FEATURE_BACKBONE=='vggt' and self.cfg.SYMMETRIZE_VIEW_ORDER else 1
        if self.active and self.cfg.ASYNC_WORKER:
            feature_cost=(2 if self.cfg.SYMMETRIZE_VIEW_ORDER else 1) if self.score.feature_weight and self.cfg.FEATURE_BACKBONE!='gmt' else 0
            self.update_cost=feature_cost+1
        self.budget_unit=('encoder_calls' if (self.score.feature_weight and self.cfg.FEATURE_BACKBONE!='gmt') or (self.score.geometry_weight and self.cfg.GEOMETRY_METHOD=='vggt_camera') else 'geometry_estimates' if self.score.geometry_weight else 'native_feature_observations')
        if self.active and self.cfg.SPARSE_UPDATE_ENABLED:
            from .sparse_scheduler import SparseScheduler
            from .update_signals import CausalBackgroundMotion
            self.scheduler=SparseScheduler(policy=self.cfg.UPDATE_POLICY,interval=self.cfg.UPDATE_INTERVAL,min_interval=self.cfg.MIN_UPDATE_INTERVAL,
                max_age=self.cfg.MAX_GEOMETRY_AGE,budget=self.cfg.COMPUTE_BUDGET//self.update_cost,window=self.cfg.BUDGET_WINDOW,seed=self.cfg.SCHEDULER_SEED,offset=self.cfg.UPDATE_OFFSET,
                total_budget=None if self.cfg.TOTAL_CALL_BUDGET<0 else self.cfg.TOTAL_CALL_BUDGET//self.update_cost)
            self.motion=CausalBackgroundMotion()
        if self.active and self.cfg.ASYNC_WORKER:
            from .update_signals import CausalBackgroundMotion
            from .temporal_geometry import TemporalGeometryConsumer
            if self.motion is None:self.motion=CausalBackgroundMotion()
            self.temporal=TemporalGeometryConsumer(self.cfg.MAX_GEOMETRY_AGE)

    def prepare_async(self):
        if not self.active or not self.cfg.ASYNC_WORKER:return
        import os
        main_gpu=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')[0]
        if main_gpu==self.cfg.ASYNC_GPU:raise ValueError('async worker must use a different physical GPU')
        if self.async_worker is not None:return
        from .process_worker import ProcessGeometryWorker,vggt_camera_factory
        self.async_worker=ProcessGeometryWorker(vggt_camera_factory,dict(checkpoint=self.cfg.VGGT_CHECKPOINT,vggt_source=self.cfg.VGGT_SOURCE),
            gpu=self.cfg.ASYNC_GPU,max_age=self.cfg.MAX_GEOMETRY_AGE,timeout=self.cfg.ASYNC_TIMEOUT,startup_timeout=self.cfg.ASYNC_STARTUP_TIMEOUT)

    def close_async(self):
        if self.async_worker is not None:self.async_shutdown_seconds+=self.async_worker.close()

    def async_report(self):
        if self.async_worker is None:return None
        return dict(encoder_calls=self.async_worker.encoder_calls,worker_stats=self.async_worker.stats,worker_error=self.async_worker.error,
            peak_gpu_bytes=max(self.async_peak_gpu_bytes,self.async_worker.ready_peak_gpu_bytes or 0) or None,shutdown_seconds=self.async_shutdown_seconds,worker_trace=self.async_worker.trace,
            consumption_trace=self.async_trace,propagation_stats=self.temporal.stats,
            scope='separate CUDA process; one pending payload; async startup overlaps main inference; shutdown wait charged; propagated F requires current independent pixel validation')

    def _consume_async(self,records,boxes,frame):
        if self.async_worker is None:raise RuntimeError('prepare_async must be called before streaming')
        message=self.async_worker.poll(frame)
        if message is not None:
            self.async_peak_gpu_bytes=max(self.async_peak_gpu_bytes,message['result'].get('peak_gpu_bytes',0))
            if message['result'].get('valid'):self.geometry_valid_calls+=1
            else:self.geometry_invalid_calls+=1
            self.temporal.receive(message,frame)
        candidate=self.temporal.candidate(frame)
        if candidate is None:return
        from .background_patch_matcher import validate_candidate
        images=[cv2.imread(r['file_name'],cv2.IMREAD_GRAYSCALE) for r in records]
        background=estimate_background(*images,*[b.cpu().numpy() for b in boxes],method='sift',mask_targets=True)
        validation=validate_candidate(candidate['F'],'F',background['points1'],background['points2'],[image.shape for image in images])
        candidate.update(validation);candidate['inlier_ratio']=candidate['inlier_ratio']*candidate['confidence_decay']
        self.geometry[frame]=candidate
        self.async_trace.append(dict(frame=frame,source_frame=candidate['source_frame'],completed_at=candidate['completed_at'],observed_at=time.monotonic(),
            valid=candidate['valid'],propagation_residual_processed_px=candidate['propagation_residual_processed_px'],validation_matches=candidate['validation_matches']))
        if candidate['valid']:self.last_success_frame=frame

    def record_identity_feedback(self,logits,instances,k,activate):
        if self.scheduler is None or self.cfg.UPDATE_POLICY not in {'identity','joint'}:return
        from .update_signals import identity_update_signal
        margin,handoff=identity_update_signal(logits,instances,k,activate)
        self.pending_margin=min(self.pending_margin,margin);self.pending_handoff|=handoff
        self.identity_source_frame=self.observed_frame

    def observe(self,instances,records,frame):
        if not self.active:return
        # Pure GMT also jitters association ROIs at inference. New modules must
        # preserve Python, NumPy and CPU/CUDA RNG streams for paired validity.
        py_state=random.getstate();np_state=np.random.get_state()
        device=instances[0].reid_features.device
        devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=='cuda' else []
        with torch.random.fork_rng(devices=devices):
            try:return self._observe_impl(instances,records,frame)
            finally:random.setstate(py_state);np.random.set_state(np_state)

    def _observe_impl(self,instances,records,frame):
        if not self.active:return
        levels=self.cfg.FEATURE_LAYERS
        projection_dim=self.cfg.FEATURE_DIM if self.cfg.PROJECTION_DIM<0 else self.cfg.PROJECTION_DIM
        per_layer_dim=projection_dim if projection_dim>0 else (1024 if self.cfg.FEATURE_BACKBONE=='dinov2' else 2048)
        if self.cfg.FEATURE_BACKBONE=='gmt':per_layer_dim=instances[0].reid_features.shape[-1]
        dim=per_layer_dim*len(levels)*(2 if self.cfg.SYMMETRIZE_VIEW_ORDER else 1)
        if not self.score.feature_weight:dim=0
        if self.head_checkpoint is not None:
            settings=self.head_checkpoint['settings'];dim=settings['output_dim']*(2 if settings['symmetric'] else 1)
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
        self.observed_frame=frame
        motion=0.;details=[]
        if self.motion is not None:
            try:motion,details=self.motion.observe(records,original_boxes)
            except (OSError,ValueError,cv2.error):self.motion_failures+=1
        if self.cfg.ASYNC_WORKER:
            self.temporal.add_motion(frame,details)
            try:self._consume_async(records,original_boxes,frame)
            except (OSError,ValueError,cv2.error):self.failed+=1;self.geometry.pop(frame,None)
        if self.scheduler is None:
            chosen=(frame-self.cfg.UPDATE_OFFSET)%self.cfg.UPDATE_INTERVAL==0
        else:
            age=float('inf') if self.last_success_frame is None else frame-self.last_success_frame
            available=sum(len(x) for x in instances)>0
            if self.cfg.ASYNC_WORKER and not self.score.feature_weight:
                available=available and self.async_worker.ready and not self.async_worker.disabled and self.async_worker.pending is None
            chosen=self.scheduler.should_update(frame,motion=motion,margin=self.pending_margin,handoff=self.pending_handoff,
                budget_available=available,state_age=age)
            self.schedule_trace.append(dict(frame=frame,chosen=chosen,motion=motion,motion_details=details,previous_identity_margin=self.pending_margin,
                identity_source_frame=self.identity_source_frame,handoff=self.pending_handoff,last_success_frame=self.last_success_frame,
                reserved_units_in_window=len(self.scheduler.calls)*self.update_cost,update_cost=self.update_cost,budget_unit=self.budget_unit))
            self.schedule_trace[-1]['reserved_total_units']=self.scheduler.total_count*self.update_cost
            self.pending_margin=1.;self.pending_handoff=False;self.identity_source_frame=None
        if not chosen:return
        start=time.perf_counter()
        camera_meta=None
        async_submitted=False
        try:
            if self.cfg.ASYNC_WORKER:
                payload=dict(records=[dict(file_name=r['file_name'],width=r['width'],height=r['height'],view_id=r['view_id']) for r in records],
                    boxes=[box.cpu().tolist() for box in original_boxes])
                async_submitted=self.async_worker.submit(frame,payload)
            if self.score.feature_weight:
                if self.cfg.FEATURE_BACKBONE=='gmt':
                    for inst in instances:
                        raw=inst.reid_features.float()
                        inst.geo_vector=torch.nn.functional.normalize(raw,dim=-1)
                        inst.geo_valid=torch.isfinite(raw).all(-1)&(raw.norm(dim=-1)>0)
                else:
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
                    if self.cfg.FEATURE_BACKBONE=='dinov2':self.dinov2_calls+=1
                    else:self.vggt_calls+=1
                    if self.cfg.FEATURE_BACKBONE=='vggt':
                        use_camera=self.score.geometry_weight!=0 and self.cfg.GEOMETRY_METHOD=='vggt_camera' and not self.cfg.ASYNC_WORKER
                        result,_,camera_meta=self.vggt.extract(images,mapped,modes=(self.cfg.POOLING,),dim=projection_dim,camera=use_camera)
                    else:
                        result,_,_=self.vggt.extract(images,mapped,modes=(self.cfg.POOLING,),dim=projection_dim)
                    reverse=None
                    if self.cfg.SYMMETRIZE_VIEW_ORDER:
                        if len(instances)!=2:raise ValueError('symmetric mode currently supports exactly two cameras')
                        self.vggt_calls+=1
                        reverse,_,_=self.vggt.extract(images.flip(0),mapped[::-1],modes=(self.cfg.POOLING,),dim=projection_dim)
                    for v,inst in enumerate(instances):
                        chunks=[result[f'layer_{l}/{self.cfg.POOLING}'][v].to(inst.reid_features.device) for l in levels]
                        if reverse is not None:chunks += [reverse[f'layer_{l}/{self.cfg.POOLING}'][1-v].to(inst.reid_features.device) for l in levels]
                        inst.geo_vector=torch.cat(chunks,dim=-1)/len(chunks)**.5
                        inst.geo_valid=torch.isfinite(inst.geo_vector).all(-1)&(inst.geo_vector.norm(dim=-1)>0)
                if self.head_checkpoint is not None:
                    if self.head is None:
                        from .lightweight_head import LowRankIdentityHead
                        head=LowRankIdentityHead(**self.head_checkpoint['settings'])
                        head.load_state_dict(self.head_checkpoint['state_dict'],strict=True)
                        self.head=head.to(instances[0].reid_features.device).eval().requires_grad_(False)
                    for inst in instances:
                        valid=inst.geo_valid
                        foundation=None if self.cfg.FEATURE_BACKBONE=='gmt' else inst.geo_vector
                        inst.geo_vector=self.head.encode(inst.reid_features,foundation)
                        inst.geo_valid=valid&torch.isfinite(inst.geo_vector).all(-1)&(inst.geo_vector.norm(dim=-1)>0)
            if self.score.geometry_weight and len(instances)==2 and not self.cfg.ASYNC_WORKER:
                # Fundamental is only used for same-time coordinates. No stale-position projection.
                images=[cv2.imread(r['file_name'],cv2.IMREAD_GRAYSCALE) for r in records]
                method='sift' if self.cfg.GEOMETRY_METHOD=='vggt_camera' else self.cfg.GEOMETRY_METHOD
                result=estimate_background(*images,*[b.cpu().numpy() for b in original_boxes],method=method,mask_targets=True)
                if self.cfg.GEOMETRY_METHOD=='vggt_camera':
                    if camera_meta is None:
                        if self.vggt is None:
                            from .vggt_adapter import FrozenVGGT
                            self.vggt=FrozenVGGT(self.cfg.VGGT_CHECKPOINT,device=str(instances[0].reid_features.device),layers=levels)
                        from vggt.utils.load_fn import load_and_preprocess_images
                        inputs=load_and_preprocess_images([r['file_name'] for r in records],mode='pad')
                        mapped=[PadMapping(r['width'],r['height']).boxes(b) for r,b in zip(records,original_boxes)]
                        for v,boxes in enumerate(mapped):
                            for x1,y1,x2,y2 in boxes.round().int().tolist():
                                inputs[v,:,max(0,y1):min(518,y2),max(0,x1):min(518,x2)]=1.
                        self.vggt_calls+=1
                        _,_,camera_meta=self.vggt.extract(inputs,mapped,modes=(),camera=True)
                    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
                    from .camera_geometry import validate_camera_fundamental
                    E,K=pose_encoding_to_extri_intri(camera_meta['pose'],(518,518))
                    mappings=[PadMapping(r['width'],r['height']) for r in records]
                    result=validate_camera_fundamental(E.numpy()[0],K.numpy()[0],mappings,result)
                result['camera_order']=tuple(int(r.get('view_id',v+1)) for v,r in enumerate(records))
                valid=result['H_valid'] if self.cfg.GEOMETRY_METHOD.endswith('_homography') else result['valid']
                if valid:self.geometry_valid_calls+=1
                else:self.geometry_invalid_calls+=1
                self.geometry[frame]=result
            if not self.cfg.ASYNC_WORKER or self.score.feature_weight or async_submitted:self.calls+=1
            if self.cfg.ASYNC_WORKER:self._consume_async(records,original_boxes,frame)
            if self.scheduler is not None:
                feature_valid=self.score.feature_weight and any(bool(inst.geo_valid.any()) for inst in instances)
                geometry_valid=self.score.geometry_weight and self.geometry.get(frame,{}).get('valid',False)
                if self.cfg.GEOMETRY_METHOD.endswith('_homography'):
                    geometry_valid=self.score.geometry_weight and self.geometry.get(frame,{}).get('H_valid',False)
                if feature_valid or geometry_valid:self.last_success_frame=frame
        except (RuntimeError,ValueError,OSError,cv2.error):
            self.failed+=1
            if not self.cfg.FALLBACK_TO_GMT:raise
            for inst in instances:
                inst.geo_valid.zero_();inst.geo_vector=inst.reid_features.new_zeros((len(inst),dim))
        finally:self.seconds+=time.perf_counter()-start

    def adapt(self,outputs,instances,k):
        if not self.active or not self.score.enabled:return outputs
        query=instances[k];refs=[x for i,x in enumerate(instances) if i!=k]
        if not self.score.feature_weight:
            if not len(query) or not query.has('geo_frame'):return outputs
            result=self.geometry.get(int(query.geo_frame[0]))
            key='H_valid' if self.cfg.GEOMETRY_METHOD.endswith('_homography') else 'valid'
            if result is None or not result.get(key,False):return outputs
        if not query.has('geo_camera') or any(not x.has('geo_camera') for x in refs):return outputs
        cams=torch.cat([x.geo_camera for x in refs]);frames=torch.cat([x.geo_frame for x in refs]);boxes=torch.cat([x.geo_xyxy for x in refs]);vecs=torch.cat([x.geo_vector for x in refs]);valid=torch.cat([x.geo_valid for x in refs])
        cross=query.geo_camera[:,None]!=cams[None,:]
        feature=None;geo=None
        if self.score.feature_weight:
            legal=cross & query.geo_valid[:,None] & valid[None,:] & ((query.geo_frame[:,None]-frames[None,:]).abs()<=self.cfg.MAX_GEOMETRY_AGE)
            if self.head_checkpoint is not None:
                feature=outputs[0].new_zeros(outputs[0].shape) if self.head is None else self.head.evidence(query.geo_vector,vecs)*legal
            elif self.calibration.get('normalization','absolute')=='candidate_row_zscore':
                raw=query.geo_vector.float()@vecs.float().T
                raw,legal=normalize_candidate_scores(raw,legal)
                feature=torch.tanh(self.calibration['slope']*raw+self.calibration['intercept'])*legal
            else:
                feature=calibrated_cosine(vecs,query.geo_vector,self.calibration['slope'],self.calibration['intercept'])*legal
        if self.score.geometry_weight:
            geo=outputs[0].new_zeros(outputs[0].shape)
            if len(query):
                t=int(query.geo_frame[0]);result=self.geometry.get(t)
                if result and (result['H_valid'] if self.cfg.GEOMETRY_METHOD.endswith('_homography') else result['valid']):
                    # F defined input view 1 -> 2; invert for reverse query.
                    same=frames==t
                    for camera in query.geo_camera.unique().tolist():
                        qi=query.geo_camera==camera;ri=same&(cams!=camera)
                        if qi.any() and ri.any():
                            if self.cfg.GEOMETRY_METHOD.endswith('_homography'):
                                H=result['H'] if camera==result['camera_order'][1] else np.linalg.inv(result['H'])
                                scores=soft_plane_evidence(H,boxes[ri].cpu().numpy(),query.geo_xyxy[qi].cpu().numpy(),True,result['H_inlier_ratio'])
                            else:
                                F=result['F'] if camera==result['camera_order'][1] else result['F'].T
                                scores=soft_box_evidence(F,boxes[ri].cpu().numpy(),query.geo_xyxy[qi].cpu().numpy(),True,result['inlier_ratio'])
                            rows=qi.nonzero(as_tuple=True)[0];cols=ri.nonzero(as_tuple=True)[0]
                            geo[rows[:,None],cols[None,:]]=torch.as_tensor(scores,device=geo.device,dtype=geo.dtype)
        return [self.score.apply(output,feature,geo,cross) for output in outputs]
