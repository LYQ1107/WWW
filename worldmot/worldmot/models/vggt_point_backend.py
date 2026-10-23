"""Official pretrained VGGT-1B Aggregator + TrackHead, current/past RGB only."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from .person_detector import sha256_file
from ..geometry.vggt_point_coordinates import VGGTPointCoordinates
from .constant_position_cache import ConstantPositionCache

VGGT_SOURCE = '/data1/liuyeqiang/WWW_vggt_geo_runtime/sources/vggt'
VGGT_WEIGHTS = '/data1/liuyeqiang/WWW_vggt_geo_runtime/weights/vggt_1b.pt'
VGGT_1B_SHA256 = 'd15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0'


@dataclass
class PointTracks:
    xy: np.ndarray
    visibility: np.ndarray
    confidence: np.ndarray
    valid: np.ndarray
    image_keys: list
    reference_key: tuple
    transforms: list
    timings: dict
    diagnostics: dict


class VGGTPointBackend:
    def __init__(self, source=VGGT_SOURCE, checkpoint=VGGT_WEIGHTS, device='cuda:0', cache_position_encoding=False):
        import torch
        self.source,self.checkpoint,self.device = Path(source).resolve(),Path(checkpoint).resolve(),device
        started=time.perf_counter()
        digest=sha256_file(self.checkpoint)
        if digest != VGGT_1B_SHA256:
            raise RuntimeError('Official VGGT-1B checkpoint SHA mismatch; Omega/substitute weights forbidden')
        sys.path.insert(0,str(self.source))
        from vggt.models.vggt import VGGT
        import vggt.heads.track_modules.base_track_predictor as track_module
        from vggt.heads.track_modules.utils import get_2d_sincos_pos_embed
        self.track_module=track_module
        self.cache_position_encoding=bool(cache_position_encoding)
        self.position_cache=ConstantPositionCache(get_2d_sincos_pos_embed,device)
        self.model=VGGT(enable_camera=False,enable_point=False,enable_depth=False,enable_track=True)
        payload=torch.load(self.checkpoint,map_location='cpu',weights_only=True,mmap=True)
        state={k:v for k,v in payload.items() if k.startswith(('aggregator.','track_head.'))}
        if not any(k.startswith('track_head.') for k in state):
            raise RuntimeError('No pretrained TrackHead parameters')
        self.model.load_state_dict(state,strict=True)
        self.model.to(device).eval().requires_grad_(False)
        del state,payload
        self.status='READY'
        self.last_timings={}
        self._manifest={
            'model':'facebook/VGGT-1B','checkpoint':str(self.checkpoint),'checkpoint_sha256':digest,
            'official_weight_pointer':'https://huggingface.co/facebook/VGGT-1B/raw/main/model.pt',
            'source':str(self.source),
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=self.source,text=True).strip(),
            'source_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=self.source,text=True)),
            'source_sha256':{n:sha256_file(self.source/n) for n in [
                'vggt/models/vggt.py','vggt/models/aggregator.py','vggt/heads/track_head.py',
                'vggt/heads/track_modules/base_track_predictor.py','vggt/utils/load_fn.py']},
            'strict_aggregator_track_load':True,'trained_in_this_experiment':False,
            'enabled_heads':['track_head'],'parameter_count':sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters':0,'aggregator_precision':'autocast float16 on V100',
            'track_head_precision':'float32','preprocessing':'official 518 pad; RGB in [0,1]',
            'initialization_seconds':time.perf_counter()-started,'torch':torch.__version__,
        }

    def manifest(self):
        return {**self._manifest,'status':self.status,'device':self.device,
                'constant_position_cache_enabled':self.cache_position_encoding,
                'constant_position_cache':self.position_cache.manifest()}

    @staticmethod
    def validate_request(images,query_points,image_keys,reference_key,current_timestamp):
        points=np.asarray(query_points,np.float32).reshape(-1,2)
        if len(images)!=len(image_keys) or not image_keys or tuple(image_keys[0])!=tuple(reference_key):
            raise ValueError('First image must be the actual source of the query points')
        if any(k[1]>current_timestamp for k in image_keys):
            raise ValueError('Future image requested')
        if not np.isfinite(points).all():
            raise ValueError('Nonfinite query points')
        transform=VGGTPointCoordinates.from_shape(np.asarray(images[0]).shape)
        if not transform.valid_original(points).all():
            raise ValueError('Query points must lie in the actual source image')
        return points

    def track(self, images, query_points, *, image_keys, reference_key, current_timestamp):
        import torch
        started=time.perf_counter()
        points=self.validate_request(images,query_points,image_keys,reference_key,current_timestamp)
        transforms=[VGGTPointCoordinates.from_shape(np.asarray(im).shape) for im in images]
        s,n=len(images),len(points)
        if not n:
            return PointTracks(np.empty((s,0,2),np.float32),np.empty((s,0),np.float32),
                np.empty((s,0),np.float32),np.empty((s,0),bool),list(image_keys),tuple(reference_key),
                [t.manifest() for t in transforms],{'e2e_s':time.perf_counter()-started},
                {'status':'EMPTY_QUERY_NO_FORWARD','forward_count':0})
        values=np.stack([t.preprocess(im) for t,im in zip(transforms,images)])
        tensor=torch.from_numpy(values)[None].to(self.device)
        queries=torch.from_numpy(transforms[0].to_network(points).astype(np.float32))[None].to(self.device)
        use_cuda=str(self.device).startswith('cuda')
        if use_cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        pre_s=time.perf_counter()-started
        with torch.inference_mode():
            begin=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.float16,enabled=use_cuda):
                tokens,patch_start=self.model.aggregator(tensor)
            if use_cuda:torch.cuda.synchronize(self.device)
            aggregator_s=time.perf_counter()-begin
            begin=time.perf_counter()
            # Keep tracker coordinates, DPT and correlation refinement stable
            # in FP32 while using the official pretrained parameters.
            with torch.autocast('cuda',enabled=False):
                with self.position_cache.apply(self.track_module,self.cache_position_encoding):
                    coords,vis,conf=self.model.track_head([x.float() if x is not None else None for x in tokens],tensor,
                        patch_start_idx=patch_start,query_points=queries)
            if use_cuda:torch.cuda.synchronize(self.device)
            head_s=time.perf_counter()-begin
            begin=time.perf_counter()
            net=coords[-1][0].float().cpu().numpy()
            visibility=vis[0].float().cpu().numpy()
            confidence=conf[0].float().cpu().numpy()
        if not (np.isfinite(net).all() and np.isfinite(visibility).all() and np.isfinite(confidence).all()):
            raise FloatingPointError('Official TrackHead returned nonfinite values')
        valid=np.stack([t.valid_network(x) for t,x in zip(transforms,net)])
        original=np.stack([t.to_original(x) for t,x in zip(transforms,net)]).astype(np.float32)
        error=float(np.max(np.abs(original[0]-points)))
        if error>1e-3:
            raise RuntimeError(f'First-image query anchor changed: {error}')
        self.last_timings={'preprocess_s':pre_s,'aggregator_s':aggregator_s,'track_head_s':head_s,
            'postprocess_s':time.perf_counter()-begin,'e2e_s':time.perf_counter()-started}
        return PointTracks(original,visibility,confidence,valid,list(image_keys),tuple(reference_key),
            [t.manifest() for t in transforms],dict(self.last_timings),
            {'status':'REAL_PRETRAINED_FORWARD','forward_count':1,'source_max_error_pixels':error,
             'query_points':n,'peak_GPU_bytes':torch.cuda.max_memory_allocated(self.device) if use_cuda else None})
