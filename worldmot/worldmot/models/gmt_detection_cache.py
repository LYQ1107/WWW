"""Checksum-checked original GMT candidate replay, with no identity labels."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .person_detector import PersonDetection,sha256_file


class ImmutableGMTDetectionCache:
    def __init__(self,root):
        self.root=Path(root)
        self.lock_path=self.root/'DETECTOR_LOCK.json'
        self.lock=json.loads(self.lock_path.read_text())
        self.status='READY'
        self.last_feature_maps={}
        self.last_timings={}
        self.last_roi_features={}
        self.last_diagnostics={}
        self._loaded_key=None
        self._arrays=None
        self._context=None

    def manifest(self):
        return {'status':self.status,'source':'gmt_pre_gta_detector',
            'cache':str(self.root.resolve()),'cache_lock_sha256':sha256_file(self.lock_path),
            'detector_lock':self.lock,'timing_label':'association_replay_FPS',
            'GT_in_cache':False,'track_ids_in_cache':False}

    def set_context(self,scene_id,timestamp,image_ids=None):
        self._context=(int(scene_id),int(timestamp),image_ids)

    def infer(self,images_by_camera):
        if self._context is None:raise RuntimeError('Cache requires explicit scene/timestamp context')
        scene,timestamp,image_ids=self._context
        return self.get(scene,timestamp,images_by_camera,image_ids)

    def get(self,scene,timestamp,images=None,image_ids=None):
        start=time.perf_counter()
        directory=self.root/f'scene_{int(scene):02d}'
        manifest=json.loads((directory/'manifest.json').read_text())
        if manifest.get('status')!='COMPLETE' or manifest['signature']!=self.lock:
            raise RuntimeError('Replay requires a complete immutable scene with matching detector lock')
        candidates=[s for s in manifest['shards'] if int(timestamp) in s['timestamps']]
        if len(candidates)!=1:raise KeyError((scene,timestamp))
        shard=candidates[0];path=directory/shard['file']
        key=(int(scene),shard['file'])
        if key!=self._loaded_key:
            if sha256_file(path)!=shard['sha256']:raise RuntimeError('Shared detector cache checksum mismatch')
            self._arrays=dict(np.load(path,allow_pickle=False))
            self._entry=json.loads(path.with_suffix('.json').read_text())
            self._loaded_key=key
        at=list(self._arrays['timestamps']).index(int(timestamp))
        record=self._entry['records'][at]
        if image_ids is not None:
            if {str(k):int(v) for k,v in image_ids.items()}!=record['image_ids']:
                raise ValueError('Image IDs do not match cached synchronized timestamp')
        if images is not None:
            if set(map(str,images))!=set(record['rgb_sha256']):raise ValueError('Camera set differs from cache')
            for v,im in images.items():
                if hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest()!=record['rgb_sha256'][str(v)]:
                    raise ValueError('RGB pixels do not match the original detector input')
        a=self._arrays;lo,hi=a['frame_offsets'][at:at+2]
        result={int(v):[] for v in record['image_ids']};self.last_roi_features={}
        for i in range(int(lo),int(hi)):
            v,d=int(a['camera_ids'][i]),int(a['detection_ids'][i])
            det=PersonDetection(v,d,tuple(map(float,a['bbox'][i])),float(a['scores'][i]),source='gmt_pre_gta_detector')
            det.detector_roi_feature=a['roi_features'][i].copy()
            result[v].append(det);self.last_roi_features[v,d]=det.detector_roi_feature
        self.last_timings={'replay_s':time.perf_counter()-start}
        self.last_diagnostics={**record,'raw_candidate_shard_sha256':shard['sha256']}
        return result
