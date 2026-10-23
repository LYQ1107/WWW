"""Exact-request replay for TRAIN calibration only; never system E2E timing."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from worldmot.models.person_detector import sha256_file
from worldmot.models.vggt_point_backend import PointTracks,VGGTPointBackend
from scripts.cache_fulltrain import atomic_json


class RequestCache:
    def __init__(self,backend,root,signature):
        self.backend=backend;self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.signature=signature;self.hits=0;self.misses=0

    def track(self,images,query_points,*,image_keys,reference_key,current_timestamp):
        started=time.perf_counter()
        points=VGGTPointBackend.validate_request(images,query_points,image_keys,reference_key,current_timestamp)
        request={'signature':self.signature,'image_keys':[list(k) for k in image_keys],
            'reference_key':list(reference_key),'query_shape':list(points.shape),
            'query_float32_sha256':hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest(),
            'images':[{'shape':list(np.asarray(im).shape),'dtype':str(np.asarray(im).dtype),
                'sha256':hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest()} for im in images]}
        key=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
        folder=self.root/key[:2];folder.mkdir(exist_ok=True);path=folder/(key+'.npz');meta=path.with_suffix('.json')
        if path.exists() and meta.exists():
            record=json.loads(meta.read_text())
            if record['request']!=request or sha256_file(path)!=record['sha256']:raise RuntimeError('Exact point-request cache integrity failed')
            with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
            self.hits+=1
            return PointTracks(arrays['xy'],arrays['visibility'],arrays['confidence'],arrays['valid'],
                list(image_keys),tuple(reference_key),record['transforms'],
                {'request_replay_s':time.perf_counter()-started,'aggregator_s':0.,'track_head_s':0.},
                {'status':'EXACT_TRAIN_REQUEST_REPLAY','forward_count':0,'request_sha256':key,
                 'origin_forward_diagnostics':record['diagnostics']})
        if path.exists() or meta.exists():raise RuntimeError('Preserve incomplete request-cache artifact before resuming: '+str(path))
        prediction=self.backend.track(images,points,image_keys=image_keys,reference_key=reference_key,current_timestamp=current_timestamp)
        temporary=path.with_name(path.stem+f'.tmp_{os.getpid()}.npz')
        np.savez_compressed(temporary,xy=prediction.xy,visibility=prediction.visibility,
            confidence=prediction.confidence,valid=prediction.valid)
        temporary.replace(path)
        atomic_json(meta,{'request':request,'sha256':sha256_file(path),'transforms':prediction.transforms,
            'origin_timings':prediction.timings,'diagnostics':prediction.diagnostics,
            'GT_or_identity_labels':False})
        self.misses+=1;return prediction

    def manifest(self):
        return {'root':str(self.root),'hits':self.hits,'misses':self.misses,
            'timing_scope':'TRAIN exact-request association replay; NOT E2E FPS',
            'key_includes':'original RGB bytes, camera/timestamp order, exact FP32 query bytes, model/source/precision signature',
            'GT_or_identity_labels':False}
