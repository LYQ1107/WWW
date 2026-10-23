"""Bounded, bit-exact TRAIN request replay with checked asynchronous persistence."""
from collections import OrderedDict
import concurrent.futures
import copy
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from worldmot.models.person_detector import sha256_file
from worldmot.models.vggt_point_backend import PointTracks,VGGTPointBackend
from scripts.cache_fulltrain import atomic_json


class RamRequestCache:
    def __init__(self,backend,root,signature,max_memory_bytes=1<<30,io_workers=8):
        self.backend=backend;self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.signature=signature;self.limit=int(max_memory_bytes);self.values=OrderedDict();self.bytes=0
        self.hits=0;self.disk_hits=0;self.misses=0;self.preloaded=0;self.quarantined=[]
        self.pool=concurrent.futures.ThreadPoolExecutor(max_workers=io_workers)
        self.pending={};self.completed_writes=0

    def _paths(self,key):
        path=self.root/key[:2]/(key+'.npz');return path,path.with_suffix('.json')

    def _remember(self,key,record,arrays):
        for x in arrays.values():x.setflags(write=False)
        size=sum(x.nbytes for x in arrays.values())
        if key in self.values:self.bytes-=self.values.pop(key)[2]
        self.values[key]=(record,arrays,size);self.bytes+=size
        while self.values and self.bytes>self.limit:
            _,(_,_,n)=self.values.popitem(last=False);self.bytes-=n

    def _load(self,meta):
        record=json.loads(meta.read_text());path=meta.with_suffix('.npz')
        if record['request']['signature']!=self.signature:return None
        key=hashlib.sha256(json.dumps(record['request'],sort_keys=True).encode()).hexdigest()
        if meta.stem!=key or record.get('GT_or_identity_labels') is not False:
            raise RuntimeError('Invalid or identity-bearing point request metadata')
        if sha256_file(path)!=record['sha256']:raise RuntimeError('Published point request checksum mismatch')
        with np.load(path,allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
        if set(arrays)!={'xy','visibility','confidence','valid'}:raise RuntimeError('Unexpected point request array')
        s=len(record['request']['images']);n=record['request']['query_shape'][0]
        if arrays['xy'].shape!=(s,n,2) or any(arrays[k].shape!=(s,n) for k in ['visibility','confidence','valid']):
            raise RuntimeError('Published point request shape mismatch')
        return key,record,arrays

    def preload(self):
        paths=list(self.root.glob('*/*.json'))
        for value in self.pool.map(self._load,paths):
            if value is not None:
                key,record,arrays=value;self._remember(key,record,arrays);self.preloaded+=1
        return self.preloaded

    def _persist(self,key,record,arrays):
        path,meta=self._paths(key);path.parent.mkdir(exist_ok=True)
        temporary=path.with_name(path.stem+f'.tmp_{os.getpid()}.npz')
        np.savez_compressed(temporary,**arrays);temporary.replace(path)
        atomic_json(meta,{**record,'sha256':sha256_file(path)})

    def flush(self):
        for key,future in list(self.pending.items()):
            future.result();self.pending.pop(key);self.completed_writes+=1

    def close(self):
        try:self.flush()
        finally:self.pool.shutdown(wait=True)

    def _replay(self,key,record,arrays,request,started,origin):
        if record['request']!=request:raise RuntimeError('Exact RAM point request differs')
        return PointTracks(arrays['xy'],arrays['visibility'],arrays['confidence'],arrays['valid'],
            [tuple(k) for k in request['image_keys']],tuple(request['reference_key']),copy.deepcopy(record['transforms']),
            {'request_replay_s':time.perf_counter()-started,'aggregator_s':0.,'track_head_s':0.},
            {'status':origin,'forward_count':0,'request_sha256':key,'origin_forward_diagnostics':record['diagnostics']})

    def track(self,images,query_points,*,image_keys,reference_key,current_timestamp):
        started=time.perf_counter()
        points=VGGTPointBackend.validate_request(images,query_points,image_keys,reference_key,current_timestamp)
        # This is deliberately the original persisted request schema, so old
        # real forwards can be reused without recomputing or changing arrays.
        request={'signature':self.signature,'image_keys':[list(k) for k in image_keys],
            'reference_key':list(reference_key),'query_shape':list(points.shape),
            'query_float32_sha256':hashlib.sha256(np.ascontiguousarray(points).tobytes()).hexdigest(),
            'images':[{'shape':list(np.asarray(im).shape),'dtype':str(np.asarray(im).dtype),
                'sha256':hashlib.sha256(np.ascontiguousarray(im).tobytes()).hexdigest()} for im in images]}
        key=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
        if key in self.values:
            record,arrays,_=self.values[key];self.values.move_to_end(key);self.hits+=1
            return self._replay(key,record,arrays,request,started,'EXACT_TRAIN_RAM_REPLAY')
        path,meta=self._paths(key)
        if key in self.pending:self.pending[key].result()
        if meta.exists():
            loaded=self._load(meta)
            if loaded is None:raise RuntimeError('Exact key namespace differs')
            _,record,arrays=loaded;self._remember(key,record,arrays);self.disk_hits+=1
            return self._replay(key,record,arrays,request,started,'EXACT_TRAIN_DISK_REPLAY')
        if path.exists():
            # An interrupted writer may publish NPZ before its metadata.
            # Preserve the orphan; never use it as a checked prediction.
            archive=self.root/'.interrupted_requests'/f'{key}_{time.time_ns()}_{os.getpid()}'
            archive.mkdir(parents=True,exist_ok=False);path.replace(archive/path.name)
            self.quarantined.append(str(archive))
        prediction=self.backend.track(images,points,image_keys=image_keys,reference_key=reference_key,current_timestamp=current_timestamp)
        arrays={k:getattr(prediction,k) for k in ['xy','visibility','confidence','valid']}
        record={'request':request,'transforms':prediction.transforms,'origin_timings':prediction.timings,
            'diagnostics':prediction.diagnostics,'GT_or_identity_labels':False}
        self._remember(key,record,arrays)
        self.pending[key]=self.pool.submit(self._persist,key,record,arrays);self.misses+=1
        if len(self.pending)>=256:self.flush()
        return prediction

    def manifest(self):
        return {'root':str(self.root),'hits':self.hits,'disk_hits':self.disk_hits,'misses':self.misses,
            'preloaded_verified_requests':self.preloaded,'RAM_bytes':self.bytes,'RAM_limit_bytes':self.limit,
            'pending_writes':len(self.pending),'completed_async_writes':self.completed_writes,
            'quarantined_orphan_requests':self.quarantined,
            'timing_scope':'TRAIN exact-request association replay; NOT E2E FPS',
            'key_includes':'original RGB bytes, camera/timestamp order, exact FP32 query bytes, unchanged model/source/precision signature',
            'GT_or_identity_labels':False,'completed_window_requires_persistence_flush':True}
