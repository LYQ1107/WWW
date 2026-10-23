"""Explicit frontends for the detector-replacement experiment."""
import json
import time
from .gmt_detector_adapter import GMTDetectorAdapter,GMT_PYTHON,detection_rows
from .gmt_detection_cache import ImmutableGMTDetectionCache


class VerifiedGMTDetector:
    """Real GMT inference, checked against the shared immutable observation set."""
    def __init__(self,cache,device):
        self.cache=ImmutableGMTDetectionCache(cache)
        self.live=GMTDetectorAdapter(device=device,python=GMT_PYTHON)
        self.status='READY';self.last_feature_maps={};self.last_timings={}

    def set_context(self,*args,**kwargs):self.cache.set_context(*args,**kwargs)

    def infer(self,images):
        started=time.perf_counter();cached=self.cache.infer(images);actual=self.live.infer(images)
        if json.dumps(detection_rows(cached),sort_keys=True)!=json.dumps(detection_rows(actual),sort_keys=True):
            raise RuntimeError('Live detector differs from frozen shared candidates')
        self.last_timings={'detector_with_IPC_and_cache_verification_s':time.perf_counter()-started}
        self.last_feature_maps=self.live.last_feature_maps
        return actual

    def manifest(self):
        return {**self.live.manifest(),'raw_cache':self.cache.manifest(),'perception_cache_used':False,
                'shared_candidates_exactly_verified':True}

    def close(self):self.live.close()


def make_person_detector(frontend,device,config,weights,expose_intermediate=False,
                         raw_cache=None,mode='live_verified'):
    if frontend=='rtmdet':
        from .person_detector import PersonDetector
        return PersonDetector(config,weights,device,expose_intermediate_features=expose_intermediate)
    if frontend!='gmt_pre_gta':raise ValueError('Unknown detector frontend: '+str(frontend))
    if raw_cache is None:raise ValueError('GMT route requires the common immutable candidate cache')
    if mode=='replay':return ImmutableGMTDetectionCache(raw_cache)
    if mode=='live_verified':return VerifiedGMTDetector(raw_cache,device)
    raise ValueError('Unknown GMT execution mode')
