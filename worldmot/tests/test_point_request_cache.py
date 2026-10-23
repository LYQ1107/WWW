from types import SimpleNamespace
import numpy as np
import pytest
from scripts.point_request_cache import RequestCache
from worldmot.models.vggt_point_backend import PointTracks


class Backend:
    def __init__(self):self.calls=0
    def track(self,images,queries,**kw):
        self.calls+=1;n=len(queries);s=len(images)
        return PointTracks(np.tile(queries[None],(s,1,1)),np.ones((s,n)),np.ones((s,n)),
            np.ones((s,n),bool),kw['image_keys'],kw['reference_key'],[],{'aggregator_s':1.},{'forward_count':1})


def test_exact_cache_separates_rgb_queries_and_checks_future(tmp_path):
    backend=Backend();cache=RequestCache(backend,tmp_path,{'weights':'verified'})
    images=[np.zeros((10,10,3),np.uint8)]*2;points=np.array([[4.,5.]],np.float32)
    kw={'image_keys':[(1,1),(2,2)],'reference_key':(1,1),'current_timestamp':2}
    first=cache.track(images,points,**kw);second=cache.track(images,points,**kw)
    assert np.array_equal(first.xy,second.xy) and backend.calls==1 and cache.hits==1
    cache.track([images[0],images[1]+1],points,**kw)
    cache.track(images,points+1,**kw)
    assert backend.calls==3
    with pytest.raises(ValueError):cache.track(images,points,**{**kw,'current_timestamp':1})


def test_corrupted_point_cache_is_never_used(tmp_path):
    cache=RequestCache(Backend(),tmp_path,{});images=[np.zeros((10,10,3),np.uint8)]
    kw={'image_keys':[(1,1)],'reference_key':(1,1),'current_timestamp':1}
    cache.track(images,[[2.,3.]],**kw)
    next(tmp_path.rglob('*.npz')).write_bytes(b'corrupted')
    with pytest.raises(RuntimeError):cache.track(images,[[2.,3.]],**kw)
