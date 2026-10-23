import numpy as np
import pytest
from scripts.point_request_cache import RequestCache
from scripts.ram_point_request_cache import RamRequestCache
from worldmot.models.vggt_point_backend import PointTracks


class Backend:
    def __init__(self):self.calls=0
    def track(self,images,queries,**kw):
        self.calls+=1;s=len(images);n=len(queries)
        return PointTracks(np.tile(queries[None],(s,1,1)),np.ones((s,n),np.float32),np.ones((s,n),np.float32),
            np.ones((s,n),bool),kw['image_keys'],kw['reference_key'],[],{'aggregator_s':1.},{'forward_count':1})


def inputs():
    return [np.zeros((10,10,3),np.uint8)],np.array([[2.,3.]],np.float32),dict(image_keys=[(1,1)],reference_key=(1,1),current_timestamp=1)


def test_original_real_storage_schema_preloads_exactly_into_readonly_ram(tmp_path):
    backend=Backend();images,points,kw=inputs();signature={'weights':'verified','source':'frozen'}
    before=RequestCache(backend,tmp_path,signature).track(images,points,**kw)
    ram=RamRequestCache(backend,tmp_path,signature)
    try:
        assert ram.preload()==1
        after=ram.track(images,points,**kw)
        assert backend.calls==1 and ram.hits==1
        for key in ['xy','visibility','confidence','valid']:
            assert np.array_equal(getattr(before,key),getattr(after,key))
            assert not getattr(after,key).flags.writeable
        with pytest.raises(ValueError):ram.track(images,points,**{**kw,'current_timestamp':0})
    finally:ram.close()


def test_async_miss_flush_is_bit_exact_for_original_disk_reader_and_lru(tmp_path):
    # Keep the bound below this request's actual array footprint so the
    # second lookup exercises the checked disk-replay path rather than RAM.
    backend=Backend();images,points,kw=inputs();ram=RamRequestCache(backend,tmp_path,{},max_memory_bytes=1)
    try:
        before=ram.track(images,points,**kw);ram.flush()
        after=RequestCache(backend,tmp_path,{}).track(images,points,**kw)
        assert backend.calls==1 and ram.completed_writes==1 and ram.bytes<=20
        for key in ['xy','visibility','confidence','valid']:assert np.array_equal(getattr(before,key),getattr(after,key))
        ram.track(images,points,**kw)
        assert backend.calls==1 and ram.disk_hits==1
    finally:ram.close()


def test_published_corruption_is_rejected_and_unpublished_orphan_is_preserved(tmp_path):
    backend=Backend();images,points,kw=inputs();old=RequestCache(backend,tmp_path,{})
    old.track(images,points,**kw);path=next(tmp_path.rglob('*.npz'));path.write_bytes(b'corrupt')
    ram=RamRequestCache(backend,tmp_path,{})
    try:
        with pytest.raises(RuntimeError):ram.preload()
    finally:ram.close()
    path.with_suffix('.json').unlink()
    ram=RamRequestCache(backend,tmp_path,{})
    try:
        ram.track(images,points,**kw);ram.flush()
        assert len(ram.quarantined)==1 and backend.calls==2
        assert list((tmp_path/'.interrupted_requests').rglob('*.npz'))
    finally:ram.close()
