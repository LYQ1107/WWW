from types import SimpleNamespace
import pytest
import torch
from worldmot.models.constant_position_cache import ConstantPositionCache


def test_constant_grid_values_are_exact_and_each_shape_computed_once():
    calls=[]
    def original(dim,grid_size,return_grid=False):
        calls.append((dim,grid_size))
        return torch.arange(dim*grid_size[0]*grid_size[1],dtype=torch.float32).reshape(1,dim,*grid_size)
    cache=ConstantPositionCache(original,'cpu')
    expected=original(4,(3,5))
    assert torch.equal(cache.get(4,(3,5)),expected)
    assert torch.equal(cache.get(4,(3,5)),expected)
    assert len(calls)==2 and cache.hits==1 and cache.misses==1
    assert cache.get(4,(2,2)).shape==(1,4,2,2)


def test_original_binding_is_restored_on_failure_and_off_has_no_cache_hits():
    original=lambda dim,grid_size,return_grid=False:torch.zeros(1,dim,*grid_size)
    module=SimpleNamespace(get_2d_sincos_pos_embed=original)
    cache=ConstantPositionCache(original,'cpu')
    with pytest.raises(RuntimeError):
        with cache.apply(module,True):
            module.get_2d_sincos_pos_embed(4,(2,2))
            raise RuntimeError('forward failed')
    assert module.get_2d_sincos_pos_embed is original
    with cache.apply(module,False):module.get_2d_sincos_pos_embed(4,(2,2))
    assert cache.misses==1 and cache.hits==0
