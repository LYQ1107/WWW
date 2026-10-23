"""Memoize only image-independent sinusoidal grids used by official TrackHead."""
from collections import OrderedDict
from contextlib import contextmanager
from threading import RLock

_HEAD_FUNCTION_LOCK=RLock()


class ConstantPositionCache:
    def __init__(self,function,device,max_entries=4):
        self.function=function;self.device=str(device);self.max_entries=int(max_entries)
        self.values=OrderedDict();self.hits=0;self.misses=0

    def get(self,embed_dim,grid_size,return_grid=False):
        if return_grid:return self.function(embed_dim,grid_size,return_grid=True)
        grid=tuple(grid_size) if isinstance(grid_size,tuple) else (grid_size,grid_size)
        key=(int(embed_dim),grid)
        if key not in self.values:
            self.values[key]=self.function(embed_dim,grid_size).to(self.device)
            self.misses+=1
            if len(self.values)>self.max_entries:self.values.popitem(last=False)
        else:self.hits+=1
        self.values.move_to_end(key)
        return self.values[key]

    @contextmanager
    def apply(self,module,enabled=True):
        # The binding is local to this process and restored even on failure.
        # OFF calls also take the lock to prevent an optimized thread from
        # changing another model's baseline call concurrently.
        with _HEAD_FUNCTION_LOCK:
            previous=module.get_2d_sincos_pos_embed
            module.get_2d_sincos_pos_embed=self.get if enabled else self.function
            try:yield
            finally:module.get_2d_sincos_pos_embed=previous

    def manifest(self):
        return {'hits':self.hits,'misses':self.misses,'entries':len(self.values),
                'bytes':sum(v.numel()*v.element_size() for v in self.values.values()),
                'cached_content':'fixed sinusoidal grid; no image/depth/track/identity data'}
