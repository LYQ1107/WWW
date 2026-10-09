"""Cache exact normalized past tokens; never change floating summation order."""
import weakref
import torch.nn.functional as F
from gtr.modeling.jev_stage2.memory import IdentityMemory

class CachedIdentityMemory(IdentityMemory):
    def __init__(self):
        super().__init__();self._history_cache={};self.cache_hits=0;self.cache_misses=0
    def update(self,ref,*args,**kwargs):
        self._history_cache.pop(int(ref),None)
        return super().update(ref,*args,**kwargs)
    def load_state_dict(self,state):
        super().load_state_dict(state);self._history_cache={};self.cache_hits=0;self.cache_misses=0
    def history_values(self,ref,g,meta,view):
        visual=g.reid_features[:,:1024]
        signature=(g.reid_features.data_ptr(),g.reid_features._version,len(g),meta.get('hits'),
                   tuple((v,m['count'],m['sum'].data_ptr(),m['sum']._version) for v,m in sorted(meta['views'].items())))
        cached=self._history_cache.get(ref)
        if cached is None or cached[0]!=signature or cached[3]() is not g.reid_features:
            # The Gallery mean uses the exact original Tensor.mean operator.
            # No incremental sum/reordered floating reduction is introduced.
            norms=[F.normalize(visual[-1],dim=-1),F.normalize(visual.mean(0),dim=-1)]
            cams={v:F.normalize(m['sum']/m['count'],dim=-1) for v,m in meta['views'].items()}
            cached=(signature,norms,cams,weakref.ref(g.reid_features));self._history_cache[ref]=cached;self.cache_misses+=1
        else:self.cache_hits+=1
        return [cached[1][0],cached[1][1],cached[2].get(view),cached[2].get(1-view)]
    def normalize_history(self,value):return value

    # Only the inherited committed metadata is serialized. Derived caches are
    # invalidated on restore, append, and Tensor in-place version changes.
