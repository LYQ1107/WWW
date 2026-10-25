"""Lazy exports preserve existing APIs without importing the V2 query runtime."""
from importlib import import_module
_EXPORTS={"TrackStatus":"state","WorldTrackState":"state","WorldIdentityMemory":"world_memory",
          "PersistentQueryRuntime":"persistent_query_runtime"}
__all__=list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value=getattr(import_module("."+_EXPORTS[name],__name__),name)
    globals()[name]=value
    return value
