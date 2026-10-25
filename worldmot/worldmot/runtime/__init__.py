"""Lazy legacy exports keep approval/metadata checks independent of model init."""
from importlib import import_module
__all__=["OnlineWorldMOT","OnlineWorldMOTV2"]


def __getattr__(name):
    modules={"OnlineWorldMOT":"online_worldmot","OnlineWorldMOTV2":"online_worldmot_v2"}
    if name not in modules:
        raise AttributeError(name)
    value=getattr(import_module("."+modules[name],__name__),name)
    globals()[name]=value
    return value
