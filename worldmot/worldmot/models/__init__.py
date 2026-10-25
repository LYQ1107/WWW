"""Lazy public exports: geometry/contracts do not initialize legacy Torch heads."""
from importlib import import_module

_EXPORTS={"GeometryBackend":"geometry_interface","GeometryFrame":"geometry_interface",
    "WorldInstance":"panoptic_instances","build_world_instances":"panoptic_instances",
    "PersistentWorldQueryTransformer":"persistent_world_queries","QueryBatchState":"persistent_world_queries",
    "MultiViewObservationEncoder":"world_observation_encoder","WorldObservationToken":"world_observation_encoder"}
__all__=list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value=getattr(import_module("."+_EXPORTS[name],__name__),name)
    globals()[name]=value
    return value
