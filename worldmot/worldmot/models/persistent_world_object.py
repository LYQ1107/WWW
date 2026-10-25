"""Canonical V3 contracts; no second, diverging state/observation definition."""
from .world_object_observation import WorldObjectObservation
from ..tracking.persistent_world_object_bank import PersistentWorldObject

WorldObjectState = PersistentWorldObject
PersistentWorldObjectObservation = WorldObjectObservation
