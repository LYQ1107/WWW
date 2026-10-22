import numpy as np

from worldmot.models.panoptic_instances import WorldInstance
from worldmot.tracking.world_memory import WorldIdentityMemory


def make_instance(x, feature=(1., 0.)):
    return WorldInstance(1, np.asarray(feature, dtype=np.float32), np.array([x, 0., 5.]), np.zeros((8, 3)), np.eye(3), 1., (1,), {})


def test_world_id_persists_and_lifecycle_is_causal():
    memory = WorldIdentityMemory(max_missed=2, min_hits=1)
    a = memory.step([make_instance(0.)], 1)
    gid = a[0].global_id
    b = memory.step([make_instance(0.1)], 2)
    assert b[0].global_id == gid
    c = memory.step([], 3, geometry_aligned=False)
    assert c[0].global_id == gid
    memory.step([], 4, geometry_aligned=False)
    memory.step([], 5, geometry_aligned=False)
    assert memory.snapshot() == []


def test_scene_reset_restarts_ids():
    memory = WorldIdentityMemory(min_hits=1)
    assert memory.step([make_instance(0.)], 1)[0].global_id == 1
    memory.reset()
    assert memory.step([make_instance(0.)], 1)[0].global_id == 1
