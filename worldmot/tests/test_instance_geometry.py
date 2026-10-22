import numpy as np

from worldmot.models.panoptic_instances import build_world_instances


def test_same_query_is_one_world_instance_across_cameras():
    pm_a = np.zeros((8, 8, 3), dtype=np.float64)
    pm_b = np.zeros((8, 8, 3), dtype=np.float64)
    pm_a[..., 2] = 5; pm_b[..., 2] = 5
    pm_a[2:6, 2:4, :2] = [1, 0]; pm_b[2:6, 3:5, :2] = [1, 0]
    conf = {1: np.ones((8, 8)) * 2, 2: np.ones((8, 8)) * 2}
    pan = {
        1: {"pan": np.where(np.indices((8, 8))[1] < 4, 7, 0), "segments_info": [{"id": 7, "query_id": 3, "category_id": 0}]},
        2: {"pan": np.where(np.indices((8, 8))[1] >= 3, 7, 0), "segments_info": [{"id": 7, "query_id": 3, "category_id": 0}]},
    }
    out = build_world_instances({1: pm_a, 2: pm_b}, conf, pan, np.eye(4, dtype=np.float32), min_confidence=1.5, min_points=3)
    assert len(out) == 1
    assert out[0].proposal_id == 3
    assert out[0].visible_cameras == (1, 2)
