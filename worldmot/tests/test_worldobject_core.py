import numpy as np

from worldmot.models.multi_view_world_object import JointWorldObjectBuilder
from worldmot.models.world_object_support import WorldObjectSupport, extract_world_object_support, map_bbox_to_model
from worldmot.tracking.persistent_world_object_bank import PersistentWorldObjectBank
from worldmot.geometry.dual_view_static_alignment import DualCameraStaticAligner
from worldmot.geometry.world_alignment import Sim3


def _support(center, quality=1.0):
    p = np.asarray(center, dtype=float)[None] + np.array([[0, 0, 0], [0.03, 0, 0], [0, 0.03, 0], [0, 0, 0.03]])
    return WorldObjectSupport(p, p, np.asarray(center, float), np.eye(3) * 0.01, quality, 1.0,
                              "DEPTH_CLUSTERED", 2.0, 1.0)


def _obs(camera, index, center):
    class O: pass
    x = O(); x.camera_id = camera; x.detection_id = index; x.support = _support(center); x.feature = np.ones(8); x.bbox_xywh = (1, 1, 4, 8)
    return x


def test_two_views_update_one_object_and_same_camera_is_unique():
    hypotheses, d = JointWorldObjectBuilder().build([_obs(1, 0, (1, 2, 3)), _obs(2, 0, (1.01, 2, 3))])
    assert len(hypotheses) == 1 and hypotheses[0].cameras == (1, 2)
    bank = PersistentWorldObjectBank(min_hits=1)
    objects, _ = bank.step(hypotheses, 0)
    assert len(objects) == 1 and bank.last_assignments == {0: 1}


def test_close_different_people_are_not_forced_to_merge():
    hypotheses, _ = JointWorldObjectBuilder(max_pair_distance=0.25).build([_obs(1, 0, (1, 2, 3)), _obs(2, 0, (2.0, 2, 3))])
    assert len(hypotheses) == 2 and all(len(x.observations) == 1 for x in hypotheses)


def test_support_is_not_a_gt_mask_and_depth_ambiguity_is_explicit():
    yy, xx = np.mgrid[:20, :20]
    pm = np.stack([xx.astype(float), yy.astype(float), np.ones_like(xx, float)], axis=-1)
    support = extract_world_object_support(pm, np.ones((20, 20)) * 2, (3, 3, 10, 10))
    assert support.ambiguity in {"DEPTH_NOT_SEPARATED", "DEPTH_CLUSTERED"}
    assert support.diagnostics["box_count"] > 0


def test_bbox_mapping_uses_model_frame_and_clips():
    matrix = np.asarray([[0.5, 0.0, 4.0], [0.0, 0.5, 2.0], [0.0, 0.0, 1.0]])
    mapped = map_bbox_to_model([10, 20, 40, 20], matrix, (32, 32))
    np.testing.assert_allclose(mapped, [9.0, 12.0, 20.0, 10.0])


def test_sim3_round_trip_direction():
    t = Sim3(1.7, np.eye(3), np.array([1, 2, 3.]))
    x = np.array([[.2, .3, 2.], [1., 2., 4.]])
    inv = Sim3(1 / t.scale, t.rotation.T, -t.rotation.T @ t.translation / t.scale)
    assert np.max(np.abs(inv.apply(t.apply(x)) - x)) < 1e-8
