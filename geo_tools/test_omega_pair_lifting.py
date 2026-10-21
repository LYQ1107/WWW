"""Independent analytical checks of the camera convention used by the probe."""
import numpy as np

from omega_pair_geometry_probe import geometry_scores, project_points, world_points


def test_known_plane_with_translated_second_camera():
    K = np.asarray([[100., 0, 50], [0, 100, 50], [0, 0, 1]])
    E = np.column_stack([np.eye(3), [-1., 0, 0]])
    # Camera B is one world unit to the right, looking along +Z.
    uv, z = project_points([[0., 0, 5], [1., 1, 5]], E, K)
    np.testing.assert_allclose(uv, [[30., 50], [50., 70]])
    np.testing.assert_allclose(z, [5., 5])
    np.testing.assert_allclose(world_points(uv, z, E, K), [[0., 0, 5], [1., 1, 5]])


def test_rotated_camera_lifting():
    # World +X points forward in a camera rotated by 90 degrees about Y.
    E = np.asarray([[0., 0, -1., 0.], [0, 1, 0, 0], [1, 0, 0, 0]])
    K = np.diag([100., 100., 1.])
    np.testing.assert_allclose(world_points([[0., 20.]], [5.], E, K), [[5., 1., 0.]])


def test_common_scale_preserves_projections():
    K = np.asarray([[100., 0, 50], [0, 100, 50], [0, 0, 1]])
    E = np.column_stack([np.eye(3), [-1., 0, 0]])
    point = np.asarray([[2., 1., 5.]])
    expected, _ = project_points(point, E, K)
    E[:, 3] *= 7
    actual, _ = project_points(point * 7, E, K)
    np.testing.assert_allclose(actual, expected)


def test_two_people_on_known_depth_plane():
    K = np.asarray([[100., 0, 50], [0, 100, 50], [0, 0, 1]])
    E0 = np.column_stack([np.eye(3), [0., 0, 0]])
    E1 = np.column_stack([np.eye(3), [-1., 0, 0]])
    # Z=5 and baseline=1 give exactly 20 pixels of horizontal disparity.
    pair = dict(images=[dict(width=100, height=100, detections=[dict(bbox=b) for b in boxes])
                        for boxes in ([[50., 20., 20., 40.], [80., 20., 10., 40.]],
                                      [[30., 20., 20., 40.], [60., 20., 10., 40.]])])
    arrays = dict(depth=np.full((2, 100, 100), 5.), extrinsics=np.stack([E0, E1]), intrinsics=np.stack([K, K]))
    scores = geometry_scores(pair, arrays)
    np.testing.assert_allclose(scores['projected_support'], np.eye(2))
    np.testing.assert_allclose(np.diag(scores['foot_3d']), [0., 0.], atol=1e-7)


def test_out_of_frame_feet_are_invalid():
    K = np.asarray([[100., 0, 50], [0, 100, 50], [0, 0, 1]])
    E = np.column_stack([np.eye(3), [0., 0, 0]])
    pair = dict(images=[dict(width=100, height=100, detections=[dict(bbox=[50., 90., 10., 30.])])] * 2)
    scores = geometry_scores(pair, dict(depth=np.full((2, 100, 100), 5.), extrinsics=np.stack([E, E]), intrinsics=np.stack([K, K])))
    assert scores['foot_3d'][0, 0] <= -1e5
