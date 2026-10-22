import numpy as np

from worldmot.geometry.camera_model import CameraModel
from worldmot.geometry.reprojection import project_support


def test_world_projection_roundtrip():
    K = np.array([[100., 0, 50], [0, 100., 40], [0, 0, 1.]])
    cam = CameraModel(K, np.eye(4), (80, 100), (80, 100))
    points = np.array([[0, 0, 5], [0.5, 0.2, 5], [-0.5, -0.2, 5], [0.2, -0.1, 4.5]])
    result = project_support(points, cam)
    assert result.visible
    assert result.bbox_xywh is not None
    uv, valid = cam.project(points)
    assert valid.all()
    assert np.allclose(uv[0], [50., 40.])


def test_projection_rejects_points_behind_camera_and_outside():
    cam = CameraModel(np.diag([100., 100., 1.]), np.eye(4), (80, 100), (80, 100))
    result = project_support(np.array([[0, 0, -1], [100, 100, 1]]), cam)
    assert not result.visible
