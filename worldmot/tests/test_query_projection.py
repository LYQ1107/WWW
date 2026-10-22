import numpy as np

from worldmot.geometry.camera_conditioned_sampling import project_world_query
from worldmot.geometry.camera_model import CameraModel


def test_projection_only_is_not_evaluation_eligible_without_calibrated_world_query():
    camera = CameraModel(np.diag([100.0, 100.0, 1.0]), np.eye(4), (80, 100), (80, 100))
    projection = project_world_query(camera, np.array([0.0, 0.0, 5.0]), 1, 4,
                                     has_current_observation=True,
                                     coordinate_system_valid=False)
    assert projection.visible
    assert not projection.evaluation_eligible
    assert projection.reason == "clip_normalized_query_projection_only"


def test_calibrated_current_observation_can_be_evaluation_eligible():
    camera = CameraModel(np.diag([100.0, 100.0, 1.0]), np.eye(4), (80, 100), (80, 100))
    projection = project_world_query(camera, np.array([0.0, 0.0, 5.0]), 1, 4,
                                     has_current_observation=True,
                                     coordinate_system_valid=True)
    assert projection.evaluation_eligible

