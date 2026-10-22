import numpy as np

from worldmot.geometry.world_alignment import Sim3Aligner


def test_alignment_reports_failure_without_static_support():
    result = Sim3Aligner(min_inliers=6).update(np.zeros((2, 3)), np.ones((2, 3)))
    assert not result.success
    assert result.reason == "too_few_correspondences"

