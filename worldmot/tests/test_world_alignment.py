import numpy as np

from worldmot.geometry.world_alignment import Sim3, estimate_sim3


def test_sim3_ransac_recovers_known_transform_with_outliers():
    rng = np.random.default_rng(4)
    src = rng.normal(size=(80, 3))
    theta = 0.31
    r = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1.]])
    truth = Sim3(1.7, r, np.array([0.4, -0.3, 1.2]))
    dst = truth.apply(src)
    dst[::9] += rng.normal(scale=1.5, size=dst[::9].shape)
    out = estimate_sim3(src, dst, threshold=1e-5, min_inliers=60, iterations=512, seed=10)
    assert out.success
    assert out.inliers >= 70
    assert abs(out.transform.scale - truth.scale) < 1e-5
    assert np.max(np.abs(out.transform.rotation - truth.rotation)) < 1e-5
    assert np.max(np.abs(out.transform.translation - truth.translation)) < 1e-5


def test_alignment_fails_without_enough_points():
    out = estimate_sim3(np.zeros((2, 3)), np.ones((2, 3)), min_inliers=3)
    assert not out.success
    assert out.reason == "too_few_correspondences"
