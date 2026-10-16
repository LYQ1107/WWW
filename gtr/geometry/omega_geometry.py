"""Same-call camera geometry; reuse the existing F and residual definitions."""
import numpy as np
from .geometry_alignment import fundamental_from_cameras, symmetric_epipolar_distance


def original_fundamental(extrinsics, intrinsics, mappings):
    E, K = np.asarray(extrinsics, dtype=float), np.asarray(intrinsics, dtype=float)
    if E.shape != (2, 3, 4) or K.shape != (2, 3, 3) or len(mappings) != 2:
        raise ValueError('two cameras in a single joint inference are required')
    if not np.isfinite(E).all() or not np.isfinite(K).all():
        raise ValueError('nonfinite camera')
    for r, k in zip(E[:, :, :3], K):
        if np.max(np.abs(r @ r.T-np.eye(3))) > 1e-3 or abs(np.linalg.det(r)-1) > 1e-3:
            raise ValueError('rotation is not SO(3)')
        if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1], atol=1e-6):
            raise ValueError('invalid pinhole intrinsics')
    relative_rotation = E[1, :, :3] @ E[0, :, :3].T
    relative_translation = E[1, :, 3]-relative_rotation @ E[0, :, 3]
    baseline = float(np.linalg.norm(relative_translation))
    if baseline <= 1e-9:
        raise ValueError('numerically degenerate translation; no metric-scale claim')
    network = fundamental_from_cameras(E[0], E[1], K[0], K[1])
    original = mappings[1].A.T @ network @ mappings[0].A
    norm = np.linalg.norm(original)
    if not np.isfinite(original).all() or norm <= 1e-12:
        raise ValueError('degenerate original-coordinate F')
    return dict(F_original=original/norm, F_network=network,
                R_BA=relative_rotation, t_BA=relative_translation,
                baseline_in_model_gauge=baseline, reliability='NOT_MEASURED')


def fundamental_equivalence(first, second):
    """F scale and sign are arbitrary; caller explicitly transposes on a swap."""
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    a, b = first/np.linalg.norm(first), second/np.linalg.norm(second)
    return float(min(np.linalg.norm(a-b), np.linalg.norm(a+b)))


def background_validity(F, first, second, original_sizes):
    """Held-out, background-only matches. No fitting or GT here.

    Coverage uses convex-hull area over the *complete original image* in each
    camera, matching the old1% definition. A crop does not shrink its denominator.
    """
    import cv2
    first = np.asarray(first, dtype=float).reshape(-1, 2)
    second = np.asarray(second, dtype=float).reshape(-1, 2)
    if len(first) != len(second):
        raise ValueError('correspondence count mismatch')
    result = dict(valid=False, validation_matches=len(first), median_residual='NOT_MEASURED',
                  p90_residual='NOT_MEASURED', inlier_ratio='NOT_MEASURED', coverage=0.,
                  median_diagonal_normalized='NOT_MEASURED', reasons=[])
    if F is None or not np.isfinite(F).all():
        result['reasons'].append('INVALID_F')
        return result
    residual = symmetric_epipolar_distance(F, first, second)
    if not len(residual) or not np.isfinite(residual).all():
        result['reasons'].append('NO_FINITE_BACKGROUND_RESIDUALS')
        return result
    areas = []
    for points, (width, height) in zip((first, second), original_sizes):
        areas.append(float(cv2.contourArea(cv2.convexHull(points.astype(np.float32))))/(width*height) if len(points) >= 3 else 0.)
    coverage = min(areas)
    median, p90, ratio = float(np.median(residual)), float(np.quantile(residual, .9)), float(np.mean(residual <= 12.))
    diagonal = float(np.sqrt(np.mean([w*w+h*h for w, h in original_sizes])))
    result.update(median_residual=median, p90_residual=p90, inlier_ratio=ratio, coverage=coverage,
                  median_diagonal_normalized=median/diagonal)
    for condition, reason in [(len(residual) < 20, 'FEWER_THAN_20_HELDOUT_MATCHES'),
                              (ratio < .25, 'INLIER_RATIO_BELOW_0.25'),
                              (median > 12., 'MEDIAN_ABOVE_12_ORIGINAL_PIXELS'),
                              (coverage < .01, 'COVERAGE_BELOW_0.01_ORIGINAL_AREA')]:
        if condition:
            result['reasons'].append(reason)
    result['valid'] = not result['reasons']
    return result
