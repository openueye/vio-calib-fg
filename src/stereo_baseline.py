"""Educational stereo VO: rectified KLT tracks -> triangulation -> PnP RANSAC.

No ground truth, IMU, map optimization, loop closure or online calibration.
"""
from __future__ import annotations

import cv2
import numpy as np

from se3 import pose_inv


def track(a, b, points):
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if not len(points):
        return np.empty((0, 2)), np.zeros(0, dtype=bool)
    options = dict(winSize=(21, 21), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    nxt, status, _ = cv2.calcOpticalFlowPyrLK(a, b, points, None, **options)
    back, reverse, _ = cv2.calcOpticalFlowPyrLK(b, a, nxt, None, **options)
    p, q = points.reshape(-1, 2), nxt.reshape(-1, 2)
    good = (status.ravel() == 1) & (reverse.ravel() == 1)
    good &= np.linalg.norm(back.reshape(-1, 2) - p, axis=1) < 1.0
    good &= (q[:, 0] >= 0) & (q[:, 0] < b.shape[1]) & (q[:, 1] >= 0) & (q[:, 1] < b.shape[0])
    return q, good


def stereo_points(left, right, P0, P1):
    points = cv2.goodFeaturesToTrack(left, maxCorners=900, qualityLevel=0.01, minDistance=9, blockSize=7)
    if points is None:
        return np.empty((0, 2)), np.empty((0, 3))
    points = points.reshape(-1, 2)
    matched, valid = track(left, right, points)
    valid &= np.abs(points[:, 1] - matched[:, 1]) < 1.5
    points, matched = points[valid], matched[valid]
    if not len(points):
        return points, np.empty((0, 3))
    homogeneous = cv2.triangulatePoints(P0, P1, points.T, matched.T)
    with np.errstate(divide="ignore", invalid="ignore"):
        xyz = (homogeneous[:3] / homogeneous[3]).T
    # Both camera depths must be positive; metric range is bounded for indoor use.
    right_depth = (P1 @ np.column_stack((xyz, np.ones(len(xyz)))).T)[2]
    valid = np.isfinite(xyz).all(1) & (xyz[:, 2] > 0.2) & (xyz[:, 2] < 30) & (right_depth > 0)
    return points[valid], xyz[valid]


def motion(previous, current, points, xyz, K):
    current_points, valid = track(previous, current, points)
    if valid.sum() < 8:
        return None, {"tracked": int(valid.sum()), "inliers": 0, "reason": "fewer than 8 tracks"}
    object_points, image_points = xyz[valid], current_points[valid]
    ok, r, t, inliers = cv2.solvePnPRansac(
        object_points, image_points, K, None, iterationsCount=100,
        reprojectionError=2.0, confidence=0.999, flags=cv2.SOLVEPNP_EPNP,
    )
    count = 0 if inliers is None else len(inliers)
    info = {"tracked": int(valid.sum()), "inliers": count}
    if not ok or count < 8:
        return None, {**info, "reason": "PnP failed or fewer than 8 inliers"}
    selected = inliers.ravel()
    r, t = cv2.solvePnPRefineLM(object_points[selected], image_points[selected], K, None, r, t)
    T_current_previous = np.eye(4)
    T_current_previous[:3, :3], _ = cv2.Rodrigues(r)
    T_current_previous[:3, 3] = t.ravel()
    # PnP transforms previous-frame 3D points into current-camera coordinates.
    return pose_inv(T_current_previous), info
