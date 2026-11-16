import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import stereo_baseline
from se3 import pose_inv


def test_pnp_direction_and_metric_translation(monkeypatch):
    rng = np.random.default_rng(5)
    xyz = rng.uniform([-1, -1, 3], [1, 1, 7], size=(100, 3))
    K = np.array([[300., 0, 256], [0, 300, 256], [0, 0, 1]])
    T_current_previous = np.eye(4)
    T_current_previous[:3, :3] = Rotation.from_euler("y", 0.05).as_matrix()
    T_current_previous[:3, 3] = [-0.12, 0.02, 0.01]
    projected, _ = cv2.projectPoints(xyz, np.array([0., 0.05, 0.]), T_current_previous[:3, 3], K, None)
    # Isolate PnP/pose convention from optical flow; independent projection is the oracle.
    monkeypatch.setattr(stereo_baseline, "track", lambda *a: (projected.reshape(-1, 2).astype(np.float32), np.ones(100, bool)))
    result, info = stereo_baseline.motion(None, None, np.zeros((100, 2)), xyz, K)
    np.testing.assert_allclose(result, pose_inv(T_current_previous), atol=1e-6)
    assert info["inliers"] == 100


def test_textureless_stereo_does_not_fabricate_points():
    image = np.zeros((120, 160), np.uint8)
    P = np.column_stack((np.eye(3), np.zeros(3)))
    points, xyz = stereo_baseline.stereo_points(image, image, P, P)
    assert points.shape == (0, 2)
    assert xyz.shape == (0, 3)


def test_insufficient_tracks_reports_failure():
    image = np.zeros((120, 160), np.uint8)
    result, info = stereo_baseline.motion(image, image, np.empty((0, 2)), np.empty((0, 3)), np.eye(3))
    assert result is None
    assert info["inliers"] == 0
