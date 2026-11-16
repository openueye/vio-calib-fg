import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from factor_graph import ate_rmse
from imu_diagnostics import integrate_rotation
from trajectory_eval import evaluate, interpolate_poses, load_tum, rigid_alignment, save_tum


def trajectory():
    t = np.linspace(0, 5, 51)
    ns = 1520530000000000000 + np.round(t*1e9).astype(np.int64)
    poses = np.repeat(np.eye(4)[None], len(t), axis=0)
    poses[:, :3, 3] = np.column_stack((np.sin(t), np.cos(t), t/4))
    poses[:, :3, :3] = Rotation.from_rotvec(np.column_stack((t/10, t/20, t/5))).as_matrix()
    return ns, poses


def test_known_global_frame_change_is_removed_without_scale():
    ns, truth = trajectory()
    frame = np.eye(4)
    frame[:3, :3] = Rotation.from_euler("xyz", [0.7, -0.3, 1.2]).as_matrix()
    frame[:3, 3] = [2, -3, 5]
    estimate = frame @ truth
    metrics, aligned, _ = evaluate(ns, estimate, ns, truth)
    assert metrics["ate_rmse_m"] < 1e-12
    assert metrics["rpe_rotation_rmse_deg"] < 1e-10
    assert metrics["orientation_rmse_deg"] < 1e-10
    np.testing.assert_allclose(aligned, truth, atol=1e-12)
    # Regression for the older synthetic evaluator's reversed SVD rotation.
    assert ate_rmse(estimate, truth) < 1e-12


def test_scale_error_is_not_hidden():
    ns, truth = trajectory()
    estimate = truth.copy()
    estimate[:, :3, 3] *= 2
    metrics, _, _ = evaluate(ns, estimate, ns, truth)
    assert metrics["ate_rmse_m"] > 0.5


def test_alignment_does_not_reflect():
    rng = np.random.default_rng(4)
    xyz = rng.normal(size=(50, 3))
    reflected = xyz * [-1, 1, 1]
    R, _ = rigid_alignment(xyz, reflected)
    assert np.linalg.det(R) == pytest.approx(1)


def test_interpolation_and_no_extrapolation():
    ns = np.array([1000000000, 3000000000], dtype=np.int64)
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, :3, 3] = [2, 4, 6]
    poses[1, :3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    middle = interpolate_poses(ns, poses, np.array([2000000000]))[0]
    np.testing.assert_allclose(middle[:3, 3], [1, 2, 3])
    np.testing.assert_allclose(middle[:3, :3], Rotation.from_euler("z", 45, degrees=True).as_matrix())
    with pytest.raises(ValueError, match="extrapolation"):
        interpolate_poses(ns, poses, np.array([999999999]))


def test_tum_roundtrip_preserves_nanosecond_timestamps(tmp_path):
    ns, poses = trajectory()
    ns += 1234567
    path = tmp_path / "poses.tum"
    save_tum(path, ns, poses)
    actual_ns, actual_poses = load_tum(path)
    np.testing.assert_array_equal(actual_ns, ns)
    np.testing.assert_allclose(actual_poses, poses, atol=1e-10)


def test_partial_gt_coverage_is_reported():
    ns, poses = trajectory()
    metrics, _, _ = evaluate(ns, poses, ns[10:-10], poses[10:-10])
    assert metrics["associated_poses"] == 31
    assert metrics["association_fraction"] == pytest.approx(31/51)


def test_duplicate_estimate_timestamps_rejected():
    ns, poses = trajectory()
    ns[1] = ns[0]
    with pytest.raises(ValueError, match="increasing"):
        evaluate(ns, poses, ns, poses)


def test_gyro_integration_uses_exact_interval_and_bias():
    ns = 1520530000000000000 + np.arange(201, dtype=np.int64)*5000000
    gyro = np.tile([0.01, -0.02, 0.53], (201, 1))
    R = integrate_rotation(ns, gyro, int(ns[0]+1234567), int(ns[0]+801234567), [0.01, -0.02, 0.03])
    np.testing.assert_allclose(R, Rotation.from_rotvec([0, 0, 0.4]).as_matrix(), atol=1e-12)
    with pytest.raises(ValueError, match="cover"):
        integrate_rotation(ns, gyro, int(ns[0]-1), int(ns[-1]))
