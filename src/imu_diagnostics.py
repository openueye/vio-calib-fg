"""Independent gyro diagnostics, with no feedback into stereo VO."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def integrate_rotation(imu_ns, gyro, start_ns: int, end_ns: int, bias=None):
    if start_ns < imu_ns[0] or end_ns > imu_ns[-1] or start_ns >= end_ns:
        raise ValueError("IMU must cover the complete positive-duration interval")
    # Subtract integer epoch first to retain sub-microsecond timestamp precision.
    times = (imu_ns - start_ns) * 1e-9
    end = (end_ns - start_ns) * 1e-9
    knots = np.r_[0.0, times[(times > 0) & (times < end)], end]
    samples = np.column_stack([np.interp(knots, times, gyro[:, j]) for j in range(3)])
    samples -= np.zeros(3) if bias is None else np.asarray(bias)
    R = np.eye(3)
    for dt, omega in zip(np.diff(knots), (samples[:-1] + samples[1:]) * 0.5):
        R = R @ Rotation.from_rotvec(omega * dt).as_matrix()
    return R


def diagnose(imu_ns, gyro, ns, poses, gt_ns, gt):
    from trajectory_eval import interpolate_poses

    mask = (ns >= max(imu_ns[0], gt_ns[0])) & (ns <= min(imu_ns[-1], gt_ns[-1]))
    ns, poses = ns[mask], poses[mask]
    if len(ns) < 2:
        return {"intervals": 0}, {}
    truth = interpolate_poses(gt_ns, gt, ns)
    visual_errors, gyro_errors, disagreement = [], [], []
    for i in range(len(ns)-1):
        visual = poses[i, :3, :3].T @ poses[i+1, :3, :3]
        actual = truth[i, :3, :3].T @ truth[i+1, :3, :3]
        inertial = integrate_rotation(imu_ns, gyro, int(ns[i]), int(ns[i+1]))
        angle = lambda R: float(np.rad2deg(Rotation.from_matrix(R).magnitude()))
        visual_errors.append(angle(actual.T @ visual))
        gyro_errors.append(angle(actual.T @ inertial))
        disagreement.append(angle(visual.T @ inertial))
    rmse = lambda a: float(np.sqrt(np.mean(np.square(a))))
    return {
        "intervals": len(visual_errors), "gyro_bias": "zero assumed; no GT-derived bias",
        "interval_median_s": float(np.median(np.diff(ns)) * 1e-9),
        "visual_increment_rmse_deg": rmse(visual_errors),
        "gyro_increment_rmse_deg": rmse(gyro_errors),
        "visual_gyro_disagreement_rmse_deg": rmse(disagreement),
        "interpretation": "independent relative-rotation diagnostic; no fused position estimate",
    }, {"time_s": (ns[1:]-ns[0])*1e-9, "visual_deg": visual_errors,
        "gyro_deg": gyro_errors, "disagreement_deg": disagreement}
