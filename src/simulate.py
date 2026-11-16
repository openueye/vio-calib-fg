"""Synthetic camera-IMU data for the factor-graph calibration prototype.

The simulation is deliberately self-contained: a smooth 3D trajectory, a camera
rigidly mounted on the body, visual relative-pose measurements between
keyframes, and IMU samples whose clock is shifted by a known offset.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from se3 import pose_from_rvec_t, pose_inv, pose_mul, so3_exp, so3_log


def _Rz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _Ry(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _Rx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def position(t: float) -> np.ndarray:
    return np.array(
        [2.0 * np.sin(0.8 * t), 1.2 * np.sin(1.6 * t), 0.6 * np.sin(1.2 * t)]
    )


def rotation(t: float) -> np.ndarray:
    yaw = 0.7 * t + 0.35 * np.sin(1.7 * t)
    pitch = 0.25 * np.sin(2.2 * t)
    roll = 0.18 * np.sin(1.8 * t)
    return _Rz(yaw) @ _Ry(pitch) @ _Rx(roll)


def pose(t: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rotation(t)
    T[:3, 3] = position(t)
    return T


def velocity(t: float, h: float = 1e-5) -> np.ndarray:
    return (position(t + h) - position(t - h)) / (2.0 * h)


def acceleration(t: float, h: float = 1e-4) -> np.ndarray:
    return (position(t + h) - 2.0 * position(t) + position(t - h)) / (h * h)


def body_omega(t: float, h: float = 1e-5) -> np.ndarray:
    R0, R1 = rotation(t - h), rotation(t + h)
    return so3_log(R0.T @ R1) / (2.0 * h)


def body_specific_force(t: float, gravity: np.ndarray) -> np.ndarray:
    R = rotation(t)
    return R.T @ (acceleration(t) - gravity)


def _add_pose_noise(T: np.ndarray, rng: np.random.Generator, rot_std: float, trans_std: float) -> np.ndarray:
    out = T.copy()
    out[:3, :3] = T[:3, :3] @ so3_exp(rng.normal(0.0, rot_std, 3))
    out[:3, 3] += rng.normal(0.0, trans_std, 3)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/sim.npz")
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--keyframe-dt", type=float, default=0.1)
    ap.add_argument("--imu-dt", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    t_kf = np.arange(0.0, args.duration, args.keyframe_dt)
    t_imu = np.arange(0.0, args.duration - 0.02, args.imu_dt)

    # True camera-from-body extrinsic and clock offset.
    T_bc_true = pose_from_rvec_t(np.array([0.03, -0.02, 0.05]), np.array([0.05, 0.02, 0.01]))
    dt_true = 0.012  # IMU clock is ahead of camera clock by 12 ms

    T_wb_gt = np.stack([pose(t) for t in t_kf])
    T_wc_gt = np.stack([pose_mul(T, T_bc_true) for T in T_wb_gt])

    T_meas = []
    for k in range(len(t_kf) - 1):
        rel = pose_mul(pose_inv(T_wc_gt[k]), T_wc_gt[k + 1])
        T_meas.append(_add_pose_noise(rel, rng, rot_std=0.0002, trans_std=0.001))
    T_meas = np.stack(T_meas)

    # IMU: sample the true motion at t + dt_true, then add bias/noise.
    gravity = np.array([0.0, 0.0, -9.81])
    bias_g = np.zeros(3)
    bias_a = np.array([0.05, -0.03, 0.04])
    omega_meas = np.stack(
        [body_omega(t + dt_true) + bias_g + rng.normal(0.0, 0.0002, 3) for t in t_imu]
    )
    acc_meas = np.stack(
        [
            body_specific_force(t + dt_true, gravity) + bias_a + rng.normal(0.0, 0.005, 3)
            for t in t_imu
        ]
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        t_kf=t_kf,
        t_imu=t_imu,
        T_wb_gt=T_wb_gt,
        T_wc_gt=T_wc_gt,
        T_meas=T_meas,
        omega_meas=omega_meas,
        acc_meas=acc_meas,
        T_bc_true=T_bc_true,
        dt_true=dt_true,
        bias_g_true=bias_g,
        bias_a_true=bias_a,
    )
    print(f"saved {out} | keyframes={len(t_kf)} imu={len(t_imu)} dt_true={dt_true}")


if __name__ == "__main__":
    main()
