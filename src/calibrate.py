"""Synthetic camera-IMU time offset and extrinsic rotation calibration.

The camera angular velocity is obtained from visual relative rotations. The IMU
angular velocity is interpolated at t - dt; the residual
    R_bc^T * omega_imu(t - dt) - omega_cam(t)
is minimized over R_bc and dt. Gyro bias is assumed zero, not estimated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from se3 import so3_exp, so3_log


def load(path: str) -> dict:
    return dict(np.load(path))


def interp_imu(t_query: np.ndarray, t_imu: np.ndarray, omega: np.ndarray) -> np.ndarray:
    out = np.empty((len(t_query), 3))
    for i, t in enumerate(t_query):
        out[i, 0] = np.interp(t, t_imu, omega[:, 0])
        out[i, 1] = np.interp(t, t_imu, omega[:, 1])
        out[i, 2] = np.interp(t, t_imu, omega[:, 2])
    return out


def camera_omega(T_meas: np.ndarray, dt: float) -> np.ndarray:
    return np.stack([so3_log(T[:3, :3]) / dt for T in T_meas])


def calibrate(data: dict) -> dict:
    t_kf = data["t_kf"]
    t_imu = data["t_imu"]
    omega_meas = data["omega_meas"]
    T_meas = data["T_meas"]
    kf_dt = float(np.median(np.diff(t_kf)))
    omega_cam = camera_omega(T_meas, kf_dt)
    t_cam = t_kf[:-1] + 0.5 * kf_dt  # average angular velocity is centered at the interval midpoint

    def residual(x: np.ndarray) -> np.ndarray:
        rvec = x[:3]
        dt = x[3]
        R_bc = so3_exp(rvec)
        omega_imu = interp_imu(t_cam - dt, t_imu, omega_meas)  # IMU timestamp t maps to motion t + dt_true; subtract the estimated offset
        pred = (R_bc.T @ omega_imu.T).T
        return (pred - omega_cam).ravel()

    x0 = np.zeros(4)
    lower = np.array([-0.6, -0.6, -0.6, -0.05])
    upper = np.array([0.6, 0.6, 0.6, 0.05])
    sol = least_squares(residual, x0, bounds=(lower, upper), xtol=1e-12, ftol=1e-12, max_nfev=500)

    R_est = so3_exp(sol.x[:3])
    R_true = data["T_bc_true"][:3, :3]
    rot_err = np.linalg.norm(so3_log(R_est.T @ R_true))
    out = {
        "rvec_bc_est": sol.x[:3].tolist(),
        "dt_est": float(sol.x[3]),
        "bias_g_est": [0.0, 0.0, 0.0],
        "rot_err_rad": float(rot_err),
        "rot_err_deg": float(np.rad2deg(rot_err)),
        "dt_true": float(data["dt_true"]),
        "bias_g_true": data["bias_g_true"].tolist(),
        "cost": float(sol.cost),
        "status": int(sol.status),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sim.npz")
    ap.add_argument("--out", default="data/calib.json")
    args = ap.parse_args()
    data = load(args.data)
    out = calibrate(data)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
