"""Visual + gyro factor graph for body poses.

Factors:
  * visual relative pose between keyframes (6 residuals each)
  * gyro-integrated relative rotation between keyframes (3 residuals each)

The body-from-camera rotation comes from calibrate.py; translation is fixed at
zero. This prototype does not constrain/estimate extrinsic translation jointly
with body motion using accelerometer or other positional observations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from se3 import (
    pose_from_rvec_t,
    pose_inv,
    pose_mul,
    pose_to_rvec_t,
    se3_log,
    so3_exp,
    so3_log,
)


def chain(T_meas: np.ndarray, T0_body: np.ndarray, T_bc: np.ndarray) -> np.ndarray:
    T_wc = pose_mul(T0_body, T_bc)
    poses = [pose_mul(T_wc, pose_inv(T_bc))]
    for T_rel in T_meas:
        T_wc = pose_mul(T_wc, T_rel)
        poses.append(pose_mul(T_wc, pose_inv(T_bc)))
    return np.stack(poses)


def interp_omega(t: float, t_imu: np.ndarray, omega: np.ndarray) -> np.ndarray:
    return np.array([np.interp(t, t_imu, omega[:, i]) for i in range(3)])


def preintegrate_gyro(
    t0: float, t1: float, t_imu: np.ndarray, omega: np.ndarray, dt_offset: float
) -> np.ndarray:
    """Body-frame rotation over [t0, t1] using the IMU samples shifted by dt_offset."""
    a, b = t0 - dt_offset, t1 - dt_offset
    inside = t_imu[(t_imu >= a) & (t_imu <= b)]
    ts = np.concatenate([[a], inside, [b]])
    dR = np.eye(3)
    for i in range(len(ts) - 1):
        dt = ts[i + 1] - ts[i]
        w0 = interp_omega(ts[i], t_imu, omega)
        w1 = interp_omega(ts[i + 1], t_imu, omega)
        dR = dR @ so3_exp(0.5 * (w0 + w1) * dt)
    return dR


def optimize(
    T_meas: np.ndarray,
    T0_body: np.ndarray,
    R_bc: np.ndarray,
    dR_gyro: np.ndarray,
    x0: np.ndarray | None = None,
    t_bc: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    n = len(T_meas) + 1
    T_bc = np.eye(4)
    T_bc[:3, :3] = R_bc
    if t_bc is not None:
        # Fixed lever arm from a prior; the calibration itself does not estimate it.
        T_bc[:3, 3] = np.asarray(t_bc, dtype=np.float64).reshape(3)
    if x0 is None:
        init = chain(T_meas, T0_body, T_bc)
        x0 = []
        for T in init[1:]:
            rvec, t = pose_to_rvec_t(T)
            x0.extend(rvec)
            x0.extend(t)
        x0 = np.asarray(x0)

    def unpack(x: np.ndarray) -> list[np.ndarray]:
        poses = [T0_body]
        for k in range(1, n):
            rvec = x[(k - 1) * 6 : (k - 1) * 6 + 3]
            t = x[(k - 1) * 6 + 3 : k * 6]
            poses.append(pose_from_rvec_t(rvec, t))
        return poses

    def residual(x: np.ndarray) -> np.ndarray:
        poses = unpack(x)
        res = []
        for k in range(n - 1):
            T_wc_k = pose_mul(poses[k], T_bc)
            T_wc_k1 = pose_mul(poses[k + 1], T_bc)
            pred = pose_mul(pose_inv(T_wc_k), T_wc_k1)
            res.append(se3_log(pose_mul(pose_inv(T_meas[k]), pred)))
            R_rel = poses[k][:3, :3].T @ poses[k + 1][:3, :3]
            res.append(so3_log(dR_gyro[k].T @ R_rel))
        return np.concatenate(res)

    sol = least_squares(residual, x0, method="trf", x_scale="jac", max_nfev=30, verbose=0)
    info = {
        "cost": float(sol.cost),
        "optimality": float(sol.optimality),
        "nfev": int(sol.nfev),
        "status": int(sol.status),
    }
    return np.stack(unpack(sol.x)), T_bc, info


def umeyama_alignment(est: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu_e, mu_g = est.mean(axis=0), gt.mean(axis=0)
    E, G = est - mu_e, gt - mu_g
    U, _, Vt = np.linalg.svd(E.T @ G)
    D = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        D[-1, -1] = -1.0
    R = Vt.T @ D @ U.T
    return R, mu_g - R @ mu_e


def ate_rmse(est_poses: np.ndarray, gt_poses: np.ndarray) -> float:
    est, gt = est_poses[:, :3, 3], gt_poses[:, :3, 3]
    R, t = umeyama_alignment(est, gt)
    aligned = (R @ est.T).T + t
    return float(np.sqrt(np.mean(np.sum((aligned - gt) ** 2, axis=1))))


def rpe_rmse(est_poses: np.ndarray, gt_poses: np.ndarray, delta: int = 1) -> tuple[float, float]:
    trans, rot = [], []
    for k in range(len(est_poses) - delta):
        e_rel = pose_mul(pose_inv(est_poses[k]), est_poses[k + delta])
        g_rel = pose_mul(pose_inv(gt_poses[k]), gt_poses[k + delta])
        err = pose_mul(pose_inv(g_rel), e_rel)
        trans.append(np.linalg.norm(err[:3, 3]))
        rot.append(np.linalg.norm(se3_log(err)[3:]))
    return float(np.sqrt(np.mean(np.square(trans)))), float(np.sqrt(np.mean(np.square(rot))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/sim.npz")
    ap.add_argument("--calib", default="data/calib.json")
    ap.add_argument("--out", default="results/metrics.json")
    args = ap.parse_args()

    data = dict(np.load(args.data))
    calib = json.loads(Path(args.calib).read_text(encoding="utf-8"))
    T_meas, T_wb_gt, T_bc_true = data["T_meas"], data["T_wb_gt"], data["T_bc_true"]
    t_kf, t_imu, omega_meas = data["t_kf"], data["t_imu"], data["omega_meas"]
    T0 = T_wb_gt[0]
    R_bc_est = so3_exp(np.asarray(calib["rvec_bc_est"]))
    R_bc_true = T_bc_true[:3, :3]
    dt_est = float(calib["dt_est"])

    dR_gyro = np.stack(
        [preintegrate_gyro(t_kf[k], t_kf[k + 1], t_imu, omega_meas, dt_est) for k in range(len(t_kf) - 1)]
    )

    baseline = chain(T_meas, T0, np.eye(4))
    calibrated = chain(T_meas, T0, _T(R_bc_est))
    fused, _, info_fused = optimize(T_meas, T0, R_bc_est, dR_gyro)
    oracle, _, info_oracle = optimize(T_meas, T0, R_bc_true, dR_gyro)

    metrics = {
        "calibration": calib,
        "baseline": {
            "ate_rmse_m": ate_rmse(baseline, T_wb_gt),
            "rpe_trans_rmse_m": rpe_rmse(baseline, T_wb_gt)[0],
            "rpe_rot_rmse_rad": rpe_rmse(baseline, T_wb_gt)[1],
        },
        "calibrated_chain": {
            "ate_rmse_m": ate_rmse(calibrated, T_wb_gt),
            "rpe_trans_rmse_m": rpe_rmse(calibrated, T_wb_gt)[0],
            "rpe_rot_rmse_rad": rpe_rmse(calibrated, T_wb_gt)[1],
        },
        "fused_graph": {
            "ate_rmse_m": ate_rmse(fused, T_wb_gt),
            "rpe_trans_rmse_m": rpe_rmse(fused, T_wb_gt)[0],
            "rpe_rot_rmse_rad": rpe_rmse(fused, T_wb_gt)[1],
            "optimizer": info_fused,
        },
        "oracle_extrinsic": {
            "ate_rmse_m": ate_rmse(oracle, T_wb_gt),
            "rpe_trans_rmse_m": rpe_rmse(oracle, T_wb_gt)[0],
            "rpe_rot_rmse_rad": rpe_rmse(oracle, T_wb_gt)[1],
            "optimizer": info_oracle,
        },
        "limitations": [
            "body-from-camera translation is fixed at zero; this prototype does not estimate it",
            "no accelerometer/velocity factor; this is a rotation + visual pose graph, not a full VIO",
            "synthetic data, not EuRoC; no real camera images/feature tracking",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez(
        out.parent / "poses.npz",
        baseline=baseline,
        calibrated=calibrated,
        fused=fused,
        oracle=oracle,
        gt=T_wb_gt,
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def _T(R: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    return T


if __name__ == "__main__":
    main()
