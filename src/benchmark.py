"""Online-calibration benchmark: four chains per sequence and their metrics.

Chains (all produce a fused visual+gyro body trajectory via the batch
pose-graph in ``factor_graph``, differing only in the calibration parameters
fed to the visual/gyro factors):

  * uncalibrated   - R_bc = I, dt = 0, b_g = 0
  * offline        - batch Gauss-Newton over the whole sequence
  * online         - sliding-window Gauss-Newton, cold start (no future data)
  * ground_truth   - the generating extrinsic / offset / bias (oracle)

Metrics per chain: ATE, 1 s RPE (translation + rotation), calibration errors
(rotation error vs true R_bc, dt error vs the final true offset), convergence
keyframe/time for the online chain and failure against explicit tolerances.
"""
from __future__ import annotations

import platform
import sys

import numpy as np

from factor_graph import ate_rmse, optimize, rpe_rmse
from online_calib import (
    OnlineCalibrator,
    batch_calibrate,
    initial_rotation,
    integrate_gyro,
)
from so3 import rotation_error_rad, so3_exp, so3_log


def observations(seq: dict) -> list[tuple]:
    t_kf = seq["t_kf"]
    return [
        (seq["R_vis"][k], t_kf[k], t_kf[k + 1], seq["t_imu"], seq["omega_meas"])
        for k in range(len(seq["T_meas"]))
    ]


def subsample(seq: dict, stride: int) -> dict:
    """Coarsen keyframes for the (comparatively expensive) pose-graph solve."""
    if stride <= 1:
        return seq
    t_kf = seq["t_kf"][::stride]
    n = len(t_kf)
    T_meas = []
    dt_intervals = []
    for k in range(n - 1):
        rel = np.eye(4)
        for j in range(k * stride, min((k + 1) * stride, len(seq["T_meas"]))):
            rel = rel @ seq["T_meas"][j]
        T_meas.append(rel)
        if "dt_true_intervals" in seq:
            dt_intervals.append(float(np.mean(seq["dt_true_intervals"][
                k * stride:min((k + 1) * stride, len(seq["dt_true_intervals"]))])))
    out = dict(seq)
    out["t_kf"] = t_kf
    out["T_meas"] = np.stack(T_meas) if T_meas else np.empty((0, 4, 4))
    out["R_vis"] = out["T_meas"][:, :3, :3]
    out["T_wb_gt"] = seq["T_wb_gt"][::stride]
    if dt_intervals:
        out["dt_true_intervals_sub"] = np.asarray(dt_intervals)
    return out


def fused_trajectory(seq: dict, R_bc: np.ndarray, dt, b_g: np.ndarray):
    """Pose-graph trajectory for one calibration setting.

    ``dt`` may be a scalar (constant estimate) or one value per keyframe
    interval (oracle chain for the stepped-offset scenario).  The first body
    pose is anchored at identity; ATE/RPE alignment removes the global frame,
    so no ground-truth pose enters estimation.
    """
    t_kf = seq["t_kf"]
    T_meas = seq["T_meas"]
    n = len(T_meas)
    dt_array = np.broadcast_to(np.asarray(dt, dtype=np.float64), (n,)) if n else np.zeros(0)
    dR_gyro = np.stack(
        [
            integrate_gyro(t_kf[k], t_kf[k + 1], seq["t_imu"], seq["omega_meas"],
                           float(dt_array[k]), b_g)
            for k in range(n)
        ]
    ) if n else np.zeros((0, 3, 3))
    poses, _, info = optimize(
        T_meas, np.eye(4), np.asarray(R_bc, dtype=np.float64), dR_gyro,
        t_bc=np.asarray(seq.get("t_bc_true", np.zeros(3)), dtype=np.float64))
    return poses, info


def trajectory_metrics(poses: np.ndarray, gt: np.ndarray) -> dict:
    ate = ate_rmse(poses, gt)
    rpe_t, rpe_r = rpe_rmse(poses, gt)
    return {
        "ate_rmse_m": ate,
        "rpe_translation_rmse_m": rpe_t,
        "rpe_rotation_rmse_deg": float(np.rad2deg(rpe_r)),
    }


def _calib_from_info(info: dict, estimate_bias: bool):
    R_bc = so3_exp(np.asarray(info["rvec_bc"]))
    dt = float(info["dt"])
    b_g = np.asarray(info["bias_g"]) if estimate_bias else np.zeros(3)
    return R_bc, dt, b_g


def initial_state(seq: dict, cfg: dict) -> np.ndarray:
    """State prior used by the offline/online chains.

    ``kabsch`` (default) initializes ``R_bc`` from angular-velocity alignment;
    ``zero`` starts from identity.  ``dt`` and bias start at zero.
    """
    n = 7 if cfg["estimate_bias"] else 4
    theta = np.zeros(n)
    mode = cfg.get("init", "kabsch")
    if mode == "kabsch":
        try:
            theta[:3] = so3_log(initial_rotation(observations(seq)))
        except ValueError:
            pass
    elif mode != "zero":
        raise ValueError(f"unknown init mode: {mode!r}")
    return theta


def calibrate_offline(seq: dict, cfg: dict, warm: np.ndarray | None = None) -> dict:
    estimate_bias = bool(cfg["estimate_bias"])
    theta0 = np.asarray(warm, dtype=np.float64) if warm is not None else initial_state(seq, cfg)
    return batch_calibrate(
        observations(seq),
        theta0,
        sigma_r=cfg["sigma_r"],
        estimate_bias=estimate_bias,
        max_iters=cfg["offline_max_iters"],
        damping=cfg["damping"],
        huber_k=cfg["huber_k"],
        chi2_level=cfg["chi2_level"],
    )


def calibrate_online(seq: dict, cfg: dict) -> tuple[OnlineCalibrator, dict]:
    estimate_bias = bool(cfg["estimate_bias"])
    cal = OnlineCalibrator(
        window=cfg["window"],
        estimate_bias=estimate_bias,
        sigma_r=cfg["sigma_r"],
        huber_k=cfg["huber_k"],
        chi2_level=cfg["chi2_level"],
        max_iters=cfg["online_max_iters"],
        damping=cfg["damping"],
        init=initial_state(seq, cfg),
        marginalize=bool(cfg.get("marginalize", True)),
        forget=float(cfg.get("forget", 1.0)),
    )
    obs = observations(seq)
    for o in obs:
        cal.push(*o)
        cal.history[-1]["t_s"] = float(o[2] - seq["t_kf"][0])
    info = cal.history[-1] if cal.history else cal.estimate()
    return cal, info


def convergence(history: list[dict], dt_ref: float, R_ref: np.ndarray,
                dt_tol: float, rot_tol_deg: float) -> tuple[int | None, float | None]:
    """First keyframe after which the estimate stays within tolerance.

    Returns ``(keyframe_index, seconds_from_start)`` or ``(None, None)``.
    """
    if not history:
        return None, None
    ok = [
        abs(entry["dt"] - dt_ref) <= dt_tol
        and rotation_error_rad(so3_exp(np.asarray(entry["rvec_bc"])), R_ref) <= np.deg2rad(rot_tol_deg)
        for entry in history
    ]
    last_bad = max((i for i, value in enumerate(ok) if not value), default=-1)
    if last_bad == len(ok) - 1:
        return None, None
    entry = history[last_bad + 1]
    return int(entry["keyframe"]), entry.get("t_s")


def steady_state(history: list[dict], ref: float, n_tail: int = 30) -> float:
    if not history:
        return float("nan")
    return float(np.mean([abs(e["dt"] - ref) for e in history[-n_tail:]]))


def _chain_result(seq: dict, sub: dict, R_bc: np.ndarray, dt, b_g: np.ndarray,
                  dt_ref: float, success_dt: float, success_rot_deg: float,
                  optimizer: dict | None = None, dt_reported: float | None = None) -> dict:
    poses, info = fused_trajectory(sub, R_bc, dt, b_g)
    dt_value = (float(dt_reported) if dt_reported is not None
                else float(np.mean(np.asarray(dt, dtype=np.float64))) if np.ndim(dt) else float(dt))
    dt_err = dt_value - dt_ref
    rot_err = float(np.rad2deg(rotation_error_rad(R_bc, seq["R_bc_true"])))
    return {
        "calib": {
            "dt": dt_value,
            "dt_err_ms": 1e3 * dt_err,
            "rvec_bc": so3_log(R_bc).tolist(),
            "rot_err_deg": rot_err,
            "bias_g": np.asarray(b_g).tolist(),
        },
        **trajectory_metrics(poses, sub["T_wb_gt"]),
        "success": bool(abs(dt_err) * 1e3 <= success_dt and rot_err <= success_rot_deg),
        "optimizer": optimizer or {"nfev": int(info["nfev"]), "cost": float(info["cost"])},
    }


def run_sequence(seq: dict, cfg: dict, run_id: str, commit: str, config_hash: str,
                 seed: int, scenario: str, sequence: str) -> dict:
    """Run all four chains for one (sequence, scenario, seed) combination."""
    estimate_bias = bool(cfg["estimate_bias"])
    dt_ref = float(seq.get("dt_true_final", seq["dt_true"]))
    sub = subsample(seq, int(cfg["ate_stride"]))
    success_dt = float(cfg["success_dt_ms"])
    success_rot = float(cfg["success_rot_deg"])
    result: dict = {
        "run_id": run_id,
        "commit": commit,
        "config_hash": config_hash,
        "sequence": sequence,
        "scenario": scenario,
        "seed": seed,
        "n_keyframes": int(len(seq["t_kf"])),
        "duration_s": float(seq["t_kf"][-1] - seq["t_kf"][0]),
        "estimate_bias": estimate_bias,
        "dt_true_ms": 1e3 * float(seq["dt_true"]),
        "dt_final_true_ms": 1e3 * dt_ref,
        "chains": {},
    }

    # Oracle chain: true per-interval offset, true rotation and bias.
    dt_gt = sub.get("dt_true_intervals_sub", seq["dt_true_intervals"])
    result["chains"]["ground_truth"] = _chain_result(
        seq, sub, seq["R_bc_true"], dt_gt, seq["b_g_true"],
        dt_ref, success_dt, success_rot, dt_reported=dt_ref)

    # Uncalibrated chain.
    result["chains"]["uncalibrated"] = _chain_result(
        seq, sub, np.eye(3), 0.0, np.zeros(3), dt_ref, success_dt, success_rot)

    # Offline chain.
    off = calibrate_offline(seq, cfg)
    off_R, off_dt, off_b = _calib_from_info(off, estimate_bias)
    result["chains"]["offline"] = _chain_result(
        seq, sub, off_R, off_dt, off_b, dt_ref, success_dt, success_rot,
        {"nfev": off["nfev"], "n_outliers": off["n_outliers"], "cost": off["cost"]})

    # Online chain (cold start; only past observations).
    cal, info = calibrate_online(seq, cfg)
    on_R, on_dt, on_b = _calib_from_info(info, estimate_bias)
    online = _chain_result(seq, sub, on_R, on_dt, on_b, dt_ref, success_dt, success_rot,
                           {"nfev": int(sum(e["nfev"] for e in cal.history))})
    conv_kf, conv_s = convergence(
        cal.history, dt_ref, seq["R_bc_true"],
        float(cfg["convergence_dt_ms"]) * 1e-3,
        float(cfg["convergence_rot_deg"]))
    online.update({
        "convergence_keyframe": conv_kf,
        "convergence_s": conv_s,
        "steady_state_dt_err_ms": 1e3 * steady_state(cal.history, dt_ref, int(cfg["steady_tail"])),
        "n_outliers": int(info["n_outliers"]),
        "window_used": int(info["window_used"]),
    })
    if conv_kf is None:
        online["success"] = False
    result["chains"]["online"] = online
    return result


def aggregate(results: list[dict]) -> list[dict]:
    """Median/mean metrics and failure rate per (scenario, chain)."""
    rows = []
    scenarios = sorted({r["scenario"] for r in results})
    chains = ["uncalibrated", "offline", "online", "ground_truth"]
    for scenario in scenarios:
        subset = [r for r in results if r["scenario"] == scenario]
        for chain in chains:
            entries = [r["chains"][chain] for r in subset if chain in r["chains"]]
            if not entries:
                continue
            def med(key, scale=1.0):
                values = [e[key] for e in entries if e.get(key) is not None]
                return float(np.median(values)) * scale if values else float("nan")

            def calib_med(key):
                values = [e["calib"][key] for e in entries
                          if e.get("calib", {}).get(key) is not None]
                return float(np.median(values)) if values else float("nan")
            conv = [e["convergence_s"] for e in entries if e.get("convergence_s") is not None]
            rows.append({
                "scenario": scenario,
                "chain": chain,
                "runs": len(entries),
                "dt_err_ms": calib_med("dt_err_ms"),
                "rot_err_deg": calib_med("rot_err_deg"),
                "ate_rmse_m": med("ate_rmse_m"),
                "rpe_t_m": med("rpe_translation_rmse_m"),
                "rpe_r_deg": med("rpe_rotation_rmse_deg"),
                "convergence_s": float(np.median(conv)) if conv else None,
                "steady_dt_err_ms": med("steady_state_dt_err_ms") if chain == "online" else None,
                "failure_rate": float(np.mean([not e["success"] for e in entries])),
            })
    return rows


def environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def run_job(job: dict) -> dict:
    """Process-pool entry point: build one sequence and run its four chains."""
    if job.get("real_asset"):
        from real_data import load_sequence
        seq = load_sequence(job["real_asset"])
        sequence_kind = "tumvi_room1_stereo_vo"
    else:
        from synthetic import make_sequence
        seq = make_sequence(job["scenario"], seed=job["seed"],
                            base=job["base"], params=job["params"])
        sequence_kind = "synthetic"
    return run_sequence(
        seq, job["cfg"], job["run_id"], job["commit"], job["config_hash"],
        job["seed"], job["scenario"], sequence_kind)
