"""Synthetic benchmark sequences and degradation-scenario injection.

The generator reuses the smooth analytic trajectory from ``simulate.py`` and
produces the observation stream an online calibrator consumes: visual relative
rotations / poses (camera frame), gyro samples (body frame, IMU clock shifted by
a known offset), and the ground-truth body trajectory for ATE/RPE.

Degradation scenarios are deterministic functions of the clean sequence:
  * vibration         - sinusoidal gyro angular velocity and visual translation
  * thermal_drift     - linearly ramping gyro bias over the sequence
  * time_offset_step  - a step on the true clock offset partway through; an
    interval-level truth profile is stored so the oracle chain stays correct
  * frame_drop        - random removal of IMU samples and vision keyframes
"""
from __future__ import annotations

import numpy as np

from se3 import pose_from_rvec_t, pose_inv, pose_mul, so3_exp
from simulate import body_omega, position, rotation

SCENARIO_DEFAULTS: dict[str, dict] = {
    "clean": {},
    "vibration": {"ang_deg": 0.3, "freq_hz": 10.0, "trans_mm": 3.0,
                  "start_s": 10.0, "duration_s": 5.0},
    "thermal_drift": {"drift_deg": 0.3},
    "time_offset_step": {"step_ms": 5.0, "step_at_s": 15.0},
    "frame_drop": {"imu_frac": 0.2, "vis_frac": 0.08},
}

BASE_DEFAULTS: dict = {
    "duration": 30.0,
    "keyframe_dt": 0.1,
    "imu_dt": 0.005,
    "dt_true": 0.012,
    "rvec_bc_true": (0.03, -0.02, 0.05),
    "t_bc_true": (0.05, 0.02, 0.01),
    "b_g_true": (0.0, 0.0, 0.0),
    "rot_std": 0.002,
    "trans_std": 0.001,
    "gyro_noise": 0.0005,
}


def generate_sequence(
    duration: float = 30.0,
    keyframe_dt: float = 0.1,
    imu_dt: float = 0.005,
    dt_true: float = 0.012,
    dt_step_ms: float = 0.0,
    dt_step_at: float | None = None,
    rvec_bc_true=(0.03, -0.02, 0.05),
    t_bc_true=(0.05, 0.02, 0.01),
    b_g_true=(0.0, 0.0, 0.0),
    rot_std: float = 0.002,
    trans_std: float = 0.001,
    gyro_noise: float = 0.0005,
    seed: int = 0,
) -> dict:
    """Generate one synthetic sequence with a known (possibly stepped) offset.

    The returned ``dt_true_intervals`` gives the offset used to sample the gyro
    for each keyframe interval.  ``dt_true`` is the initial offset and
    ``dt_true_final`` the offset after any step; the latter is the correct
    steady-state reference for the stepped scenario.
    """
    rng = np.random.default_rng(seed)
    t_kf = np.arange(0.0, duration, keyframe_dt)
    t_imu = np.arange(0.0, duration - 0.5 * imu_dt, imu_dt)

    def dt_profile(t: float) -> float:
        if dt_step_ms and dt_step_at is not None and t >= dt_step_at:
            return dt_true + dt_step_ms * 1e-3
        return dt_true

    T_bc_true = pose_from_rvec_t(np.asarray(rvec_bc_true), np.asarray(t_bc_true))
    T_wb_gt = np.stack([pose_from_rvec_t(np.zeros(3), position(t)) for t in t_kf])
    for k, t in enumerate(t_kf):
        T_wb_gt[k][:3, :3] = rotation(t)
    T_wc_gt = np.stack([pose_mul(T, T_bc_true) for T in T_wb_gt])

    T_meas = []
    for k in range(len(t_kf) - 1):
        rel = pose_mul(pose_inv(T_wc_gt[k]), T_wc_gt[k + 1])
        rel[:3, :3] = rel[:3, :3] @ so3_exp(rng.normal(0.0, rot_std, 3))
        rel[:3, 3] += rng.normal(0.0, trans_std, 3)
        T_meas.append(rel)
    T_meas = np.stack(T_meas)

    b_g = np.asarray(b_g_true, dtype=np.float64)
    omega_meas = np.stack([body_omega(t + dt_profile(t)) + b_g
                           + rng.normal(0.0, gyro_noise, 3) for t in t_imu])

    dt_intervals = np.array([dt_profile(0.5 * (t_kf[k] + t_kf[k + 1]))
                             for k in range(len(t_kf) - 1)])
    return {
        "t_kf": t_kf,
        "t_imu": t_imu,
        "T_meas": T_meas,
        "R_vis": T_meas[:, :3, :3],
        "omega_meas": omega_meas,
        "T_wb_gt": T_wb_gt,
        "R_bc_true": T_bc_true[:3, :3],
        "t_bc_true": T_bc_true[:3, 3],
        "dt_true": dt_true,
        "dt_true_final": float(dt_profile(duration)),
        "dt_true_intervals": dt_intervals,
        "b_g_true": b_g.copy(),
        "scenario": "clean",
    }


def apply_vibration(
    seq: dict,
    ang_deg: float = 0.3,
    freq_hz: float = 10.0,
    trans_mm: float = 3.0,
    start_s: float = 10.0,
    duration_s: float = 5.0,
    seed: int = 0,
) -> dict:
    """Burst of sinusoidal gyro vibration and visual translation jitter.

    The burst is bounded in time so robust gating can reject the corrupted
    intervals; ``duration_s <= 0`` applies it to the whole sequence.
    """
    out = {k: v for k, v in seq.items()}
    rng = np.random.default_rng(seed + 1000)
    phase = rng.uniform(0.0, 2.0 * np.pi, 3)
    # Angular-velocity amplitude equivalent to ang_deg of angular oscillation.
    amp = np.deg2rad(ang_deg) * 2.0 * np.pi * freq_hz
    t_imu = out["t_imu"]
    if duration_s > 0:
        window = (t_imu >= start_s) & (t_imu <= start_s + duration_s)
    else:
        window = np.ones(len(t_imu), dtype=bool)
    omega = out["omega_meas"].copy()
    for j in range(3):
        omega[window, j] += amp * np.sin(2.0 * np.pi * freq_hz * t_imu[window] + phase[j])
    out["omega_meas"] = omega
    T_meas = out["T_meas"].copy()
    t_kf = out["t_kf"]
    for k in range(len(T_meas)):
        active = duration_s <= 0 or start_s <= t_kf[k] <= start_s + duration_s
        if active:
            T_meas[k][:3, 3] += rng.normal(0.0, trans_mm * 1e-3, 3)
    out["T_meas"] = T_meas
    out["R_vis"] = T_meas[:, :3, :3]
    out["scenario"] = (f"vibration_{ang_deg}deg_{freq_hz}hz_{trans_mm}mm"
                       f"_{start_s}-{start_s + duration_s}s")
    return out


def apply_thermal_drift(
    seq: dict, drift_deg: float = 0.3, duration_s: float | None = None
) -> dict:
    """Linearly ramping gyro bias over the sequence (not tracked by the model)."""
    out = {k: v for k, v in seq.items()}
    duration_s = float(out["t_kf"][-1]) if duration_s is None else duration_s
    drift_rate = np.deg2rad(drift_deg) / duration_s  # rad/s per second
    axis = np.array([1.0, 0.5, -0.7])
    axis = axis / np.linalg.norm(axis)
    out["omega_meas"] = out["omega_meas"].copy() + out["t_imu"][:, None] * (drift_rate * axis)[None, :]
    out["drift_rate_rad_s"] = drift_rate * axis
    out["scenario"] = f"thermal_drift_{drift_deg}deg"
    return out


def apply_frame_drop(
    seq: dict, imu_frac: float = 0.2, vis_frac: float = 0.08, seed: int = 0
) -> dict:
    """Randomly drop a fraction of IMU samples and vision keyframes.

    Dropped keyframes are removed from the timeline and their two visual
    relative poses are composed into one longer-baseline measurement, so the
    surviving keyframes remain contiguous and every visual interval still has
    a matching gyro integration interval.
    """
    out = {k: v for k, v in seq.items()}
    rng = np.random.default_rng(seed + 2000)
    if imu_frac > 0:
        keep = rng.random(len(out["t_imu"])) > imu_frac
        keep[0] = keep[-1] = True
        out["t_imu"] = out["t_imu"][keep]
        out["omega_meas"] = out["omega_meas"][keep]
    if vis_frac > 0:
        n = len(out["t_kf"])
        keep_kf = np.ones(n, dtype=bool)
        count = min(int(round(vis_frac * n)), max(n - 2, 0))
        if count:
            drop_idx = rng.choice(np.arange(1, n - 1), size=count, replace=False)
            keep_kf[drop_idx] = False
        kept = np.flatnonzero(keep_kf)
        T_meas = []
        dt_intervals = []
        for a, b in zip(kept[:-1], kept[1:]):
            rel = np.eye(4)
            for j in range(a, b):
                rel = rel @ out["T_meas"][j]
            T_meas.append(rel)
            dt_intervals.append(float(np.mean(out["dt_true_intervals"][a:b])))
        out["t_kf"] = out["t_kf"][kept]
        out["T_wb_gt"] = out["T_wb_gt"][kept]
        out["T_meas"] = np.stack(T_meas) if T_meas else np.empty((0, 4, 4))
        out["R_vis"] = out["T_meas"][:, :3, :3]
        out["dt_true_intervals"] = np.asarray(dt_intervals)
    out["scenario"] = f"frame_drop_imu{imu_frac}_vis{vis_frac}"
    return out


SCENARIOS = ("clean", "vibration", "thermal_drift", "time_offset_step", "frame_drop")


def make_sequence(
    scenario: str,
    seed: int = 0,
    base: dict | None = None,
    params: dict | None = None,
) -> dict:
    """Build a sequence for a named scenario.

    ``base`` overrides the generation settings (duration, noise, true
    calibration, ...); ``params`` overrides the scenario parameters.  Unknown
    keys are rejected so a typo in the YAML config cannot silently change the
    experiment.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")
    gen = {**BASE_DEFAULTS, **(base or {})}
    scenario_params = {**SCENARIO_DEFAULTS[scenario], **(params or {})}
    unknown = set(scenario_params) - set(SCENARIO_DEFAULTS[scenario])
    if unknown:
        raise ValueError(f"unknown {scenario} parameters: {sorted(unknown)}")

    if scenario == "time_offset_step":
        seq = generate_sequence(
            **gen,
            dt_step_ms=float(scenario_params["step_ms"]),
            dt_step_at=float(scenario_params["step_at_s"]),
            seed=seed,
        )
        seq["scenario"] = (f"time_offset_step_{scenario_params['step_ms']}ms"
                           f"_at{scenario_params['step_at_s']}s")
        return seq

    seq = generate_sequence(**gen, seed=seed)
    if scenario == "clean":
        return seq
    if scenario == "vibration":
        return apply_vibration(seq, seed=seed, **scenario_params)
    if scenario == "thermal_drift":
        return apply_thermal_drift(seq, **scenario_params)
    return apply_frame_drop(seq, seed=seed, **scenario_params)
