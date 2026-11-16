"""Timestamp-aware metric SE(3) evaluation; no scale fitting or GT in estimation."""
from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from se3 import pose_inv


def rigid_alignment(estimate: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return R,t such that R @ estimate + t matches truth (column convention)."""
    e, g = estimate - estimate.mean(0), truth - truth.mean(0)
    U, _, Vt = np.linalg.svd(e.T @ g)
    D = np.eye(3)
    D[2, 2] = np.linalg.det(Vt.T @ U.T)
    R = Vt.T @ D @ U.T
    return R, truth.mean(0) - R @ estimate.mean(0)


def interpolate_poses(ns: np.ndarray, poses: np.ndarray, query_ns: np.ndarray) -> np.ndarray:
    if np.any(query_ns < ns[0]) or np.any(query_ns > ns[-1]):
        raise ValueError("Ground-truth extrapolation is not allowed")
    t, q = (ns - ns[0]) * 1e-9, (query_ns - ns[0]) * 1e-9
    out = np.repeat(np.eye(4)[None], len(q), axis=0)
    out[:, :3, :3] = Slerp(t, Rotation.from_matrix(poses[:, :3, :3]))(q).as_matrix()
    out[:, :3, 3] = np.column_stack([np.interp(q, t, poses[:, j, 3]) for j in range(3)])
    return out


def evaluate(ns: np.ndarray, poses: np.ndarray, gt_ns: np.ndarray, gt: np.ndarray,
             rpe_seconds: float = 1.0) -> tuple[dict, np.ndarray, np.ndarray]:
    if len(ns) != len(poses) or len(ns) < 3 or np.any(np.diff(ns) <= 0):
        raise ValueError("Need at least three poses with strictly increasing timestamps")
    mask = (ns >= gt_ns[0]) & (ns <= gt_ns[-1])
    associated_ns, est = ns[mask], poses[mask]
    if len(est) < 3:
        raise ValueError("Fewer than three poses overlap ground truth")
    truth = interpolate_poses(gt_ns, gt, associated_ns)
    R, t = rigid_alignment(est[:, :3, 3], truth[:, :3, 3])
    aligned = est.copy()
    aligned[:, :3, :3] = R @ est[:, :3, :3]
    aligned[:, :3, 3] = (R @ est[:, :3, 3].T).T + t
    position_error = np.linalg.norm(aligned[:, :3, 3] - truth[:, :3, 3], axis=1)
    rotation_error = Rotation.from_matrix(np.swapaxes(truth[:, :3, :3], 1, 2) @ aligned[:, :3, :3]).magnitude()
    target = associated_ns + round(rpe_seconds * 1e9)
    indices = np.searchsorted(associated_ns, target)
    rpe_t, rpe_r = [], []
    tolerance_ns = int(np.median(np.diff(associated_ns)) // 2 + 1)
    for i, j in enumerate(indices):
        # Use nearest timestamp, not an arbitrary number of frames.
        candidates = [k for k in (j - 1, j) if i < k < len(est)]
        if not candidates:
            continue
        j = min(candidates, key=lambda k: abs(int(associated_ns[k]) - int(target[i])))
        if abs(int(associated_ns[j]) - int(target[i])) > tolerance_ns:
            continue
        e_rel = pose_inv(est[i]) @ est[j]
        g_rel = pose_inv(truth[i]) @ truth[j]
        error = pose_inv(g_rel) @ e_rel
        rpe_t.append(np.linalg.norm(error[:3, 3]))
        rpe_r.append(np.linalg.norm(Rotation.from_matrix(error[:3, :3]).as_rotvec()))
    rmse = lambda x: float(np.sqrt(np.mean(np.square(x)))) if len(x) else None
    metrics = {
        "alignment": "SE(3), no scale fitting", "input_poses": len(ns), "associated_poses": len(est),
        "association_fraction": float(mask.mean()), "duration_s": float((associated_ns[-1]-associated_ns[0])*1e-9),
        "gt_path_m": float(np.linalg.norm(np.diff(truth[:, :3, 3], axis=0), axis=1).sum()),
        "ate_rmse_m": rmse(position_error), "orientation_rmse_deg": rmse(np.rad2deg(rotation_error)),
        "rpe_interval_s": rpe_seconds, "rpe_pairs": len(rpe_t),
        "rpe_translation_rmse_m": rmse(rpe_t), "rpe_rotation_rmse_deg": rmse(np.rad2deg(rpe_r)),
    }
    return metrics, aligned, truth


def save_tum(path: Path, ns: np.ndarray, poses: np.ndarray):
    quaternion = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    with path.open("w", encoding="utf-8") as stream:
        stream.write("# timestamp_s tx ty tz qx qy qz qw; T_world_body\n")
        for timestamp, pose, q in zip(ns, poses, quaternion):
            timestamp = int(timestamp)
            stamp = f"{timestamp // 10**9}.{timestamp % 10**9:09d}"
            stream.write(stamp + " " + " ".join(f"{v:.12g}" for v in [*pose[:3, 3], *q]) + "\n")


def load_tum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")]
    ns = np.array([int(Decimal(row[0]) * 10**9) for row in rows], dtype=np.int64)
    values = np.array([list(map(float, row[1:])) for row in rows])
    if values.ndim != 2 or values.shape[1] != 7:
        raise ValueError("TUM trajectory requires timestamp tx ty tz qx qy qz qw")
    poses = np.repeat(np.eye(4)[None], len(rows), axis=0)
    poses[:, :3, 3] = values[:, :3]
    poses[:, :3, :3] = Rotation.from_quat(values[:, 3:]).as_matrix()
    return ns, poses
