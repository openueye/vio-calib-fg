"""Minimal SE(3) utilities used by the VIO calibration prototype."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def hat(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def so3_exp(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = np.linalg.norm(phi)
    if theta < 1e-10:
        return np.eye(3) + hat(phi)
    return Rotation.from_rotvec(phi).as_matrix()


def so3_log(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(R, dtype=float)).as_rotvec()


def _V(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = np.linalg.norm(phi)
    Phi = hat(phi)
    if theta < 1e-10:
        return np.eye(3) + 0.5 * Phi
    return (
        np.eye(3)
        + (1.0 - np.cos(theta)) / theta**2 * Phi
        + (theta - np.sin(theta)) / theta**3 * Phi @ Phi
    )


def _V_inv(phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float).reshape(3)
    theta = np.linalg.norm(phi)
    Phi = hat(phi)
    if theta < 1e-10:
        return np.eye(3) - 0.5 * Phi
    return (
        np.eye(3)
        - 0.5 * Phi
        + (1.0 / theta**2 - (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta)))
        * Phi
        @ Phi
    )


def se3_exp(xi: np.ndarray) -> np.ndarray:
    xi = np.asarray(xi, dtype=float).reshape(6)
    rho, phi = xi[:3], xi[3:]
    T = np.eye(4)
    T[:3, :3] = so3_exp(phi)
    T[:3, 3] = _V(phi) @ rho
    return T


def se3_log(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    phi = so3_log(T[:3, :3])
    rho = _V_inv(phi) @ T[:3, 3]
    return np.concatenate([rho, phi])


def pose_inv(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def pose_mul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.asarray(A, dtype=float) @ np.asarray(B, dtype=float)


def pose_from_rvec_t(rvec: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = so3_exp(rvec)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def pose_to_rvec_t(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T, dtype=float)
    return so3_log(T[:3, :3]), T[:3, 3].copy()

