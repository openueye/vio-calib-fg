"""Self-contained SO(3) math with explicit formulas.

This module is the single source of truth for the NumPy/SciPy reference and is
mirrored line-for-line in ``cpp/calib_core/so3.h`` so the C++ and Python
implementations agree to better than 1e-6. It deliberately avoids
``scipy.spatial.transform.Rotation`` so the two implementations use identical
trigonometry.
"""
from __future__ import annotations

import numpy as np

_TINY = 1e-12


def hat(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
    )


def vee(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, dtype=np.float64)
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """Rodrigues formula. Maps a 3-vector (rotation vector) to SO(3)."""
    phi = np.asarray(phi, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(phi))
    if theta < _TINY:
        return np.eye(3) + hat(phi)
    axis = phi / theta
    K = hat(axis)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """Inverse of ``so3_exp``. Returns the rotation vector in [-pi, pi]."""
    R = np.asarray(R, dtype=np.float64)
    c = 0.5 * (np.trace(R) - 1.0)
    c = float(np.clip(c, -1.0, 1.0))
    theta = float(np.arccos(c))
    S = (R - R.T) / 2.0
    v = vee(S)
    if theta < _TINY:
        return v
    return (theta / np.sin(theta)) * v


def left_jacobian(phi: np.ndarray) -> np.ndarray:
    """Left Jacobian of SO(3) at ``phi``.

    ``exp(phi + delta) = exp(left_jacobian(phi) @ delta) @ exp(phi)`` to first
    order in ``delta`` (Euclidean rotation-vector perturbation).
    """
    phi = np.asarray(phi, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(phi))
    K = hat(phi)
    if theta < _TINY:
        return np.eye(3) + 0.5 * K
    return (
        np.eye(3)
        + (1.0 - np.cos(theta)) / (theta * theta) * K
        + (theta - np.sin(theta)) / (theta**3) * (K @ K)
    )


def left_jacobian_inverse(phi: np.ndarray) -> np.ndarray:
    """Inverse of the left Jacobian of SO(3), evaluated at ``phi``.

    For R = exp(phi_hat) and a small left perturbation exp(delta_hat) * R,
    ``d(so3_log(R)) = left_jacobian_inverse(phi) @ delta``.
    """
    phi = np.asarray(phi, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(phi))
    K = hat(phi)
    if theta < _TINY:
        return np.eye(3) - 0.5 * K
    a = 1.0 / (theta * theta)
    b = (1.0 + np.cos(theta)) / (2.0 * theta * np.sin(theta))
    return np.eye(3) - 0.5 * K + (a - b) * (K @ K)


def right_jacobian_inverse(phi: np.ndarray) -> np.ndarray:
    """Inverse right Jacobian of SO(3), evaluated at ``phi``.

    For R = exp(phi_hat) and a small right perturbation R * exp(delta_hat),
    ``d(so3_log(R)) = right_jacobian_inverse(phi) @ delta``.  Equivalently
    ``right_jacobian_inverse(phi) = left_jacobian_inverse(-phi)``.
    """
    return left_jacobian_inverse(-np.asarray(phi, dtype=np.float64).reshape(3))


def rotation_error_rad(R1: np.ndarray, R2: np.ndarray) -> float:
    """Magnitude of the relative rotation between two rotation matrices."""
    return float(np.linalg.norm(so3_log(np.asarray(R1).T @ np.asarray(R2))))


def so3_exp_batch(Phi: np.ndarray) -> np.ndarray:
    """Vectorized Rodrigues formula. ``Phi`` is ``(M, 3)`` -> ``(M, 3, 3)``."""
    Phi = np.asarray(Phi, dtype=np.float64)
    M = Phi.shape[0]
    theta = np.linalg.norm(Phi, axis=1)
    eye = np.broadcast_to(np.eye(3), (M, 3, 3)).copy()
    small = theta < _TINY
    if small.any():
        eye[small] = eye[small] + _hat_batch(Phi[small])
    big = ~small
    if big.any():
        axis = Phi[big] / theta[big, None]
        K = _hat_batch(axis)
        s = np.sin(theta[big])
        c = np.cos(theta[big])
        eye[big] = eye[big] + s[:, None, None] * K + (1.0 - c)[:, None, None] * (K @ K)
    return eye


def _hat_batch(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    M = v.shape[0]
    out = np.zeros((M, 3, 3))
    out[:, 0, 1] = -v[:, 2]
    out[:, 0, 2] = v[:, 1]
    out[:, 1, 0] = v[:, 2]
    out[:, 1, 2] = -v[:, 0]
    out[:, 2, 0] = -v[:, 1]
    out[:, 2, 1] = v[:, 0]
    return out
