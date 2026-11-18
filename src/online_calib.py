"""Online camera-IMU time-offset / extrinsic-rotation calibration.

Sliding-window Gauss-Newton over visual relative rotations and gyro-integrated
relative rotations. The state is the constant calibration vector

    theta = [rvec_bc (3), dt (1)]                 (base)
    theta = [rvec_bc (3), dt (1), b_g (3)]        (with gyro bias)

where ``R_bc = exp(rvec_bc)`` rotates a camera-frame vector into the body frame
(``T_wb = T_wc @ T_bc`` with ``T_bc[:3, :3] = R_bc``) and ``dt`` is the camera
clock offset such that an IMU sample stamped ``t`` corresponds to motion time
``t + dt``.

The extrinsic translation ``t_bc`` is *not* estimated: with relative-rotation
observations only (no accelerometer / no visual translation), it is
unobservable, so it is held fixed at its prior. This mirrors the documented
boundary of the parent project.

Residual (one keyframe interval k -> k+1), in the body frame:

    R_pred  = R_bc @ R_vis @ R_bc.T          (vision predicted body rotation)
    R_gyro  = integrate gyro over [t_k - dt, t_{k+1} - dt] with bias removed
    r_k     = so3_log(R_pred.T @ R_gyro) / sigma_r

Jacobians.  With ``A = R_pred.T @ R_gyro`` and ``r = so3_log(A)`` the exact
first-order derivatives are

    dr/dphi   = Jr^-1(r) @ R_gyro.T @ (R_pred - I) @ Jl(phi)
    dr/ddt    = central difference of the discrete gyro integration
    dr/db_g   = Jr^-1(r) @ vee(R_gyro.T @ dR_gyro/db_g)

with ``a = t_k - dt``, ``b = t_{k+1} - dt`` and the bias derivative propagated
through the discrete exponential product.  ``phi`` is a Euclidean rotation
vector, so the chain rule includes the left Jacobian ``Jl``.  The rotation and
bias derivatives are analytic; the time-offset derivative uses a central
difference of the discrete integrator so the exact trapezoidal boundary terms
do not have to be hand-derived.  The formulas are mirrored line-for-line in
``cpp/calib_core/calib.cpp``; tests compare both implementations, and the
Python tests compare the analytic Jacobian against central differences.

Robustness: Huber kernel on the normalized residual norm followed by a
chi-square hard-rejection gate (3 degrees of freedom).  Optionally,
marginalization of old keyframes folds their (robust-weighted) linearized
factors into a Gaussian prior, making the estimator an information filter over
a constant state.  The benchmark disables marginalization for the degradation
scenarios: a constant-parameter filter cannot follow a true offset step, while
a pure sliding window can (with a larger variance).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from so3 import (
    hat,
    left_jacobian,
    right_jacobian_inverse,
    so3_exp,
    so3_exp_batch,
    so3_log,
    vee,
)
# Default chi-square thresholds (3 degrees of freedom).
CHI2_3 = {0.90: 6.2514, 0.95: 7.8147, 0.99: 11.3449}


def _interp_columns(t_query: np.ndarray, t_imu: np.ndarray, omega: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(t_query, t_imu, omega[:, j]) for j in range(3)])


def gyro_terms(
    t0: float,
    t1: float,
    t_imu: np.ndarray,
    omega: np.ndarray,
    dt_offset: float = 0.0,
    bias: np.ndarray | None = None,
) -> dict:
    """Interpolated gyro segment for the interval ``[t0 - dt, t1 - dt]``.

    Returns the bias-corrected angular rates at the two boundaries, the
    trapezoidal increments and their durations.  The segment is found with a
    binary search so cost depends on the interval length, not the full stream.
    """
    a, b = t0 - dt_offset, t1 - dt_offset
    bias = np.zeros(3) if bias is None else np.asarray(bias, dtype=np.float64)
    if b <= a:
        return {"a": a, "b": b, "rates": np.zeros((0, 3)),
                "increments": np.zeros((0, 3)), "durations": np.zeros(0)}
    i0 = int(np.searchsorted(t_imu, a, side="right"))
    i1 = int(np.searchsorted(t_imu, b, side="left"))
    knots = np.concatenate(([a], t_imu[i0:i1], [b]))
    rates = _interp_columns(knots, t_imu, omega) - bias
    durations = np.diff(knots)
    increments = 0.5 * (rates[:-1] + rates[1:]) * durations[:, None]
    return {"a": a, "b": b, "rates": rates,
            "increments": increments, "durations": durations}


def _product_of_exponentials(increments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(R, [exp(inc_i)])`` for a short sequence of increments."""
    if len(increments) == 0:
        return np.eye(3), np.zeros((0, 3, 3))
    factors = so3_exp_batch(increments)
    R = np.eye(3)
    for factor in factors:
        R = R @ factor
    return R, factors


def integrate_gyro(
    t0: float,
    t1: float,
    t_imu: np.ndarray,
    omega: np.ndarray,
    dt_offset: float = 0.0,
    bias: np.ndarray | None = None,
) -> np.ndarray:
    """Integrate body angular velocity over ``[t0 - dt_offset, t1 - dt_offset]``.

    Linear interpolation of the (bias-corrected) gyro samples plus a
    trapezoidal midpoint rule, mirroring ``factor_graph.preintegrate_gyro``.
    """
    terms = gyro_terms(t0, t1, t_imu, omega, dt_offset, bias)
    return _product_of_exponentials(terms["increments"])[0]


def _unpack(theta: np.ndarray, estimate_bias: bool):
    """Return (R_bc, phi, dt, b_g) from the flat state vector."""
    phi = np.asarray(theta, dtype=np.float64)[:3]
    dt = float(theta[3])
    b_g = np.asarray(theta[4:7], dtype=np.float64) if estimate_bias else np.zeros(3)
    return so3_exp(phi), phi, dt, b_g


def _bias_derivative(factors: np.ndarray, increments: np.ndarray,
                     durations: np.ndarray) -> np.ndarray:
    """Derivative of the exponential product with respect to the gyro bias.

    Returns an array shaped ``(3, 3, 3)``: ``result[j] = dR/db_j``.  The
    Euclidean perturbation of a factor is ``dtheta_i = -duration_i * e_j``; the
    corresponding right perturbation is ``Jr(theta_i) @ dtheta_i`` because
    ``exp(theta + dtheta) = exp(theta) @ exp(Jr(theta) @ dtheta)``.
    """
    dR = np.zeros((3, 3, 3))
    R = np.eye(3)
    unit = np.eye(3)
    for factor, inc, duration in zip(factors, increments, durations):
        for j in range(3):
            dtheta = -duration * unit[j]
            dE = factor @ hat(left_jacobian(-inc) @ dtheta)
            dR[j] = dR[j] @ factor + R @ dE
        R = R @ factor
    return dR


def residual_jacobian(
    theta: np.ndarray,
    observations: list[tuple],
    sigma_r: float,
    estimate_bias: bool,
    fd_step: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Stacked normalized residuals and analytic Jacobian for the window.

    ``observations`` is a list of ``(R_vis, t_k, t_k1, t_imu, omega)``.
    Returns ``(r, J)`` with ``r`` shaped ``(3*M,)`` and ``J`` shaped
    ``(3*M, n)`` where ``n`` is 4 (base) or 7 (with bias).  ``fd_step`` is
    retained for signature compatibility; derivatives are analytic.
    """
    del fd_step  # analytic derivatives do not need a step size
    n = 7 if estimate_bias else 4
    M = len(observations)
    r = np.zeros(3 * M)
    J = np.zeros((3 * M, n))
    if M == 0:
        return r, J

    R_bc, phi, dt, b_g = _unpack(theta, estimate_bias)
    Jl_phi = left_jacobian(phi)
    for k, (R_vis, t_k, t_k1, t_imu, omega) in enumerate(observations):
        s = slice(3 * k, 3 * k + 3)
        R_pred = R_bc @ R_vis @ R_bc.T
        terms = gyro_terms(t_k, t_k1, t_imu, omega, dt, b_g)
        R_gyro, factors = _product_of_exponentials(terms["increments"])
        r_k = so3_log(R_pred.T @ R_gyro)
        r[s] = r_k / sigma_r
        Jr_inv = right_jacobian_inverse(r_k)

        J[s, :3] = Jr_inv @ (R_gyro.T @ (R_pred - np.eye(3)) @ Jl_phi) / sigma_r

        # Central difference of the discrete integration over the offset.  The
        # step is 0.1 ms, far below the IMU sample spacing (5 ms by default),
        # so the sampled knot set stays fixed and the derivative is smooth.
        h = 1e-4
        R_dt_plus = integrate_gyro(t_k, t_k1, t_imu, omega, dt + h, b_g)
        R_dt_minus = integrate_gyro(t_k, t_k1, t_imu, omega, dt - h, b_g)
        J[s, 3] = (so3_log(R_pred.T @ R_dt_plus)
                   - so3_log(R_pred.T @ R_dt_minus)) / (2.0 * h * sigma_r)

        if estimate_bias and len(terms["increments"]):
            dR = _bias_derivative(factors, terms["increments"], terms["durations"])
            for j in range(3):
                J[s, 4 + j] = Jr_inv @ vee(R_gyro.T @ dR[j]) / sigma_r

    return r, J


def numeric_jacobian(
    theta: np.ndarray,
    observations: list[tuple],
    sigma_r: float,
    estimate_bias: bool,
    step: float = 1e-6,
) -> np.ndarray:
    """Central-difference Jacobian used by the tests as an independent oracle."""
    theta = np.asarray(theta, dtype=np.float64)
    n = len(theta)
    J = np.zeros((3 * len(observations), n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = step
        r_plus, _ = residual_jacobian(theta + e, observations, sigma_r, estimate_bias)
        r_minus, _ = residual_jacobian(theta - e, observations, sigma_r, estimate_bias)
        J[:, j] = (r_plus - r_minus) / (2.0 * step)
    return J


def initial_rotation(observations: list[tuple]) -> np.ndarray:
    """Closed-form body-from-camera rotation from angular-velocity alignment.

    For each interval the camera angular velocity is ``log(R_vis)/dt`` and the
    body angular velocity is the mean gyro increment over the same interval
    (offset zero).  Solving the orthogonal Procrustes problem
    ``R @ omega_cam ~= omega_body`` gives a robust initialization even when the
    prior is far from the truth (e.g. the ~180 deg rectified-camera frame of
    TUM VI), where Gauss-Newton from identity would stall in a local minimum.

    Returns ``R_bc``.  Raises ``ValueError`` when there are too few usable
    intervals.
    """
    camera, body = [], []
    for (R_vis, t_k, t_k1, t_imu, omega) in observations:
        duration = t_k1 - t_k
        if duration <= 0:
            continue
        terms = gyro_terms(t_k, t_k1, t_imu, omega, 0.0, None)
        if len(terms["increments"]) == 0:
            continue
        camera.append(so3_log(R_vis) / duration)
        body.append(terms["increments"].sum(axis=0) / duration)
    if len(camera) < 3:
        raise ValueError("at least three usable intervals are needed for initialization")
    A, B = np.asarray(camera), np.asarray(body)
    U, _, Vt = np.linalg.svd(A.T @ B)
    D = np.eye(3)
    if np.linalg.det(Vt.T @ U.T) < 0:
        D[-1, -1] = -1.0
    return Vt.T @ D @ U.T


def robust_weights(r: np.ndarray, huber_k: float, chi2_thresh: float) -> np.ndarray:
    """Per-observation robust weights (shape ``(M,)``).

    ``r`` is the stacked normalized residual (3 components per observation).
    Huber downweights the observation norm; a chi-square test (3 dof) then
    hard-rejects gross outliers.
    """
    M = len(r) // 3
    w = np.ones(M)
    for k in range(M):
        rho = float(np.linalg.norm(r[3 * k : 3 * k + 3]))
        if rho * rho > chi2_thresh:
            w[k] = 0.0
        elif rho > huber_k:
            w[k] = huber_k / rho
    return w


def _gauss_newton_step(theta, observations, sigma_r, estimate_bias,
                       huber_k, chi2_thresh, damping,
                       prior_lambda=None, prior_xi=None):
    r, J = residual_jacobian(theta, observations, sigma_r, estimate_bias)
    w = robust_weights(r, huber_k, chi2_thresh)
    W = np.repeat(w, 3)
    H = (J.T * W) @ J
    g = (J.T * W) @ r
    if prior_lambda is not None:
        H = H + prior_lambda
        g = g + prior_lambda @ theta - prior_xi
    H = H + damping * np.eye(len(theta))
    try:
        delta = np.linalg.solve(H, -g)
    except np.linalg.LinAlgError:
        delta = np.zeros_like(theta)
    cost = float(0.5 * (r * W * r).sum()) if w.sum() > 0 else 0.0
    return delta, cost, w


def batch_calibrate(
    observations: list[tuple],
    theta0: np.ndarray,
    sigma_r: float = 0.01,
    estimate_bias: bool = False,
    max_iters: int = 20,
    damping: float = 1e-6,
    huber_k: float = 1.345,
    chi2_level: float = 0.99,
    tol: float = 1e-10,
) -> dict:
    """Offline Gauss-Newton over the full observation set (no sliding window).

    Used as the warm-start reference and the "offline-calibrated" chain. Returns
    a JSON-serializable summary with the same fields as ``OnlineCalibrator``.
    """
    theta = np.asarray(theta0, dtype=np.float64).reshape(7 if estimate_bias else 4).copy()
    chi2_thresh = CHI2_3[chi2_level]
    nfev = 0
    cost = float("inf")
    w = np.ones(len(observations))
    for _ in range(max_iters):
        delta, cost, w = _gauss_newton_step(
            theta, observations, sigma_r, estimate_bias,
            huber_k, chi2_thresh, damping)
        theta = theta + delta
        nfev += 1
        if np.linalg.norm(delta) < tol:
            break
    return {
        "rvec_bc": theta[:3].tolist(),
        "dt": float(theta[3]),
        "bias_g": theta[4:7].tolist() if estimate_bias else [0.0, 0.0, 0.0],
        "cost": cost,
        "nfev": nfev,
        "n_outliers": int((w == 0).sum()),
        "n_obs": len(observations),
    }


class OnlineCalibrator:
    """Sliding-window Gauss-Newton estimator for ``(R_bc, dt, [b_g])``.

    ``push`` adds one keyframe interval, marginalizes the oldest interval once
    the window is full and re-runs Gauss-Newton.  ``history`` records one entry
    per update, which the benchmark uses for convergence metrics.
    """

    def __init__(
        self,
        window: int = 25,
        estimate_bias: bool = False,
        sigma_r: float = 0.01,
        huber_k: float = 1.345,
        chi2_level: float = 0.99,
        max_iters: int = 8,
        damping: float = 1e-6,
        init: str | np.ndarray = "cold",
        tol: float = 1e-9,
        marginalize: bool = True,
        forget: float = 1.0,
    ):
        if window < 1:
            raise ValueError("window must be positive")
        if chi2_level not in CHI2_3:
            raise ValueError(f"chi2_level must be one of {sorted(CHI2_3)}")
        self.window = window
        self.estimate_bias = estimate_bias
        # With marginalization the estimator is a constant-parameter information
        # filter (best for a truly constant offset).  ``forget`` < 1 exponentially
        # forgets old marginalized information, giving a bounded memory that can
        # follow a slowly changing offset.  Disabling marginalization gives a
        # pure sliding window instead.
        if not 0.0 < forget <= 1.0:
            raise ValueError("forget must be in (0, 1]")
        self.marginalize = marginalize
        self.forget = float(forget)
        self.sigma_r = sigma_r
        self.huber_k = huber_k
        self.chi2_thresh = CHI2_3[chi2_level]
        self.max_iters = max_iters
        self.damping = damping
        self.tol = tol
        self.n = 7 if estimate_bias else 4

        if isinstance(init, str):
            if init != "cold":
                raise ValueError(f"unknown init mode: {init!r}; pass a state vector to warm start")
            theta = np.zeros(self.n)
        else:
            theta = np.asarray(init, dtype=np.float64).reshape(self.n)

        self.theta = theta.copy()
        self._obs: deque[tuple] = deque()
        # Gaussian prior (information matrix / vector) from marginalized keyframes.
        self._prior_lambda = np.zeros((self.n, self.n))
        self._prior_xi = np.zeros(self.n)
        self._n_keyframes = 0
        self.history: list[dict] = []

    def push(self, R_vis, t_k, t_k1, t_imu, omega) -> dict:
        """Add one keyframe interval observation and re-estimate."""
        obs = (np.asarray(R_vis, dtype=np.float64), float(t_k), float(t_k1),
               np.asarray(t_imu, dtype=np.float64), np.asarray(omega, dtype=np.float64))
        self._obs.append(obs)
        self._n_keyframes += 1
        if len(self._obs) > self.window:
            self._marginalize_oldest()
        return self.estimate()

    def _marginalize_oldest(self) -> None:
        """Fold the oldest observation into the Gaussian prior and drop it."""
        obs = self._obs[0]
        r, J = residual_jacobian(self.theta, [obs], self.sigma_r, self.estimate_bias)
        w = float(robust_weights(r, self.huber_k, self.chi2_thresh)[0])
        if self.marginalize:
            if self.forget < 1.0:
                self._prior_lambda *= self.forget
                self._prior_xi *= self.forget
            if w > 0.0:
                self._prior_lambda += w * (J.T @ J)
                self._prior_xi += w * (J.T @ (J @ self.theta - r))
        self._obs.popleft()

    def estimate(self) -> dict:
        """Run Gauss-Newton on the prior + window; return a summary dict."""
        theta = self.theta.copy()
        nfev = 0
        cost = float("inf")
        w = np.ones(len(self._obs))
        for _ in range(self.max_iters):
            delta, cost, w = _gauss_newton_step(
                theta, list(self._obs), self.sigma_r, self.estimate_bias,
                self.huber_k, self.chi2_thresh, self.damping,
                self._prior_lambda, self._prior_xi)
            theta = theta + delta
            nfev += 1
            if np.linalg.norm(delta) < self.tol:
                break
        self.theta = theta.copy()
        b_g = self.theta[4:7] if self.estimate_bias else np.zeros(3)
        info = {
            "keyframe": self._n_keyframes,
            "rvec_bc": self.theta[:3].tolist(),
            "dt": float(self.theta[3]),
            "bias_g": b_g.tolist(),
            "rot_err_deg": 0.0,  # filled by the benchmark when truth is known
            "cost": cost,
            "nfev": nfev,
            "n_outliers": int((w == 0).sum()),
            "window_used": len(self._obs),
        }
        self.history.append(info)
        return info
