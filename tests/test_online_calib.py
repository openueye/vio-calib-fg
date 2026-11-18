import numpy as np
import pytest

from benchmark import observations
from online_calib import (
    OnlineCalibrator,
    batch_calibrate,
    integrate_gyro,
    numeric_jacobian,
    residual_jacobian,
    robust_weights,
)
from so3 import rotation_error_rad, so3_exp
from synthetic import make_sequence


def constant_gyro_sequence(rate=(0.01, -0.02, 0.53), bias=(0.01, -0.02, 0.03)):
    t_imu = np.arange(0.0, 2.0, 0.005)
    omega = np.tile(np.asarray(rate) + np.asarray(bias), (len(t_imu), 1))
    return t_imu, omega


def test_integrate_gyro_constant_rate_with_bias():
    t_imu, omega = constant_gyro_sequence()
    R = integrate_gyro(0.3, 0.7, t_imu, omega, 0.0, np.asarray([0.01, -0.02, 0.03]))
    expected = so3_exp(np.array([0.01, -0.02, 0.53]) * 0.4)
    np.testing.assert_allclose(R, expected, atol=1e-12)


def test_integrate_gyro_shifted_interval_is_exact_for_constant_rate():
    t_imu, omega = constant_gyro_sequence()
    R = integrate_gyro(0.3, 0.7, t_imu, omega, 0.012,
                       np.asarray([0.01, -0.02, 0.03]))
    np.testing.assert_allclose(R, so3_exp(np.array([0.01, -0.02, 0.53]) * 0.4), atol=1e-12)


def test_integrate_gyro_empty_interval_is_identity():
    t_imu, omega = constant_gyro_sequence()
    np.testing.assert_allclose(integrate_gyro(0.7, 0.3, t_imu, omega), np.eye(3))


def test_analytic_jacobian_matches_finite_difference():
    seq = make_sequence("clean", seed=3, base={"duration": 6.0})
    obs = observations(seq)[:10]
    for estimate_bias in (False, True):
        n = 7 if estimate_bias else 4
        rng = np.random.default_rng(7)
        theta = np.zeros(n)
        theta[:3] = rng.normal(0.0, 0.05, 3)
        theta[3] = 0.013
        if estimate_bias:
            theta[4:] = [1e-3, -2e-3, 5e-4]
        _, J = residual_jacobian(theta, obs, 0.01, estimate_bias)
        Jn = numeric_jacobian(theta, obs, 0.01, estimate_bias, step=1e-6)
        assert np.abs(J - Jn).max() < 1e-5


def test_offline_recovers_offset_and_rotation():
    seq = make_sequence("clean", seed=0, base={"duration": 20.0})
    info = batch_calibrate(observations(seq), np.zeros(4),
                           sigma_r=0.01, estimate_bias=False, max_iters=30)
    assert abs(info["dt"] - seq["dt_true"]) < 1e-3
    R_est = so3_exp(np.asarray(info["rvec_bc"]))
    assert np.rad2deg(rotation_error_rad(R_est, seq["R_bc_true"])) < 0.5
    assert info["nfev"] <= 30


def test_online_converges_and_tracks_with_forgetting():
    seq = make_sequence("clean", seed=0, base={"duration": 15.0})
    cal = OnlineCalibrator(window=20, estimate_bias=False, max_iters=8,
                           marginalize=True, forget=0.99)
    for obs in observations(seq):
        cal.push(*obs)
    assert len(cal.history) == len(seq["T_meas"])
    assert abs(cal.theta[3] - seq["dt_true"]) < 1e-3
    R_est = so3_exp(cal.theta[:3])
    assert np.rad2deg(rotation_error_rad(R_est, seq["R_bc_true"])) < 0.6


def test_online_tracks_a_time_offset_step():
    seq = make_sequence("time_offset_step", seed=0, base={"duration": 20.0},
                        params={"step_ms": 5.0, "step_at_s": 10.0})
    cal = OnlineCalibrator(window=20, estimate_bias=False, max_iters=8,
                           marginalize=True, forget=0.99)
    for obs in observations(seq):
        cal.push(*obs)
    # The final estimate should follow the stepped truth, unlike a constant fit.
    assert abs(cal.theta[3] - seq["dt_true_final"]) < 2e-3
    assert cal.theta[3] > seq["dt_true"] + 2e-3


def test_optional_bias_state_is_finite_and_reasonable():
    seq = make_sequence("clean", seed=1, base={"duration": 10.0})
    info = batch_calibrate(observations(seq), np.zeros(7),
                           sigma_r=0.01, estimate_bias=True, max_iters=20)
    assert abs(info["dt"] - seq["dt_true"]) < 2e-3
    assert np.all(np.isfinite(info["bias_g"]))
    assert np.linalg.norm(info["bias_g"]) < 1.5e-2  # weakly observable over 10 s


def test_huber_and_chi_square_weights():
    r = np.array([0.1, 0.1, 0.1,        # inside Huber
                  1.5, 1.5, 1.5,        # Huber zone (norm^2 = 6.75)
                  5.0, 5.0, 5.0])       # rejected at 99 %
    w = robust_weights(r, huber_k=1.345, chi2_thresh=11.3449)
    assert w[0] == 1.0
    assert 0.0 < w[1] < 1.0
    assert w[2] == 0.0
    assert w[1] == pytest.approx(1.345 / np.linalg.norm(r[3:6]))


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError):
        OnlineCalibrator(window=0)
    with pytest.raises(ValueError):
        OnlineCalibrator(chi2_level=0.5)
    with pytest.raises(ValueError):
        OnlineCalibrator(forget=0.0)
    with pytest.raises(ValueError):
        OnlineCalibrator(init="warm")
