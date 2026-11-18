import numpy as np
import pytest

from so3 import (
    hat,
    left_jacobian,
    left_jacobian_inverse,
    right_jacobian_inverse,
    rotation_error_rad,
    so3_exp,
    so3_exp_batch,
    so3_log,
    vee,
)


def test_hat_vee_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(10):
        v = rng.normal(size=3)
        np.testing.assert_allclose(vee(hat(v)), v, atol=1e-15)
        S = hat(v)
        np.testing.assert_allclose(S, -S.T, atol=1e-15)


def test_exp_log_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        phi = rng.normal(size=3)
        phi *= rng.uniform(0.0, 2.5) / max(np.linalg.norm(phi), 1e-12)
        R = so3_exp(phi)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)
        np.testing.assert_allclose(so3_log(R), phi, atol=1e-10)


def test_exp_small_angle_is_first_order():
    phi = np.array([1e-14, -2e-14, 3e-14])
    np.testing.assert_allclose(so3_exp(phi), np.eye(3) + hat(phi), atol=1e-15)
    np.testing.assert_allclose(so3_log(np.eye(3)), np.zeros(3), atol=1e-15)


def test_batch_matches_scalar():
    rng = np.random.default_rng(2)
    Phi = rng.normal(size=(7, 3)) * 0.4
    batch = so3_exp_batch(Phi)
    for k, phi in enumerate(Phi):
        np.testing.assert_allclose(batch[k], so3_exp(phi), atol=1e-14)


def test_left_jacobian_matches_definition():
    rng = np.random.default_rng(3)
    phi = rng.normal(size=3)
    Jl = left_jacobian(phi)
    np.testing.assert_allclose(Jl @ left_jacobian_inverse(phi), np.eye(3), atol=1e-10)
    step = 1e-7
    for j in range(3):
        delta = np.zeros(3)
        delta[j] = step
        lhs = so3_log(so3_exp(phi + delta))
        rhs = so3_log(so3_exp(Jl @ delta) @ so3_exp(phi))
        np.testing.assert_allclose(lhs, rhs, atol=1e-10)


def test_right_jacobian_inverse_matches_definition():
    rng = np.random.default_rng(4)
    phi = rng.normal(size=3)
    Jr_inv = right_jacobian_inverse(phi)
    step = 1e-7
    for j in range(3):
        delta = np.zeros(3)
        delta[j] = step
        lhs = so3_log(so3_exp(phi) @ so3_exp(delta)) - phi
        rhs = Jr_inv @ delta
        np.testing.assert_allclose(lhs, rhs, atol=1e-11)


def test_rotation_error_is_symmetric_and_zero_on_equal():
    rng = np.random.default_rng(5)
    R1 = so3_exp(rng.normal(size=3))
    R2 = so3_exp(rng.normal(size=3))
    assert rotation_error_rad(R1, R1) == pytest.approx(0.0, abs=1e-15)
    assert rotation_error_rad(R1, R2) == pytest.approx(rotation_error_rad(R2, R1), abs=1e-12)
