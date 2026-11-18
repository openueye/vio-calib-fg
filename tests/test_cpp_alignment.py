"""Numeric alignment between the C++/Eigen core and the NumPy reference.

Skipped when the pybind11 module has not been built (``cmake --build build``).
Set ``CALIB_CORE_PATH`` to the directory or file of ``calib_core`` to override
the search path.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from benchmark import observations
from online_calib import batch_calibrate, integrate_gyro, residual_jacobian, robust_weights
from so3 import so3_exp, so3_log
from synthetic import make_sequence

ROOT = Path(__file__).resolve().parents[1]


def _find_module():
    override = os.environ.get("CALIB_CORE_PATH")
    candidates = []
    if override:
        path = Path(override)
        candidates.extend([path] if path.is_file() else sorted(path.glob("calib_core*")))
    for directory in (ROOT / "build", ROOT / "cpp" / "build"):
        candidates.extend(sorted(directory.glob("calib_core*.so")))
        candidates.extend(sorted(directory.glob("calib_core*.pyd")))
        candidates.extend(sorted(directory.glob("**/calib_core*.so")))
    for candidate in candidates:
        if candidate.suffix in (".so", ".pyd") or candidate.name.endswith((".so", ".pyd")):
            spec = importlib.util.spec_from_file_location("calib_core", candidate)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module
    return None


MODULE = _find_module()
pytestmark = pytest.mark.skipif(MODULE is None, reason="calib_core C++ module not built")


def test_so3_alignment():
    rng = np.random.default_rng(0)
    for _ in range(20):
        phi = rng.normal(size=3) * 0.5
        np.testing.assert_allclose(MODULE.so3_exp(phi), so3_exp(phi), atol=1e-12)
        R = so3_exp(phi)
        np.testing.assert_allclose(MODULE.so3_log(R), so3_log(R), atol=1e-12)
        np.testing.assert_allclose(MODULE.left_jacobian(phi), _left_jacobian(phi), atol=1e-12)


def _left_jacobian(phi):
    from so3 import left_jacobian
    return left_jacobian(phi)


def test_integrate_gyro_alignment():
    seq = make_sequence("clean", seed=0, base={"duration": 4.0})
    for k in range(0, 10):
        python = integrate_gyro(seq["t_kf"][k], seq["t_kf"][k + 1],
                                seq["t_imu"], seq["omega_meas"], 0.011,
                                np.array([1e-4, -2e-4, 3e-4]))
        cpp = MODULE.integrate_gyro(seq["t_kf"][k], seq["t_kf"][k + 1],
                                    seq["t_imu"], seq["omega_meas"], 0.011,
                                    np.array([1e-4, -2e-4, 3e-4]))
        np.testing.assert_allclose(cpp, python, atol=1e-13)


def test_residual_and_jacobian_alignment():
    seq = make_sequence("clean", seed=0, base={"duration": 4.0})
    obs = observations(seq)[:8]
    for estimate_bias in (False, True):
        n = 7 if estimate_bias else 4
        theta = np.zeros(n)
        theta[:3] = [0.03, -0.02, 0.05]
        theta[3] = 0.0115
        if estimate_bias:
            theta[4:] = [1e-4, -2e-4, 3e-4]
        r_py, J_py = residual_jacobian(theta, obs, 0.01, estimate_bias)
        r_cpp, J_cpp = MODULE.residual_jacobian(theta, _obs_list(obs), 0.01, estimate_bias)
        assert np.abs(r_cpp - r_py).max() < 1e-10
        assert np.abs(J_cpp - J_py).max() < 1e-9


def test_robust_weights_alignment():
    rng = np.random.default_rng(3)
    r = rng.normal(size=(15,)) * 2.0
    np.testing.assert_allclose(
        MODULE.robust_weights(r, 1.345, 11.3449),
        robust_weights(r, 1.345, 11.3449), atol=1e-15)


def test_batch_calibration_alignment():
    seq = make_sequence("clean", seed=0, base={"duration": 8.0})
    obs = observations(seq)
    py = batch_calibrate(obs, np.zeros(4), 0.01, False, 20, 1e-6)
    cpp = MODULE.batch_calibrate(_obs_list(obs), np.zeros(4), 0.01, False, 20, 1e-6,
                                 1.345, 0.99)
    assert abs(cpp["dt"] - py["dt"]) < 1e-7
    np.testing.assert_allclose(cpp["rvec_bc"], py["rvec_bc"], atol=1e-7)
    assert abs(cpp["cost"] - py["cost"]) < 1e-7
    assert cpp["n_outliers"] == py["n_outliers"]


def _obs_list(obs):
    return [(R, t0, t1, t_imu, omega) for (R, t0, t1, t_imu, omega) in obs]
