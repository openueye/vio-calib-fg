"""Python<->C++ alignment through the standalone ``alignment_dump`` executable.

This path does not require pybind11, so the numeric contract can still be
checked on a machine where only a C++ compiler is available.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from benchmark import observations
from online_calib import batch_calibrate, residual_jacobian, robust_weights
from synthetic import make_sequence

ROOT = Path(__file__).resolve().parents[1]


def _find_dump():
    override = os.environ.get("CALIB_ALIGNMENT_DUMP")
    candidates = [Path(override)] if override else []
    for directory in (ROOT / "build", ROOT / "cpp" / "build"):
        candidates.extend([directory / "alignment_dump", directory / "alignment_dump.exe"])
    # The default build directory is build/cpp; mirror the module lookup.
    for candidate in (ROOT / "build").rglob("alignment_dump*"):
        if candidate.is_file() and candidate.suffix in ("", ".exe"):
            candidates.append(candidate)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


DUMP = _find_dump()
pytestmark = pytest.mark.skipif(DUMP is None, reason="alignment_dump executable not built")


def _write_problem(path: Path, obs, theta, sigma, estimate_bias):
    with path.open("w", encoding="utf-8") as stream:
        stream.write(f"{len(obs)} {len(theta)} {sigma!r} {int(estimate_bias)}\n")
        for R, t0, t1, t_imu, omega in obs:
            stream.write(" ".join(f"{float(v):.17g}" for v in np.asarray(R).reshape(-1)) + "\n")
            stream.write(f"{float(t0):.17g} {float(t1):.17g} {len(t_imu)}\n")
            stream.write(" ".join(f"{float(v):.17g}" for v in np.asarray(t_imu)) + "\n")
            stream.write(" ".join(f"{float(v):.17g}" for v in np.asarray(omega).reshape(-1)) + "\n")
        stream.write(" ".join(f"{float(v):.17g}" for v in theta) + "\n")
        stream.write("20 1e-6 1.345 0.99\n")


def _run(tmp_path, estimate_bias):
    seq = make_sequence("clean", seed=0, base={"duration": 3.0})
    obs = observations(seq)[:6]
    theta = np.zeros(7 if estimate_bias else 4)
    theta[:3] = [0.03, -0.02, 0.05]
    theta[3] = 0.0115
    if estimate_bias:
        theta[4:] = [1e-4, -2e-4, 3e-4]
    problem = tmp_path / "problem.txt"
    _write_problem(problem, obs, theta, 0.01, estimate_bias)
    process = subprocess.run([str(DUMP), str(problem)], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    cpp = json.loads(process.stdout)
    r, J = residual_jacobian(theta, obs, 0.01, estimate_bias)
    w = robust_weights(r, 1.345, 11.3449)
    batch = batch_calibrate(obs, theta, 0.01, estimate_bias, 20, 1e-6)
    return cpp, r, J, w, batch


@pytest.mark.parametrize("estimate_bias", [False, True])
def test_dump_alignment(tmp_path, estimate_bias):
    cpp, r, J, w, batch = _run(tmp_path, estimate_bias)
    assert np.abs(np.asarray(cpp["r"]) - r).max() < 1e-9
    assert np.abs(np.asarray(cpp["J"]).reshape(J.shape) - J).max() < 1e-7
    np.testing.assert_allclose(cpp["w"], w, atol=1e-15)
    assert abs(cpp["batch"]["dt"] - batch["dt"]) < 1e-7
    np.testing.assert_allclose(cpp["batch"]["rvec_bc"], batch["rvec_bc"], atol=1e-7)
    assert abs(cpp["batch"]["cost"] - batch["cost"]) < 1e-7
    assert cpp["batch"]["n_outliers"] == batch["n_outliers"]
