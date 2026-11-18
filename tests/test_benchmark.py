import numpy as np

import benchmark
from synthetic import make_sequence

CFG = {
    "estimate_bias": False,
    "marginalize": True,
    "forget": 0.99,
    "window": 20,
    "sigma_r": 0.01,
    "huber_k": 1.345,
    "chi2_level": 0.99,
    "damping": 1e-6,
    "offline_max_iters": 20,
    "online_max_iters": 8,
    "ate_stride": 20,
    "success_dt_ms": 1.0,
    "success_rot_deg": 0.5,
    "convergence_dt_ms": 2.0,
    "convergence_rot_deg": 1.0,
    "steady_tail": 10,
}


def test_four_chains_produce_finite_metrics():
    seq = make_sequence("clean", seed=0, base={"duration": 4.0})
    result = benchmark.run_sequence(seq, CFG, "test", "deadbeef", "hash", 0, "clean", "synthetic")
    assert set(result["chains"]) == {"uncalibrated", "offline", "online", "ground_truth"}
    for chain in result["chains"].values():
        for key in ("ate_rmse_m", "rpe_translation_rmse_m", "rpe_rotation_rmse_deg"):
            assert np.isfinite(chain[key])
        assert "dt_err_ms" in chain["calib"]
        assert "rot_err_deg" in chain["calib"]
    assert result["chains"]["ground_truth"]["success"] is True
    assert result["chains"]["uncalibrated"]["success"] is False


def test_aggregate_medians_and_failure_rate():
    seq = make_sequence("clean", seed=0, base={"duration": 4.0})
    a = benchmark.run_sequence(seq, CFG, "a", "c", "h", 0, "clean", "s")
    b = benchmark.run_sequence(seq, CFG, "b", "c", "h", 1, "clean", "s")
    rows = benchmark.aggregate([a, b])
    clean = [row for row in rows if row["scenario"] == "clean"]
    assert len(clean) == 4
    oracle = next(row for row in clean if row["chain"] == "ground_truth")
    assert oracle["dt_err_ms"] == 0.0
    assert oracle["failure_rate"] == 0.0
    uncal = next(row for row in clean if row["chain"] == "uncalibrated")
    assert uncal["failure_rate"] == 1.0
    assert np.isfinite(uncal["ate_rmse_m"])


def test_convergence_finds_first_sustained_index():
    history = [
        {"keyframe": 1, "dt": 0.03, "rvec_bc": [0.0, 0.0, 0.0]},
        {"keyframe": 2, "dt": 0.012, "rvec_bc": [0.0, 0.0, 0.0]},
        {"keyframe": 3, "dt": 0.030, "rvec_bc": [0.0, 0.0, 0.0]},
        {"keyframe": 4, "dt": 0.0125, "rvec_bc": [0.0, 0.0, 0.0]},
        {"keyframe": 5, "dt": 0.0124, "rvec_bc": [0.0, 0.0, 0.0]},
    ]
    kf, _ = benchmark.convergence(history, 0.012, np.eye(3), 2e-3, 1.0)
    assert kf == 4
    never = [dict(entry, dt=0.05) for entry in history]
    assert benchmark.convergence(never, 0.012, np.eye(3), 2e-3, 1.0) == (None, None)
