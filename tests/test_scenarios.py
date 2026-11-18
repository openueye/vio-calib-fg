import numpy as np
import pytest

from synthetic import SCENARIOS, make_sequence

BASE = {"duration": 6.0}


def test_all_scenarios_are_deterministic():
    for scenario in SCENARIOS:
        a = make_sequence(scenario, seed=4, base=BASE)
        b = make_sequence(scenario, seed=4, base=BASE)
        for key in ("t_kf", "t_imu", "T_meas", "omega_meas", "T_wb_gt"):
            np.testing.assert_array_equal(a[key], b[key], err_msg=f"{scenario}:{key}")
        assert a["scenario"] == b["scenario"]


def test_different_seeds_change_noise_but_not_truth():
    a = make_sequence("clean", seed=1, base=BASE)
    b = make_sequence("clean", seed=2, base=BASE)
    np.testing.assert_array_equal(a["T_wb_gt"], b["T_wb_gt"])
    assert not np.array_equal(a["omega_meas"], b["omega_meas"])
    assert not np.array_equal(a["T_meas"], b["T_meas"])


def test_frame_drop_keeps_visual_and_time_alignment():
    seq = make_sequence("frame_drop", seed=0, base=BASE,
                        params={"imu_frac": 0.2, "vis_frac": 0.1})
    assert len(seq["T_meas"]) == len(seq["t_kf"]) - 1
    assert len(seq["R_vis"]) == len(seq["T_meas"])
    assert len(seq["T_wb_gt"]) == len(seq["t_kf"])
    assert len(seq["dt_true_intervals"]) == len(seq["T_meas"])
    assert len(seq["t_imu"]) < len(make_sequence("clean", seed=0, base=BASE)["t_imu"])
    assert np.all(np.diff(seq["t_kf"]) > 0)


def test_time_offset_step_truth_profile():
    seq = make_sequence("time_offset_step", seed=0, base={"duration": 10.0},
                        params={"step_ms": 5.0, "step_at_s": 4.0})
    assert seq["dt_true"] == pytest.approx(0.012)
    assert seq["dt_true_final"] == pytest.approx(0.017)
    before = seq["dt_true_intervals"][seq["t_kf"][:-1] < 4.0]
    after = seq["dt_true_intervals"][seq["t_kf"][:-1] >= 4.0]
    assert np.allclose(before, 0.012)
    assert np.allclose(after, 0.017)


def test_vibration_and_thermal_change_the_measurements():
    clean = make_sequence("clean", seed=0, base=BASE)
    vibration = make_sequence("vibration", seed=0, base=BASE,
                              params={"start_s": 2.0, "duration_s": 2.0})
    drift = make_sequence("thermal_drift", seed=0, base=BASE)
    assert not np.array_equal(clean["omega_meas"], vibration["omega_meas"])
    assert not np.array_equal(clean["T_meas"][:, :3, 3], vibration["T_meas"][:, :3, 3])
    delta = drift["omega_meas"] - clean["omega_meas"]
    np.testing.assert_allclose(delta[0], 0.0, atol=1e-15)
    assert np.all(np.diff(delta[:, 0]) > 0)  # linear ramp along the first axis


def test_unknown_scenario_and_parameter_rejected():
    with pytest.raises(ValueError, match="unknown scenario"):
        make_sequence("not_a_scenario")
    with pytest.raises(ValueError, match="unknown vibration parameters"):
        make_sequence("vibration", params={"amplitude": 1.0})
