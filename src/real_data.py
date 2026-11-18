"""Loader for the small derived TUM VI calibration asset.

The asset is produced by ``scripts/prepare_real_clip.py`` from a local TUM VI
room1 download and an existing stereo-VO trajectory.  It contains camera-frame
relative poses (the real visual front end output), gyro samples and ground-truth
body poses, in the same shape as ``synthetic.make_sequence``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FIELDS = (
    "t_kf", "t_imu", "T_meas", "omega_meas", "T_wb_gt",
    "R_bc_true", "t_bc_true", "b_g_true", "dt_true_intervals",
)


def load_sequence(path: str | Path) -> dict:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in FIELDS if key not in data]
        if missing:
            raise ValueError(f"{path}: missing fields {missing}")
        seq = {key: np.array(data[key]) for key in FIELDS}
        seq["dt_true"] = float(data["dt_true"])
        seq["dt_true_final"] = float(data["dt_true_final"])
        seq["scenario"] = str(data["scenario"])
        seq["source"] = str(data["source"]) if "source" in data else str(path)
    seq["R_vis"] = seq["T_meas"][:, :3, :3]
    if len(seq["T_meas"]) != len(seq["t_kf"]) - 1:
        raise ValueError("T_meas length must be len(t_kf) - 1")
    if len(seq["T_wb_gt"]) != len(seq["t_kf"]):
        raise ValueError("T_wb_gt length must match t_kf")
    return seq
