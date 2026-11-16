"""Plots for the VIO calibration / factor-graph prototype."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    metrics = json.loads(Path("results/metrics.json").read_text(encoding="utf-8"))
    poses = np.load("results/poses.npz")
    gt = poses["gt"]
    results = Path("results")
    results.mkdir(exist_ok=True)

    plt.figure(figsize=(6, 5))
    for key, label in [("gt", "GT"), ("baseline", "Baseline (no calib.)"), ("calibrated", "Calibrated chain"), ("fused", "Visual+gyro graph")]:
        p = poses[key][:, :3, 3]
        plt.plot(p[:, 0], p[:, 1], label=label, linewidth=1.8)
    plt.axis("equal")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Trajectory (top view)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(results / "trajectory.png", dpi=160)
    plt.close()

    labels = ["Baseline", "Calibrated", "Fused graph", "Oracle ext."]
    ate = [metrics[k]["ate_rmse_m"] for k in ["baseline", "calibrated_chain", "fused_graph", "oracle_extrinsic"]]
    rpe_rot = [metrics[k]["rpe_rot_rmse_rad"] for k in ["baseline", "calibrated_chain", "fused_graph", "oracle_extrinsic"]]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].bar(labels, ate, color=["#8c8c8c", "#4c78a8", "#f58518", "#54a24b"])
    axes[0].set_ylabel("ATE RMSE [m]")
    axes[0].set_title("Absolute trajectory error")
    axes[1].bar(labels, rpe_rot, color=["#8c8c8c", "#4c78a8", "#f58518", "#54a24b"])
    axes[1].set_ylabel("RPE rotation RMSE [rad]")
    axes[1].set_title("Relative rotation error")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=15)
    plt.tight_layout()
    plt.savefig(results / "metrics.png", dpi=160)
    plt.close()
    print("saved results/trajectory.png, results/metrics.png")


if __name__ == "__main__":
    main()
