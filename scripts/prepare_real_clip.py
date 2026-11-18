"""Derive a small TUM VI room1 calibration asset from a local download.

Inputs (both local; nothing is downloaded):
  * data/indoor/dataset-room1_512_16  - official TUM VI room1 512x16 export
  * results/indoor_dev/trajectory.tum - stereo-VO T_world_body trajectory

The output ``assets/tumvi_room1_20s.npz`` is a few hundred kilobytes and can be
uploaded to the remote host.  It contains real camera-frame relative poses
from the stereo VO, 200 Hz gyro samples, ground-truth body poses, and the
camchain extrinsic (rectified camera -> body).  TUM VI cameras and IMU share a
hardware clock, so ``dt_true = 0``; the clock-offset estimate is therefore a
null check on real data.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indoor_data import IndoorDataset, Rectifier  # noqa: E402
from se3 import pose_inv, so3_log  # noqa: E402
from trajectory_eval import interpolate_poses, load_tum  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path,
                    default=ROOT / "data/indoor/dataset-room1_512_16")
    ap.add_argument("--trajectory", type=Path,
                    default=ROOT / "results/indoor_dev/trajectory.tum")
    ap.add_argument("--out", type=Path, default=ROOT / "assets/tumvi_room1_20s.npz")
    ap.add_argument("--margin", type=float, default=0.2,
                    help="seconds of IMU kept beyond both clip ends")
    args = ap.parse_args()

    dataset = IndoorDataset(args.dataset)
    rectifier = Rectifier(dataset.cameras)
    T_BC = rectifier.T_BC  # body from rectified camera
    kf_ns, T_wb = load_tum(args.trajectory)
    if len(kf_ns) < 10:
        raise ValueError("trajectory is too short")
    origin = int(kf_ns[0])
    t_kf = (kf_ns - origin) * 1e-9
    if np.any(np.diff(t_kf) <= 0):
        raise ValueError("trajectory timestamps must increase")

    T_wc = np.stack([T @ T_BC for T in T_wb])
    T_meas = np.stack([pose_inv(T_wc[k]) @ T_wc[k + 1] for k in range(len(T_wc) - 1)])

    lo = kf_ns[0] - int(args.margin * 1e9)
    hi = kf_ns[-1] + int(args.margin * 1e9)
    mask = (dataset.imu_ns >= lo) & (dataset.imu_ns <= hi)
    imu_ns = dataset.imu_ns[mask]
    omega = dataset.gyro[mask]
    t_imu = (imu_ns - origin) * 1e-9

    gt_ns, gt_poses = dataset.groundtruth()
    T_wb_gt = interpolate_poses(gt_ns, gt_poses, kf_ns)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        t_kf=t_kf,
        t_imu=t_imu,
        T_meas=T_meas,
        omega_meas=omega,
        T_wb_gt=T_wb_gt,
        R_bc_true=T_BC[:3, :3],
        t_bc_true=T_BC[:3, 3],
        b_g_true=np.zeros(3),
        dt_true=np.array(0.0),
        dt_true_final=np.array(0.0),
        dt_true_intervals=np.zeros(len(T_meas)),
        scenario=np.array("tumvi_room1"),
        source=np.array(f"{args.dataset} + {args.trajectory}; camchain T_cam_imu; dt_true=0"),
    )
    print(f"wrote {args.out} | keyframes={len(t_kf)} intervals={len(T_meas)} "
          f"imu={len(t_imu)} duration={t_kf[-1] - t_kf[0]:.1f}s")
    print(f"R_bc_true rotvec={so3_log(T_BC[:3, :3])} t_bc={T_BC[:3, 3]}")


if __name__ == "__main__":
    main()
