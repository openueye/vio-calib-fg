"""Offline civilian indoor stereo VO and independent IMU evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from indoor_data import IndoorDataset, Rectifier
from indoor_report import make_report
from imu_diagnostics import diagnose
from se3 import pose_inv
from stereo_baseline import motion, stereo_points
from trajectory_eval import evaluate, load_tum, save_tum


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True, help="EuRoC-format sequence root or mav0")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "indoor")
    ap.add_argument("--start", type=float, default=5.0, help="Seconds from first stereo frame")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--trajectory", type=Path, help="Evaluate an external TUM-format T_world_body trajectory instead of running VO")
    args = ap.parse_args()
    cv2.setNumThreads(2)
    cv2.setRNGSeed(7)
    dataset = IndoorDataset(args.dataset)
    args.out.mkdir(parents=True, exist_ok=True)
    times, records, failure = [], [], None
    if args.trajectory:
        ns, poses = load_tum(args.trajectory)
        selected = dataset.select(args.start, args.duration, args.stride)
        mask = (ns >= selected[0]) & (ns <= selected[-1])
        ns, poses = ns[mask], poses[mask]
        status = "external trajectory evaluation"
    else:
        rectifier = Rectifier(dataset.cameras)
        selected = dataset.select(args.start, args.duration, args.stride)
        previous, right = rectifier.read(dataset, int(selected[0]))
        points, xyz = stereo_points(previous, right, rectifier.P0, rectifier.P1)
        T_firstC_currentC = np.eye(4)
        body_poses = [T_firstC_currentC @ pose_inv(rectifier.T_BC)]
        for k, timestamp in enumerate(selected[1:], 1):
            start = time.perf_counter()
            current, right = rectifier.read(dataset, int(timestamp))
            relative, info = motion(previous, current, points, xyz, rectifier.K)
            info.update({"timestamp_ns": int(timestamp), "stereo_points": len(xyz)})
            if relative is None:
                failure = info
                break  # Do not insert fake zero-motion poses or silently bridge a failed interval.
            T_firstC_currentC = T_firstC_currentC @ relative
            body_poses.append(T_firstC_currentC @ pose_inv(rectifier.T_BC))
            points, xyz = stereo_points(current, right, rectifier.P0, rectifier.P1)
            previous = current
            elapsed = (time.perf_counter()-start)*1000
            times.append(elapsed)
            records.append({**info, "processing_ms": elapsed})
            if k == 1:
                preview = cv2.cvtColor(current, cv2.COLOR_GRAY2BGR)
                for p in points:
                    cv2.circle(preview, tuple(np.round(p).astype(int)), 2, (80, 220, 100), 1)
                cv2.imwrite(str(args.out / "features.png"), preview)
            if k % 50 == 0:
                print(f"{k}/{len(selected)-1} intervals, {len(xyz)} stereo points, {elapsed:.1f} ms", flush=True)
        ns, poses = selected[:len(body_poses)], np.asarray(body_poses)
        status = "stopped at tracking failure" if failure else "completed selected interval"
    save_tum(args.out / "trajectory.tum", ns, poses)
    gt_ns, gt = dataset.groundtruth()
    evaluation_error = None
    try:
        metrics, aligned, truth = evaluate(ns, poses, gt_ns, gt)
        imu_summary, diagnostics = diagnose(dataset.imu_ns, dataset.gyro, ns, poses, gt_ns, gt)
    except ValueError as error:
        if not failure:
            raise
        evaluation_error = str(error)
        metrics, aligned, truth = {"evaluation_error": evaluation_error}, None, None
        imu_summary, diagnostics = {}, {}
    summary = {
        "dataset": str(dataset.root.resolve()), "status": status,
        "calibration_source": dataset.calibration_source, "groundtruth_source": dataset.groundtruth_source,
        "estimator": "external TUM trajectory" if args.trajectory else "OpenCV stereo KLT + PnP (no IMU fusion)",
        "configuration": {"start_s": args.start, "duration_s": args.duration, "stride": args.stride, "seed": 7},
        "selected_frames": len(selected), "estimated_frames": len(ns),
        "frame_coverage": len(ns)/len(selected) if not args.trajectory else None,
        "time_span_coverage": float((ns[-1]-ns[0])/(selected[-1]-selected[0])) if len(ns) else 0.0,
        "failure": failure, "evaluation_error": evaluation_error,
        "metrics": metrics, "imu_diagnostics": imu_summary,
        "processing_ms": {"p50": float(np.median(times)), "p95": float(np.percentile(times, 95))} if times else None,
        "environment": {"python": sys.version, "executable": sys.executable, "opencv": cv2.__version__, "numpy": np.__version__},
        "limitations": ["educational offline stereo VO, not full VIO or OpenVINS",
                        "no accelerometer fusion, loop closure or online calibration",
                        "GT used only after estimation; gyro bias assumed zero",
                        "single-room clips do not establish cross-scene generalization"],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "tracking.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    make_report(args.out, summary, aligned, truth, diagnostics)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Report: {args.out / 'report.html'}")
    if failure:
        sys.exit(2)


if __name__ == "__main__":
    main()
