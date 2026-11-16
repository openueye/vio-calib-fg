"""Exercise the real loader and CLI failure contract with a textureless recording."""
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import yaml


def test_textureless_recording_saves_failure_and_does_not_invent_motion(tmp_path):
    dataset = tmp_path / "recording" / "mav0"
    stamps = [1000000000, 1100000000, 1200000000, 1300000000]
    for i in (0, 1):
        camera = dataset / f"cam{i}"
        (camera / "data").mkdir(parents=True)
        T = np.eye(4)
        T[0, 3] = i * 0.1
        metadata = {"T_BS": {"data": T.ravel().tolist()}, "resolution": [160, 120],
                    "intrinsics": [100., 100., 80., 60.], "distortion_coefficients": [0., 0., 0., 0.],
                    "distortion_model": "radial-tangential"}
        (camera / "sensor.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
        rows = []
        for stamp in stamps:
            name = f"{stamp}.png"
            cv2.imwrite(str(camera / "data" / name), np.zeros((120, 160), np.uint8))
            rows.append(f"{stamp},{name}")
        (camera / "data.csv").write_text("\n".join(rows), encoding="utf-8")
    (dataset / "imu0").mkdir()
    (dataset / "imu0/data.csv").write_text("\n".join(f"{stamp},0,0,0,0,0,9.81" for stamp in stamps), encoding="utf-8")
    (dataset / "state_groundtruth_estimate0").mkdir()
    (dataset / "state_groundtruth_estimate0/data.csv").write_text(
        "\n".join(f"{stamp},0,0,0,1,0,0,0" for stamp in stamps), encoding="utf-8")
    out = tmp_path / "results"
    process = subprocess.run([
        sys.executable, str(Path(__file__).resolve().parents[1] / "scripts/run_indoor.py"),
        "--dataset", str(dataset), "--out", str(out), "--start", "0", "--duration", "1", "--stride", "1",
    ], capture_output=True, text=True)
    assert process.returncode == 2, process.stdout + process.stderr
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["estimated_frames"] == 1
    assert summary["frame_coverage"] == 0.25
    assert summary["failure"]["inliers"] == 0
    assert summary["evaluation_error"]
    assert (out / "report.html").exists()
    assert len([line for line in (out / "trajectory.tum").read_text().splitlines() if not line.startswith("#")]) == 1
