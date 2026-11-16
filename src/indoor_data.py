"""Read EuRoC-format stereo/IMU recordings. All transforms use T_target_source."""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import yaml

from se3 import pose_inv


def sensor_yaml(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return yaml.safe_load("\n".join(line for line in text.splitlines() if not line.startswith("%YAML")))


def transform(value) -> np.ndarray:
    if isinstance(value, dict):
        value = value["data"]
    return np.asarray(value, dtype=float).reshape(4, 4)


def image_index(path: Path) -> dict[int, Path]:
    with (path / "data.csv").open(encoding="utf-8") as stream:
        rows = csv.reader(line for line in stream if not line.startswith("#"))
        return {int(row[0]): path / "data" / row[1].strip() for row in rows if row}


class IndoorDataset:
    def __init__(self, root: Path):
        self.root = root / "mav0" if (root / "mav0").is_dir() else root
        calibration_path = self.root.parent / "dso" / "camchain.yaml"
        if (self.root / "cam0" / "sensor.yaml").exists():
            self.cameras = [sensor_yaml(self.root / f"cam{i}" / "sensor.yaml") for i in range(2)]
            self.calibration_source = "mav0/cam*/sensor.yaml, T_BS"
        else:
            # Official TUM VI export stores calibration beside mav0, in Kalibr convention.
            chain = sensor_yaml(calibration_path)
            self.cameras = []
            for i in range(2):
                camera = dict(chain[f"cam{i}"])
                camera["T_BS"] = pose_inv(transform(camera["T_cam_imu"]))
                camera["distortion_coefficients"] = camera["distortion_coeffs"]
                self.cameras.append(camera)
            self.calibration_source = str(calibration_path)
        self.images = [image_index(self.root / f"cam{i}") for i in range(2)]
        self.timestamps = np.array(sorted(self.images[0].keys() & self.images[1].keys()), dtype=np.int64)
        if len(self.timestamps) < 2:
            raise ValueError("Need at least two synchronized stereo image pairs")
        imu = pd.read_csv(self.root / "imu0" / "data.csv", comment="#", header=None)
        self.imu_ns = imu.iloc[:, 0].to_numpy(dtype=np.int64)
        self.gyro = imu.iloc[:, 1:4].to_numpy(dtype=float)
        self.accel = imu.iloc[:, 4:7].to_numpy(dtype=float)
        if np.any(np.diff(self.imu_ns) <= 0):
            raise ValueError("IMU timestamps must be strictly increasing")

    def groundtruth(self) -> tuple[np.ndarray, np.ndarray]:
        """Load GT only for evaluation, never for visual motion estimation."""
        imu_gt = self.root.parent / "dso" / "gt_imu.csv"
        for name in ("state_groundtruth_estimate0", "mocap0"):
            directory = self.root / name
            if (directory / "data.csv").exists():
                break
        else:
            raise FileNotFoundError("No state_groundtruth_estimate0/data.csv or mocap0/data.csv")
        source = imu_gt if imu_gt.exists() else directory / "data.csv"
        self.groundtruth_source = str(source)
        frame = pd.read_csv(source, comment="#", header=None)
        ns = frame.iloc[:, 0].to_numpy(dtype=np.int64)
        data = frame.iloc[:, 1:8].to_numpy(dtype=float)
        poses = np.repeat(np.eye(4)[None], len(ns), axis=0)
        poses[:, :3, 3] = data[:, :3]
        # EuRoC-format GT quaternion is w,x,y,z; scipy takes x,y,z,w.
        poses[:, :3, :3] = Rotation.from_quat(data[:, [4, 5, 6, 3]]).as_matrix()
        if name == "mocap0" and not imu_gt.exists():
            # T_WS -> T_WB: marker sensor S is rigidly attached to IMU/body B.
            calibration = sensor_yaml(directory / "sensor.yaml")
            poses = poses @ pose_inv(transform(calibration["T_BS"]))
        if np.any(np.diff(ns) <= 0):
            raise ValueError("Ground-truth timestamps must be strictly increasing")
        return ns, poses

    def select(self, start: float, duration: float, stride: int) -> np.ndarray:
        if start < 0 or duration <= 0 or stride < 1:
            raise ValueError("start >= 0, duration > 0 and stride >= 1 are required")
        relative = (self.timestamps - self.timestamps[0]) * 1e-9
        selected = self.timestamps[(relative >= start) & (relative < start + duration)][::stride]
        if len(selected) < 2:
            raise ValueError("Selected interval contains fewer than two stereo frames")
        return selected


class Rectifier:
    def __init__(self, cameras: list[dict]):
        self.size = tuple(cameras[0]["resolution"])
        matrices, distortions = [], []
        for camera in cameras:
            fx, fy, cx, cy = camera["intrinsics"]
            matrices.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]))
            distortions.append(np.asarray(camera["distortion_coefficients"], dtype=float))
        T_BC = [transform(camera["T_BS"]) for camera in cameras]
        T_C1C0 = pose_inv(T_BC[1]) @ T_BC[0]
        model = cameras[0]["distortion_model"]
        if model != cameras[1]["distortion_model"]:
            raise ValueError("Both cameras must use the same distortion model")
        if model == "equidistant":
            R0, R1, P0, P1, _ = cv2.fisheye.stereoRectify(
                matrices[0], distortions[0], matrices[1], distortions[1], self.size,
                T_C1C0[:3, :3], T_C1C0[:3, 3], flags=cv2.CALIB_ZERO_DISPARITY,
                balance=0.0, fov_scale=1.0,
            )
            make_map = cv2.fisheye.initUndistortRectifyMap
        elif model in ("radial-tangential", "radtan"):
            R0, R1, P0, P1, _, _, _ = cv2.stereoRectify(
                matrices[0], distortions[0], matrices[1], distortions[1], self.size,
                T_C1C0[:3, :3], T_C1C0[:3, 3], flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
            )
            make_map = cv2.initUndistortRectifyMap
        else:
            raise ValueError(f"Unsupported distortion model: {model}")
        self.maps = [make_map(K, D, R, P[:3, :3], self.size, cv2.CV_32FC1)
                     for K, D, R, P in zip(matrices, distortions, (R0, R1), (P0, P1))]
        self.K, self.P0, self.P1 = P0[:3, :3], P0, P1
        # Rectification maps original camera coordinates into rectified coordinates.
        original_from_rectified = np.eye(4)
        original_from_rectified[:3, :3] = R0.T
        self.T_BC = T_BC[0] @ original_from_rectified
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def read(self, dataset: IndoorDataset, ns: int) -> tuple[np.ndarray, np.ndarray]:
        pair = []
        for i in range(2):
            path = dataset.images[i][int(ns)]
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise FileNotFoundError(path)
            if image.dtype == np.uint16:
                # TUM VI publishes 16-bit images; use the same fixed mapping for both cameras.
                image = (image >> 8).astype(np.uint8)
            elif image.dtype != np.uint8:
                raise ValueError(f"Expected 8/16-bit image: {path}")
            pair.append(self.clahe.apply(cv2.remap(image, *self.maps[i], cv2.INTER_LINEAR)))
        return tuple(pair)
