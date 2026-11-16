# 室内定位学习与评测

在原 `vio-calib` Conda 环境运行的独立民用离线学习项目，包含两条链路：

- **真实室内数据**：TUM VI room1 双目图像 → 校正 → KLT 跟踪 → 三角化 → PnP → 轨迹评测；IMU 用于独立旋转对照。
- **仿真原型**：时间偏移／外参旋转标定，以及视觉相对位姿＋陀螺旋转位姿图。

当前未实现完整 VIO，也没有运行 OpenVINS／VINS／ORB-SLAM3。

## 直接查看

- [室内评测总览](results/indoor_index.html)：点击两个片段名称查看完整报告。
- [学习路线](LEARNING.md)：数据流、坐标约定、阅读顺序和实验问题。
- [技术报告](report.md)：运行结果、评测口径和能力边界。

HTML 在本地浏览器直接打开，单个片段报告内嵌图像，无外部脚本或服务依赖。

## 在现有环境复现

在项目根目录执行：

```powershell
& "$env:USERPROFILE/miniconda3/envs/vio-calib/python.exe" scripts/run_indoor_benchmark.py
& "$env:USERPROFILE/miniconda3/envs/vio-calib/python.exe" -m pytest -q
```

或者先 `conda activate vio-calib`，然后使用 `python`。新建环境可使用 `environment.yml`；本轮没有新建环境。

首次准备真实数据：

```powershell
python scripts/fetch_indoor.py
```

下载官方 TUM VI room1 512×512／16-bit 数据包，约 1.59 GiB，支持断点续传。
压缩包与解包数据均留在 `data/indoor/`，占用大于下载大小。
来源及实际字节数见 `data/indoor/download.json`；使用应遵守官方数据集条款。

单独运行一个片段：

```powershell
python scripts/run_indoor.py --dataset data/indoor/dataset-room1_512_16 --out results/my_clip --start 5 --duration 20 --stride 2
```

评测已有方法导出的 **T_world_body** 轨迹（TUM 格式、米制、`qx qy qz qw`）：

```powershell
python scripts/run_indoor.py --dataset data/indoor/dataset-room1_512_16 --trajectory results/indoor_dev/trajectory.tum --out results/imported --start 5 --duration 20 --stride 2
```

`--start` 相对录制的第一对双目帧；`--duration` 为选取范围。
跟踪失败会保存原因和已完成轨迹，生成报告并返回退出码 2；不会填入虚假静止位姿。

## 已验证的真实数据结果

同一房间两个不重叠片段，固定参数，20 Hz 图像隔帧处理约 10 Hz。
真值仅在估计完成后用于评测；采用 SE(3) 对齐，**不拟合尺度**。

| 片段 | 输出帧 | ATE RMSE | 1 s RPE 平移 | 1 s RPE 旋转 |
| --- | ---: | ---: | ---: | ---: |
| 5–25 s | 200/200 | 0.1414 m | 0.0562 m | 1.2809° |
| 35–55 s | 200/200 | 0.1294 m | 0.0525 m | 1.1908° |

ATE 已用独立 evo 实现核对，差异小于 1e-9 m。时间上分开的验证不代表跨房间泛化。

## 保留的仿真实验

```powershell
python scripts/run_all.py
```

120 关键帧／12 s／200 Hz IMU；视觉输入为带噪相对位姿而非图像。
时间偏移 12 ms → 11.90 ms，旋转外参误差 0.0385°。

| 方法 | 修正后的 ATE RMSE | 相邻帧旋转 RPE |
| --- | ---: | ---: |
| 未标定链式 | 0.046389 m | 0.003839 rad |
| 标定链式 | 0.046389 m | 0.000326 rad |
| 视觉＋陀螺位姿图 | 0.046222 m | 0.000163 rad |

评测已修复刚体对齐的 SVD 旋转方向。旧“ATE 约减半”结论失效：正确对齐后，
旋转标定未降低该仿真链式轨迹的 ATE，旋转 RPE 则改善。

## 文件与边界

- `src/indoor_data.py`：EuRoC 格式与 TUM VI 官方导出格式读取、双目校正。
- `src/stereo_baseline.py`：仅视觉的双目 VO，前后向光流和几何检查。
- `src/trajectory_eval.py`：纳秒时间戳、真值插值、ATE／RPE、TUM 导入导出。
- `src/imu_diagnostics.py`：独立陀螺旋转对照，零偏假设为零。
- `results/indoor_*/`：轨迹、指标、逐帧跟踪记录、报告和图像。

真实数据链路没有融合加速度计、速度、零偏，没有回环、在线标定或实时部署。
仿真链路仅优化位姿和旋转约束，外参平移固定为零；双目 VO 的尺度来自双目基线。

参考：[TUM VI](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset)、
[官方数据目录](https://vision.in.tum.de/tumvi/exported/euroc/512_16/)、
[OpenCV PnP](https://docs.opencv.org/4.12.0/d5/d1f/calib3d_solvePnP.html)。
