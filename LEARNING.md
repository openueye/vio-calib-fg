# 室内定位学习路线

本项目使用公开室内录制数据，研究基础视觉运动估计和评测方法。当前真实数据链路是
**双目 VO + 独立 IMU 旋转对照**，不称完整 VIO。

## 从数据流理解

1. `indoor_data.py`：相机、IMU 都使用整数纳秒时间戳。双目只配对时间戳相同的图像。
   按官方内参、畸变、相机间外参校正图像；16-bit 图像统一右移 8 位，避免左右图分别归一化。
2. `stereo_baseline.py`：左图选角点，左右光流匹配并做前后向一致性检查、极线检查、正深度检查；
   三角化得到米制三维点，再跟踪到下一帧，用 PnP RANSAC 估计运动。
3. `imu_diagnostics.py`：对相同时间区间积分陀螺角速度，独立比较视觉、IMU 与运动捕捉真值的相对旋转。
   默认陀螺零偏为零，不从真值反推零偏；比力虽被读取，但未用于位置估计。
4. `trajectory_eval.py`：插值真值、SE(3) 刚体对齐、ATE 与 1 秒 RPE；报告覆盖率，不拟合尺度。
   `indoor_report.py` 生成本地可直接打开的报告和 PNG。

## 必须分清的坐标关系

统一使用 `T_target_source`，即 `p_target = T_target_source @ p_source`。

- `T_BC`：相机到 IMU/body；校正后还需要乘以原始相机到校正相机的逆旋转。
- PnP 返回 `T_current_previous`，串接相机轨迹时必须取逆。
- 相机轨迹转换为 body 轨迹后才能与 body 真值比较。
- TUM VI mocap 数据若表示标记体 S，必须用标记体外参把 `T_WS` 转成 `T_WB`。
- 真值只在估计完成后进入评测，绝不能用真值平移为视觉轨迹补尺度。

## 推荐的阅读与实验

- 先看 `results/indoor_dev/report.html`，再从报告中的特征预览定位到三角化和 PnP。
- 理解 `tests/test_stereo.py` 的已知三维点投影实验：为什么 PnP 的平移方向与相机移动方向相反？
- 理解 `tests/test_evaluation.py` 的全局旋转、尺度错误实验：为什么前者应被消除，后者必须保留？
- 比较两个不重叠的室内片段；固定参数后再看验证片段，保留较差或失败的结果。
- 调整 `--stride`，观察采样间隔、特征存活、误差和处理耗时如何变化。更快不自动等于更准确。
- 导入其他离线方法的 TUM 轨迹，使用同一评测器比较。应确认其输出已经是 `T_world_body`。

## 方法边界

没有实现加速度计融合、速度/零偏状态估计、滑动窗口优化、回环、在线标定或实时部署。
不会因完成一个录制片段就称“熟悉 OpenVINS”，也不能把双目尺度解释为 IMU 恢复的尺度。
同一房间两个片段只是时间上分开的验证，不能证明跨房间泛化。

## 参考

- [TUM VI 官方数据集](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset)
- [官方室内数据下载目录](https://vision.in.tum.de/tumvi/exported/euroc/512_16/)
- [OpenCV PnP 坐标定义](https://docs.opencv.org/4.12.0/d5/d1f/calib3d_solvePnP.html)
- [OpenVINS 数据集说明](https://docs.openvins.com/gs-datasets.html)

如后续具备 Linux/C++ 构建环境，可把实际运行 OpenVINS 作为独立学习阶段；当前没有完成该阶段。
