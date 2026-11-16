"""Generate an offline report and portable plots from one evaluated run."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def make_report(out: Path, summary: dict, aligned, truth, diagnostics: dict):
    if aligned is None:
        (out / "report.html").write_text(
            '<!doctype html><meta charset="utf-8"><title>Indoor experiment failure</title>'
            '<h1>实验未完成</h1><pre>' + html.escape(json.dumps(summary, ensure_ascii=False, indent=2)) + '</pre>',
            encoding="utf-8",
        )
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for poses, name in ((truth, "Motion-capture GT"), (aligned, "Stereo VO, SE(3) aligned")):
        axes[0].plot(poses[:, 0, 3], poses[:, 1, 3], label=name)
    axes[0].axis("equal")
    axes[0].set(xlabel="world x [m]", ylabel="world y [m]", title="Indoor trajectory")
    axes[0].legend(fontsize=8)
    axes[1].plot(np.linalg.norm(aligned[:, :3, 3]-truth[:, :3, 3], axis=1))
    axes[1].set(xlabel="Associated frame", ylabel="position error [m]", title="Aligned position error")
    fig.tight_layout()
    fig.savefig(out / "trajectory.png", dpi=160)
    plt.close(fig)
    plots = ["trajectory.png"]
    if diagnostics:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(diagnostics["time_s"], diagnostics["visual_deg"], label="Visual vs GT")
        ax.plot(diagnostics["time_s"], diagnostics["gyro_deg"], label="Gyro vs GT")
        ax.set(xlabel="Time [s]", ylabel="interval rotation error [deg]", title="Independent rotation diagnostics")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "rotation.png", dpi=160)
        plt.close(fig)
        plots.append("rotation.png")
    if (out / "features.png").exists():
        plots.append("features.png")
    embedded = "".join(f'<figure><img alt="{name}" src="data:image/png;base64,{base64.b64encode((out/name).read_bytes()).decode()}"><figcaption>{name}</figcaption></figure>' for name in plots)
    metrics = summary["metrics"]
    rows = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in metrics.items())
    details = html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
    (out / "report.html").write_text(f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>室内定位学习与评测</title><style>
body{{font:16px/1.65 system-ui,sans-serif;color:#1e293b;background:#f5f7fa;margin:0}}main{{max-width:1050px;margin:auto;padding:36px 24px}}
h1{{font-size:30px}}.card,figure{{background:white;border:1px solid #dce3eb;border-radius:12px;padding:20px;margin:20px 0}}
img{{max-width:100%;height:auto}}table{{width:100%;border-collapse:collapse}}th,td{{padding:8px;text-align:left;border-bottom:1px solid #e2e8f0}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}}.tag{{color:#12635a}}summary{{cursor:pointer}}
</style><main><p class="tag">公开室内数据 · 离线学习实验</p><h1>双目视觉里程计与 IMU 对照评测</h1>
<p>{html.escape(summary['dataset'])} · {html.escape(summary['status'])}</p>
<div class="card"><b>实验范围</b><p>真实双目图像 → 去畸变与校正 → 特征跟踪 → 三角化 → PnP 运动估计 → 轨迹评测。
IMU 陀螺仪用于独立旋转对照，不参与位置估计。真值只进入评测，轨迹采用 SE(3) 对齐，不拟合尺度。</p>
<p>这是基础 stereo VO 与传感器评测实验，没有实现完整视觉惯性融合、回环或 OpenVINS。</p></div>
<div class="card"><h2>本次结果</h2><table>{rows}</table></div>{embedded}
<details class="card"><summary>配置、运行范围、失败信息与 IMU 对照结果</summary><pre>{details}</pre></details>
<p>结果只适用于记录的片段、采样与参数；单次片段不能证明跨场景鲁棒性。</p></main></html>""", encoding="utf-8")
