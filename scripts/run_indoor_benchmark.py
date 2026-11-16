"""Reproduce two fixed, non-overlapping indoor clips and their report index."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=ROOT / "data/indoor/dataset-room1_512_16")
    args = ap.parse_args()
    rows = []
    failed = False
    for name, start in (("indoor_dev", 5), ("indoor_validation", 35)):
        out = ROOT / "results" / name
        process = subprocess.run([
            sys.executable, str(ROOT / "scripts/run_indoor.py"),
            "--dataset", str(args.dataset.resolve()), "--out", str(out),
            "--start", str(start), "--duration", "20", "--stride", "2",
        ], cwd=ROOT)
        if process.returncode not in (0, 2):
            raise SystemExit(process.returncode)
        failed |= process.returncode != 0
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        metric = summary["metrics"]
        value = lambda key: f"{metric[key]:.4f}" if metric.get(key) is not None else "不可评估"
        rows.append(f'<tr><td><a href="{name}/report.html">{name}</a></td><td>{start}–{start+20} s</td>'
                    f'<td>{summary["estimated_frames"]}/{summary["selected_frames"]}</td>'
                    f'<td>{value("ate_rmse_m")}</td><td>{value("rpe_translation_rmse_m")}</td>'
                    f'<td>{value("rpe_rotation_rmse_deg")}</td><td>{html.escape(summary["status"])}</td></tr>')
    (ROOT / "results/indoor_index.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>室内定位评测总览</title>'
        '<style>body{max-width:1150px;margin:50px auto;padding:20px;font:16px/1.7 system-ui;color:#1e293b}'
        'table{border-collapse:collapse;width:100%}td,th{padding:12px;border-bottom:1px solid #ddd;text-align:left}'
        'a{color:#087d73}</style><h1>室内定位学习与评测</h1>'
        '<p>TUM VI room1 官方录制数据。两个固定、不重叠的时间片段；参数相同。点击名称查看离线报告。</p>'
        '<table><tr><th>片段</th><th>时间范围</th><th>输出帧/选定帧</th><th>ATE [m]</th>'
        '<th>1 s RPE 平移 [m]</th><th>1 s RPE 旋转 [°]</th><th>状态</th></tr>'
        + ''.join(rows) + '</table><p>双目 VO + 独立 IMU 对照，未实现完整 VIO。'
        'SE(3) 对齐不拟合尺度；同一房间的两个片段不代表跨场景泛化。处理耗时为本机离线观测。</p></html>',
        encoding="utf-8",
    )
    print(ROOT / "results/indoor_index.html")
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
