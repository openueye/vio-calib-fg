"""Run the full pipeline: simulate -> calibrate -> factor graph -> plots."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(script: str, *args: str) -> None:
    cmd = [PY, str(ROOT / "src" / script), *args]
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    run("simulate.py", "--out", "data/sim.npz")
    run("calibrate.py", "--data", "data/sim.npz", "--out", "data/calib.json")
    run("factor_graph.py", "--data", "data/sim.npz", "--calib", "data/calib.json", "--out", "results/metrics.json")
    run("plot_results.py")
