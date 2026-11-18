"""Run the synthetic online-calibration benchmark and write tables.

Four chains (uncalibrated / offline / online / ground truth) are evaluated per
(scenario, seed).  Scenarios: clean, vibration, thermal drift, time-offset step
and frame drops.  Outputs ``benchmark.json`` and ``summary.md`` under
``results/online_calib`` by default.

Examples
--------
    python scripts/run_online_benchmark.py --quick
    python scripts/run_online_benchmark.py --jobs 8
    python scripts/run_online_benchmark.py --scenarios clean vibration --seeds 0 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
try:  # keep the Chinese caveat readable in redirected Windows consoles
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import benchmark  # noqa: E402


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    for section in ("sequence", "scenarios", "seeds", "calibration", "evaluation"):
        if section not in config:
            raise ValueError(f"missing section {section!r} in {path}")
    config["scenarios"] = {name: (params or {}) for name, params in config["scenarios"].items()}
    return config


def flattened_calibration(config: dict) -> dict:
    cfg = {**config["calibration"], **config["evaluation"]}
    required = ("estimate_bias", "window", "sigma_r", "huber_k", "chi2_level", "damping",
                "offline_max_iters", "online_max_iters", "ate_stride", "success_dt_ms",
                "success_rot_deg", "convergence_dt_ms", "convergence_rot_deg", "steady_tail")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"missing calibration/evaluation keys: {missing}")
    return cfg


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def render_summary(results: list[dict], rows: list[dict], metadata: dict) -> str:
    lines = [
        "# Online camera-IMU calibration benchmark",
        "",
        "Synthetic visual relative poses + 200 Hz gyro (plus one derived real TUM VI "
        "room1 clip); sliding-window Gauss-Newton with Huber and chi-square rejection. "
        "Medians over seeds except the single real run.",
        "",
        f"- Commit: `{metadata['commit']}`",
        f"- Config SHA-256: `{metadata['config_hash'][:16]}`",
        f"- Seeds: {metadata['seeds']}  Scenarios: {', '.join(metadata['scenarios'])}",
        f"- Duration: {metadata['duration_s']:.0f} s per run; "
        f"{metadata['n_runs']} runs, {metadata['errors']} failed to run",
        "",
        "## Calibration quality",
        "",
        "| scenario | chain | dt err [ms] | rot err [deg] | convergence [s] | steady dt err [ms] | failure rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['chain']} | {fmt(row['dt_err_ms'])} | "
            f"{fmt(row['rot_err_deg'])} | {fmt(row['convergence_s'], 2)} | "
            f"{fmt(row['steady_dt_err_ms'])} | {fmt(100 * row['failure_rate'], 1)}% |")
    lines += [
        "",
        f"Failure = |dt error| > {metadata['success_dt_ms']:g} ms or rotation error "
        f"> {metadata['success_rot_deg']:g}° (and, for the online chain, no sustained "
        "convergence). The uncalibrated chain is expected to fail.",
        "",
        "## Trajectory (pose graph with each chain's calibration)",
        "",
        "| scenario | chain | ATE [m] | 1 s RPE trans [m] | 1 s RPE rot [deg] |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['chain']} | {fmt(row['ate_rmse_m'], 4)} | "
            f"{fmt(row['rpe_t_m'], 4)} | {fmt(row['rpe_r_deg'], 4)} |")
    lines += [
        "",
        "## Scope caveat",
        "",
        "实验室离线/半在线，不是量产在线标定。 Calibration consumes precomputed visual "
        "relative rotations and gyro only; there is no accelerometer, no extrinsic "
        "translation, no real-time camera front end and no embedded deployment. "
        "The trajectory ATE/RPE come from a batch pose graph, so they characterise the "
        "calibration values, not a production estimator.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs/online_calib.yaml")
    ap.add_argument("--out", type=Path, default=ROOT / "results/online_calib")
    ap.add_argument("--scenarios", nargs="*", help="subset of configured scenarios")
    ap.add_argument("--seeds", nargs="*", type=int, help="override the configured seeds")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--quick", action="store_true",
                    help="short smoke run: 8 s, one seed, clean + time-offset step")
    ap.add_argument("--estimate-bias", action="store_true",
                    help="estimate the optional 7-state gyro bias")
    ap.add_argument("--real-asset", type=Path,
                    help="derived TUM VI clip (.npz); overrides the config path")
    args = ap.parse_args()

    config = load_config(args.config)
    cfg = flattened_calibration(config)
    if args.estimate_bias:
        cfg["estimate_bias"] = True

    sequence = dict(config["sequence"])
    scenarios = args.scenarios or list(config["scenarios"])
    unknown = [name for name in scenarios if name not in config["scenarios"]]
    if unknown:
        raise SystemExit(f"scenarios not in config: {unknown}")
    seeds = args.seeds if args.seeds is not None else list(config["seeds"])
    if args.quick:
        sequence["duration"] = min(float(sequence["duration"]), 8.0)
        cfg["ate_stride"] = max(int(cfg["ate_stride"]), 20)
        seeds = seeds[:1]
        scenarios = [name for name in ("clean", "time_offset_step") if name in scenarios]
        if "time_offset_step" in scenarios:
            step = config["scenarios"]["time_offset_step"]
            step["step_at_s"] = min(float(step.get("step_at_s", 15.0)),
                                    0.5 * float(sequence["duration"]))

    commit = git_commit()
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode("utf-8")).hexdigest()

    jobs = []
    for scenario in scenarios:
        for seed in seeds:
            jobs.append({
                "scenario": scenario,
                "seed": seed,
                "base": sequence,
                "params": config["scenarios"][scenario],
                "cfg": cfg,
                "run_id": f"{scenario}-s{seed}",
                "commit": commit,
                "config_hash": config_hash,
                "sequence": "synthetic",
            })

    real_asset = args.real_asset
    if real_asset is None and config.get("real_asset") and not args.quick:
        candidate = Path(config["real_asset"])
        real_asset = candidate if candidate.is_absolute() else ROOT / candidate
    real_used = None
    if real_asset is not None and Path(real_asset).is_file():
        real_used = str(Path(real_asset).resolve())
        real_name = Path(real_used).stem
        if real_name.endswith("_20s"):
            real_name = real_name[:-4]
        jobs.append({
            "scenario": real_name,
            "seed": 0,
            "base": sequence,
            "params": {},
            "cfg": cfg,
            "run_id": f"{real_name}-real",
            "commit": commit,
            "config_hash": config_hash,
            "sequence": f"{real_name}_stereo_vo",
            "real_asset": real_used,
        })
        scenarios = [*scenarios, real_name]
    elif real_asset is not None:
        print(f"note: real asset not found, skipping: {real_asset}")

    results, errors = [], []
    if args.jobs > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(jobs))) as pool:
            futures = {pool.submit(benchmark.run_job, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    results.append(future.result())
                    print(f"done {job['run_id']}", flush=True)
                except Exception as error:  # keep the rest of the matrix running
                    errors.append({"run_id": job["run_id"], "error": repr(error)})
                    print(f"FAILED {job['run_id']}: {error}", flush=True)
    else:
        for job in jobs:
            try:
                results.append(benchmark.run_job(job))
                print(f"done {job['run_id']}", flush=True)
            except Exception as error:
                errors.append({"run_id": job["run_id"], "error": repr(error)})
                print(f"FAILED {job['run_id']}: {error}", flush=True)

    results.sort(key=lambda r: (r["scenario"], r["seed"]))
    rows = benchmark.aggregate(results)
    metadata = {
        "commit": commit,
        "config_hash": config_hash,
        "seeds": seeds,
        "scenarios": scenarios,
        "duration_s": float(sequence["duration"]),
        "success_dt_ms": float(cfg["success_dt_ms"]),
        "success_rot_deg": float(cfg["success_rot_deg"]),
        "real_asset": real_used,
        "n_runs": len(jobs),
        "errors": len(errors),
        "environment": benchmark.environment(),
        "caveat": "实验室离线/半在线，不是量产在线标定",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "config": config, "summary": rows,
               "runs": results, "failures": errors}
    (args.out / "benchmark.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    summary = render_summary(results, rows, metadata)
    (args.out / "summary.md").write_text(summary, encoding="utf-8")

    print()
    print(summary)
    print(f"JSON:   {args.out / 'benchmark.json'}")
    print(f"Markdown: {args.out / 'summary.md'}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
