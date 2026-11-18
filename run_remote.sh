#!/usr/bin/env bash
# Reproducible remote run for project A (vio-calib-fg).
#
#   bash run_remote.sh
#
# Environment overrides:
#   REMOTE=stuttgart      ssh alias
#   JOBS=8                benchmark worker processes (24 cores on the host)
#   SKIP_PIP=1            reuse an existing venv
#
# The script only touches ~/vio-calib and ~/vio-calib-env on the remote host.
# It does not use or disturb the shared tmux session `api`.
set -euo pipefail

REMOTE=${REMOTE:-stuttgart}
JOBS=${JOBS:-8}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

echo "== sync repository to ${REMOTE}:~/vio-calib =="
rsync -az --delete \
  --exclude '.git' --exclude 'data' --exclude 'results' \
  --exclude 'build' --exclude '__pycache__' --exclude '.pytest_cache' \
  "$ROOT/" "${REMOTE}:~/vio-calib/"

echo "== remote install, C++ build, tests and benchmark =="
ssh -o ConnectTimeout=20 "$REMOTE" "JOBS=${JOBS} bash -s" <<'REMOTE_SCRIPT' | tee "$ROOT/remote_run.log"
set -euo pipefail
cd ~/vio-calib

if [ ! -x ~/vio-calib-env/bin/python ]; then
  if ! python3 -m venv ~/vio-calib-env 2>/dev/null; then
    # Some Ubuntu images omit python3-venv; virtualenv from pip is a fallback.
    python3 -m pip install --user -q virtualenv
    python3 -m virtualenv -q ~/vio-calib-env
  fi
fi
# shellcheck disable=SC1090
source ~/vio-calib-env/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build -j "$(nproc)"
ctest --test-dir build --output-on-failure

PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/run_online_benchmark.py --jobs "$JOBS"
echo "== remote run complete =="
REMOTE_SCRIPT

echo "== collect benchmark results =="
mkdir -p "$ROOT/results/online_calib"
rsync -az "${REMOTE}:~/vio-calib/results/online_calib/" "$ROOT/results/online_calib/"

echo
echo "Results: $ROOT/results/online_calib/benchmark.json"
echo "         $ROOT/results/online_calib/summary.md"
echo "Log:     $ROOT/remote_run.log"
