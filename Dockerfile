# Reproducible Linux image for the Python reference, pytest and the C++ core.
# OpenCV is installed for the optional calib_vision module; PCL is deliberately
# not installed because no PCL algorithm is part of the calibration core.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake pkg-config \
        libeigen3-dev libopencv-dev \
        python3 python3-pip python3-venv \
        rsync git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .
RUN cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
        -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) \
    && cmake --build build -j "$(nproc)" \
    && ctest --test-dir build --output-on-failure

ENV PYTHONPATH=/workspace/src
CMD ["bash", "-lc", "python3 -m pytest -q && python3 scripts/run_online_benchmark.py --jobs 4"]
