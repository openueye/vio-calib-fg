PYTHON ?= python3
BUILD ?= build
JOBS ?= 4
CMAKE ?= cmake

.PHONY: all cpp test pytest benchmark quick benchmark-parallel docker clean

all: cpp test

cpp:
	$(CMAKE) -S . -B $(BUILD) -DCMAKE_BUILD_TYPE=Release \
		-Dpybind11_DIR=$$($(PYTHON) -m pybind11 --cmakedir)
	$(CMAKE) --build $(BUILD) -j $(JOBS)
	ctest --test-dir $(BUILD) --output-on-failure

pytest:
	PYTHONPATH=src $(PYTHON) -m pytest -q

test: pytest

benchmark:
	PYTHONPATH=src $(PYTHON) scripts/run_online_benchmark.py --jobs $(JOBS)

benchmark-parallel:
	PYTHONPATH=src $(PYTHON) scripts/run_online_benchmark.py --jobs 8

quick:
	PYTHONPATH=src $(PYTHON) scripts/run_online_benchmark.py --quick --jobs 2

docker:
	docker build -t vio-calib-fg .

clean:
	rm -rf $(BUILD) .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
