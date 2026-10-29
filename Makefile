PYTHON ?= python3.13
VENV   := venv
PIP    := $(VENV)/bin/pip

CONFIG ?= configs/ru.yaml
SUITE  ?= configs/suites/ru.yaml

# GPU=3 (or GPU=0,1) pins the run to those cards; without it every visible card is used
GPU    ?=
ifneq ($(GPU),)
export CUDA_VISIBLE_DEVICES := $(GPU)
endif

# GPUs this job may use: the CUDA_VISIBLE_DEVICES list when a scheduler set it (nvidia-smi
# still lists every card of the node), otherwise all cards, otherwise one CPU process
comma  := ,
GPUS   ?= $(if $(CUDA_VISIBLE_DEVICES),$(words $(subst $(comma), ,$(CUDA_VISIBLE_DEVICES))),$(shell nvidia-smi -L 2>/dev/null | wc -l | awk '{print ($$1 > 0) ? $$1 : 1}'))
# CPU=0 (or 1) keeps every stage on the logical CPUs of that socket of a two-socket node;
# CPU=all lifts the limit. A Mac has one socket and no taskset, so nothing is pinned there.
CPU    ?= 0
ifeq ($(shell uname),Linux)
ifeq ($(CPU),all)
CORES  := $(shell lscpu -p=core | grep -v '^[[:punct:]]' | sort -u | wc -l)
PIN    :=
else
CPULIST := $(shell lscpu -p=cpu,socket | grep -v '^[[:punct:]]' | awk -F, '$$2 == $(CPU) {print $$1}' | paste -sd, -)
CORES  := $(shell lscpu -p=core,socket | grep -v '^[[:punct:]]' | awk -F, '$$2 == $(CPU) {print $$1}' | sort -u | wc -l)
PIN    := $(if $(CPULIST),taskset -c $(CPULIST))
endif
else
CORES  := $(shell sysctl -n hw.physicalcpu 2>/dev/null || echo 4)
PIN    :=
endif
PY     := $(PIN) $(VENV)/bin/python

# evaluation workers: one per physical core of the socket (SMT siblings add little to numpy), two left free
JOBS   ?= $(shell echo $(CORES) | awk '{print ($$1 > 3) ? $$1 - 2 : 1}')
MODELS ?=
# finished score files are kept on a rerun; FORCE=1 re-scores them
FORCE  ?=

.PHONY: all install install-torch install-models install-moirai download data forecast evaluate report run suite experiment \
        smoke smoke-gpu smoke-suite cpuinfo test clean clean-runs

# data/processed/{ru,europe} come with the clone; the download and data targets rebuild them
all: install install-models experiment

# ==========================
# ===  Environment       ===
# ==========================

install:
	$(PYTHON) -m venv $(VENV) && $(PIP) install -U pip && $(PIP) install -r requirements.txt

# torch from PyPI is built for the newest CUDA; an older cluster driver then hides the GPUs and
# every model silently runs on the CPU. The wheel is taken for the CUDA version of the driver
# (nvidia-smi); TORCH_CUDA=cu124 sets it by hand.
DRIVER_CUDA := $(shell nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | grep -o '[0-9.]*$$')
TORCH_CUDA  ?= $(shell echo "$(DRIVER_CUDA)" | awk -F. 'NF { v = $$1 * 100 + $$2; \
	print (v >= 1208) ? "cu128" : (v >= 1206) ? "cu126" : (v >= 1204) ? "cu124" : (v >= 1108) ? "cu118" : "" }')

install-torch: install
ifneq ($(TORCH_CUDA),)
	$(PIP) install --force-reinstall "torch>=2.6" --index-url https://download.pytorch.org/whl/$(TORCH_CUDA)
endif
	$(PY) -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.is_available())"

install-models: install-torch
	$(PIP) install -r requirements-models.txt

# uni2ts pins numpy 1.26 and torch < 2.5: neither exists for Python 3.13, so its own pins are skipped
install-moirai: install-models
	$(PIP) install --no-deps uni2ts && $(PIP) install -r requirements-moirai.txt
	$(PY) -c "from uni2ts.model.moirai2 import Moirai2Module; print('moirai2 import ok')"

# ==========================
# ===  Pipeline          ===
# ==========================

# raw data from open sources, no keys; every script skips finished files, so rerun after a break.
# weather: the fact comes from NASA POWER in minutes; the issued forecast from Open-Meteo
# (about 10 000 calls a day) needs about a day. On the limit the script exits with code 75,
# which is not an error here; run `make download` again the next day to continue.
download:
	$(PY) -u src/download_so_ups.py dict genconsum subrf subrf_z2 forecast_ips ats --start 2015
	$(PY) -u src/download_europe.py --out data/raw/europe
	@test -s data/meta/weather_points.csv || $(PY) -u src/build_weather_points.py
	$(PY) -u src/download_weather.py --out data/raw/weather || test $$? -eq 75

data:
	$(PY) -m hierfm.prepare all

# one card or none: a plain process; several: torchrun, one rank per card
LAUNCH = $(if $(filter 1,$(GPUS)),$(PY),$(PIN) $(VENV)/bin/torchrun --standalone --nproc_per_node=$(GPUS))

forecast:
	$(LAUNCH) -m hierfm forecast --config $(CONFIG) $(if $(MODELS),--models $(MODELS))

evaluate:
	$(PY) -m hierfm evaluate --config $(CONFIG) --jobs $(JOBS) $(if $(MODELS),--models $(MODELS)) $(if $(FORCE),--force)

report:
	$(PY) -m hierfm report --config $(CONFIG)

run: forecast evaluate report

# every case of a suite in turn, then one table of all scores (runs/<suite>/summary.md)
suite:
	@set -e; for c in $$($(PY) -m hierfm suite $(SUITE)); do \
		echo "=== $$c"; $(MAKE) --no-print-directory run CONFIG=$$c; \
	done
	$(PY) -m hierfm collect $(SUITE)

# the whole programme: Russia, Europe, the synthetic Monte Carlo study
experiment:
	$(MAKE) suite SUITE=configs/suites/ru.yaml
	$(MAKE) suite SUITE=configs/suites/europe.yaml
	$(MAKE) suite SUITE=configs/suites/synthetic.yaml

# ==========================
# ===  Checks            ===
# ==========================

# what a run will use; pick a GPU whose "CPU Affinity" in the topology lies on the chosen socket
cpuinfo:
	@echo "socket CPU=$(CPU) · physical cores $(CORES) · evaluation workers JOBS=$(JOBS)"
	@echo "pinned to: $(if $(PIN),$(PIN),nothing)"
	@echo "GPUs: $(if $(CUDA_VISIBLE_DEVICES),CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES),all visible) · forecast processes $(GPUS)"
	@echo "driver CUDA $(DRIVER_CUDA) · torch wheel $(if $(TORCH_CUDA),$(TORCH_CUDA),from PyPI)"
	@$(PY) -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.is_available())" 2>/dev/null || echo 'torch not installed'
	@nvidia-smi topo -m 2>/dev/null || true

smoke:
	$(PY) -m hierfm synth --config configs/smoke.yaml
	$(PY) -m hierfm forecast --config configs/smoke.yaml
	$(PY) -m hierfm evaluate --config configs/smoke.yaml --jobs $(JOBS)
	$(PY) -m hierfm report --config configs/smoke.yaml

# every pretrained wrapper on a GPU, one model at a time so a broken one does not stop the rest
SMOKE_GPU_MODELS := snaive chronos2 chronos2_temp chronos2_joint bolt_small timesfm25 timesfm3 timesfm3_joint moirai2
smoke-gpu:
	$(PY) -m hierfm synth --config configs/smoke.yaml
	@for m in $(SMOKE_GPU_MODELS); do \
		if $(PY) -m hierfm forecast --config configs/smoke-gpu.yaml --models $$m > runs-smoke-gpu-$$m.log 2>&1; \
		then if grep -q "] skip" runs-smoke-gpu-$$m.log; then echo "skip  $$m  (package not installed)"; \
		else echo "ok    $$m"; fi; \
		else echo "FAIL  $$m  (runs-smoke-gpu-$$m.log)"; tail -3 runs-smoke-gpu-$$m.log; fi; \
	done
	$(PY) -m hierfm evaluate --config configs/smoke-gpu.yaml --jobs $(JOBS)
	$(PY) -m hierfm report --config configs/smoke-gpu.yaml

smoke-suite:
	$(MAKE) suite SUITE=configs/suites/smoke.yaml

test:
	$(PY) -m pytest -q tests

clean-runs:
	rm -rf runs

clean:
	rm -rf $(VENV) runs .pytest_cache hierfm/__pycache__ hierfm/models/__pycache__ tests/__pycache__
