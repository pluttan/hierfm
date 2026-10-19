PYTHON ?= python3.13
VENV   := venv
PY     := $(VENV)/bin/python
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
JOBS   ?= $(shell $(PYTHON) -c "import os; print(max(1, (os.cpu_count() or 2) - 2))")
MODELS ?=

.PHONY: all install install-models install-moirai data forecast evaluate report run suite experiment \
        smoke smoke-suite test clean clean-runs

all: install install-models data experiment

# ==========================
# ===  Environment       ===
# ==========================

install:
	$(PYTHON) -m venv $(VENV) && $(PIP) install -U pip && $(PIP) install -r requirements.txt

install-models: install
	$(PIP) install -r requirements-models.txt

# uni2ts pins numpy 1.26 and torch < 2.5: neither exists for Python 3.13, so its own pins are skipped
install-moirai: install-models
	$(PIP) install --no-deps uni2ts && $(PIP) install -r requirements-moirai.txt
	$(PY) -c "from uni2ts.model.moirai2 import Moirai2Module; print('moirai2 import ok')"

# ==========================
# ===  Pipeline          ===
# ==========================

data:
	$(PY) -m hierfm.prepare all

# one card or none: a plain process; several: torchrun, one rank per card
LAUNCH = $(if $(filter 1,$(GPUS)),$(PY),$(VENV)/bin/torchrun --standalone --nproc_per_node=$(GPUS))

forecast:
	$(LAUNCH) -m hierfm forecast --config $(CONFIG) $(if $(MODELS),--models $(MODELS))

evaluate:
	$(PY) -m hierfm evaluate --config $(CONFIG) --jobs $(JOBS) $(if $(MODELS),--models $(MODELS))

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

smoke:
	$(PY) -m hierfm synth --config configs/smoke.yaml
	$(PY) -m hierfm forecast --config configs/smoke.yaml
	$(PY) -m hierfm evaluate --config configs/smoke.yaml --jobs $(JOBS)
	$(PY) -m hierfm report --config configs/smoke.yaml

smoke-suite:
	$(MAKE) suite SUITE=configs/suites/smoke.yaml

test:
	$(PY) -m pytest -q tests

clean-runs:
	rm -rf runs

clean:
	rm -rf $(VENV) runs .pytest_cache hierfm/__pycache__ hierfm/models/__pycache__ tests/__pycache__
