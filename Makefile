PYTHON ?= python3.13
VENV   := venv
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip

CONFIG ?= configs/ru.yaml
GPUS   ?= $(shell nvidia-smi -L 2>/dev/null | wc -l | awk '{print ($$1 > 0) ? $$1 : 1}')
JOBS   ?= $(shell $(PYTHON) -c "import os; print(max(1, (os.cpu_count() or 2) - 2))")
MODELS ?=

.PHONY: all install install-models install-moirai data forecast evaluate report experiment \
        smoke test clean clean-runs

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

forecast:
	$(VENV)/bin/torchrun --standalone --nproc_per_node=$(GPUS) -m hierfm forecast --config $(CONFIG) \
		$(if $(MODELS),--models $(MODELS))

evaluate:
	$(PY) -m hierfm evaluate --config $(CONFIG) --jobs $(JOBS) $(if $(MODELS),--models $(MODELS))

report:
	$(PY) -m hierfm report --config $(CONFIG)

experiment:
	$(MAKE) forecast evaluate report CONFIG=configs/ru.yaml
	$(MAKE) forecast evaluate report CONFIG=configs/europe.yaml

# ==========================
# ===  Checks            ===
# ==========================

smoke:
	$(PY) -m hierfm synth --config configs/smoke.yaml
	$(PY) -m hierfm forecast --config configs/smoke.yaml
	$(PY) -m hierfm evaluate --config configs/smoke.yaml --jobs $(JOBS)
	$(PY) -m hierfm report --config configs/smoke.yaml

test:
	$(PY) -m pytest -q tests

clean-runs:
	rm -rf runs

clean:
	rm -rf $(VENV) runs .pytest_cache hierfm/__pycache__ hierfm/models/__pycache__ tests/__pycache__
