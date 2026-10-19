"""Experiment configuration: one YAML file per run.

A file may start from another one with `base: configs/ru.yaml` and override any field;
dictionaries merge key by key, lists and scalars replace. Suites (suite.py) generate such files.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ModelSpec:
    name: str                       # registry key: chronos2, chronos_bolt, timesfm25, timesfm3, moirai2,
                                    # snaive, gbm; `ensemble` exists only at evaluation
    params: dict = field(default_factory=dict)
    label: str = ""                 # tells apart two runs of one model, e.g. chronos2_joint

    @property
    def key(self):
        return self.label or self.name


@dataclass
class Config:
    name: str
    processed: Path                 # folder with load.parquet, hierarchy.json, temp.parquet
    runs: Path                      # scores and report go here
    forecasts: Path = None          # base forecasts; several runs may share one folder
    freq: str = "h"
    season: int = 168               # weekly seasonality for hourly load, used by MASE and snaive
    horizons: list = field(default_factory=lambda: [24])
    context: int = 2048             # points of history given to a model
    history_days: int = 0           # >0: pretend every node has only this much history (cold start)
    temp_source: str = "forecast"   # temperature a model sees for the future: forecast, actual, none
    origin_hour: int = 0            # UTC hour at which a forecast is issued
    origin_step: int = 24           # hours between consecutive origins
    test_start: str = ""
    test_end: str = ""
    windows: list = field(default_factory=lambda: [7, 28, 91, 365])   # covariance windows, days
    quantiles: list = field(default_factory=lambda: [round(0.05 * i, 2) for i in range(1, 20)])
    samples: int = 200
    seed: int = 0
    models: list = field(default_factory=list)
    methods: list = field(default_factory=lambda: ["base", "bu", "ols", "wls_struct", "wls_var",
                                                   "mint_shrink", "prior_shrink"])
    dependence: list = field(default_factory=lambda: ["independent", "schaake_hist"])
    coverage: list = field(default_factory=lambda: [0.8, 0.95])
    # method A prior: kind identity | block | temp | temp_block, block rho, temp/block mix,
    # own error share of every node, shuffle=True destroys the prior on purpose (misspecification)
    prior: dict = field(default_factory=lambda: {"kind": "temp", "rho": 0.3, "mix": 0.5, "own": 0.2,
                                                 "shuffle": False})
    holidays: list = field(default_factory=list)       # "MM-DD" days scored as a separate stratum
    cold_quantile: float = 0.1                         # coldest share of test days, another stratum
    synthetic: dict = field(default_factory=dict)      # parameters of synthetic.make, for synthetic runs
    tags: dict = field(default_factory=dict)           # suite name and varied values, copied into scores

    def __post_init__(self):
        if self.forecasts is None:
            self.forecasts = Path(self.runs) / "forecasts"

    @property
    def max_window(self):
        return max(self.windows)

    @property
    def history_hours(self):
        return 24 * self.history_days if self.history_days else 10 ** 9


def merge(base, over):
    out = dict(base)
    for k, v in over.items():
        out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def read_raw(path):
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    if "base" in raw:
        b = Path(raw.pop("base"))
        raw = merge(read_raw(b if b.is_absolute() else ROOT / b), raw)
    return raw


def from_raw(raw):
    raw = dict(raw)
    for key in ("processed", "runs", "forecasts"):
        if raw.get(key):
            p = Path(raw[key])
            raw[key] = p if p.is_absolute() else ROOT / p
    defaults = Config.__dataclass_fields__["prior"].default_factory()
    raw["prior"] = merge(defaults, raw.get("prior", {}))
    raw["models"] = [ModelSpec(**m) if isinstance(m, dict) else ModelSpec(m) for m in raw.get("models", [])]
    return Config(**raw)


def load(path):
    return from_raw(read_raw(path))


def dump(cfg, path):
    d = asdict(cfg)
    for key in ("processed", "runs", "forecasts"):
        d[key] = str(d[key])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False))
