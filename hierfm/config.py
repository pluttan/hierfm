"""Experiment configuration: one YAML file per dataset, everything else derives from it."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ModelSpec:
    name: str                       # registry key: chronos2, timesfm25, timesfm3, moirai2, snaive, gbm
    params: dict = field(default_factory=dict)
    label: str = ""                 # tells apart two runs of one model, e.g. chronos2_joint

    @property
    def key(self):
        return self.label or self.name


@dataclass
class Config:
    name: str
    processed: Path                 # folder with load.parquet, hierarchy.json, temp.parquet
    runs: Path                      # forecasts and scores go here
    freq: str = "h"
    season: int = 168               # weekly seasonality for hourly load, used by MASE and snaive
    horizons: list = field(default_factory=lambda: [24])
    context: int = 2048             # points of history given to a model
    origin_hour: int = 0            # local hour at which a forecast is issued
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

    @property
    def max_window(self):
        return max(self.windows)


def load(path):
    raw = yaml.safe_load(Path(path).read_text())
    for key in ("processed", "runs"):
        p = Path(raw[key])
        raw[key] = p if p.is_absolute() else ROOT / p
    raw["models"] = [ModelSpec(**m) if isinstance(m, dict) else ModelSpec(m) for m in raw.get("models", [])]
    return Config(**raw)
