"""Suites: one YAML expands into many runs, then their scores are collected into one table.

    name: ru
    base: configs/ru.yaml                      # every experiment starts from it
    experiments:
      main: {}                                 # the base as is, tag "default"
      context:
        set:  {horizons: [24]}                 # overrides shared by all cases
        grid: {context: [256, 1024, 4096]}     # cartesian product, dotted keys reach into dicts
      cold_start:
        cases:                                 # explicit cases when values must go together
          - {history_days: 7, windows: [7]}
          - {history_days: 28, windows: [7, 28]}
      prior:
        share_forecasts: main                  # evaluation-only experiment: reuse main forecasts
        grid: {prior.kind: [identity, block]}

Each case gets runs/<suite>/<experiment>/<tag>/ with its config.yaml, scores and report.
"""

import itertools
import sys
from pathlib import Path

import pandas as pd
import yaml

from . import config
from .config import ROOT
from .report import table


def _nest(flat):
    out = {}
    for k, v in flat.items():
        cur = out
        parts = k.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = v
    return out


def _tag(case):
    if not case:
        return "default"
    return ",".join(f"{k}={'-'.join(map(str, v)) if isinstance(v, list) else v}" for k, v in sorted(case.items()))


def _cases(exp):
    if "cases" in exp:
        return exp["cases"]
    grid = exp.get("grid", {})
    keys = list(grid)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(grid[k] for k in keys))] or [{}]


def expand(path):
    """Write one config per case and return their paths in run order (shared forecasts first)."""
    suite = yaml.safe_load(Path(path).read_text())
    name = suite["name"]
    base = config.read_raw(ROOT / suite["base"])
    root = ROOT / "runs" / name
    first_run, first_data = {}, {}
    out = []
    for exp_name, exp in suite["experiments"].items():
        exp = exp or {}
        start = config.merge(base, _nest(exp.get("set", {})))
        if "base" in exp:
            start = config.merge(config.read_raw(ROOT / exp["base"]), _nest(exp.get("set", {})))
        for case in _cases(exp):
            raw = config.merge(start, _nest(case))
            tag = _tag(case)
            run_dir = root / exp_name / tag
            raw["runs"] = str(run_dir)
            raw["forecasts"] = str(first_run[exp["share_forecasts"]] / "forecasts") if exp.get("share_forecasts") \
                else str(run_dir / "forecasts")
            if exp.get("share_forecasts"):
                raw["processed"] = first_data[exp["share_forecasts"]]     # same data as the forecasts
            elif raw.get("synthetic"):
                raw["processed"] = str(ROOT / "data" / "processed" / "synthetic" / name / exp_name / tag)
            raw["tags"] = {"suite": name, "experiment": exp_name, "case": tag,
                           **{k.replace(".", "_"): (str(v) if isinstance(v, list) else v) for k, v in case.items()}}
            cfg = config.from_raw(raw)
            config.dump(cfg, run_dir / "config.yaml")
            first_run.setdefault(exp_name, run_dir)
            first_data.setdefault(exp_name, raw["processed"])
            out.append(run_dir / "config.yaml")
    return out


def collect(path):
    """All scores of a suite into one parquet, plus a per-experiment summary of the key numbers."""
    name = yaml.safe_load(Path(path).read_text())["name"]
    root = ROOT / "runs" / name
    files = sorted(root.glob("*/*/scores/*.parquet"))
    if not files:
        sys.exit(f"collect: no scores under {root}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df.to_parquet(root / "all_scores.parquet")
    cov_cols = [c for c in df.columns if c.startswith("cov")]
    lines = [f"# Suite {name}: {len(files)} score files, {len(df)} rows\n"]
    for exp, g in df.groupby("experiment"):
        t = g.groupby(["case", "model", "h", "level", "dependence", "method"])[["mase", "crps_scaled"] + cov_cols].mean()
        t.to_csv(root / f"summary_{exp}.csv")
        top = t.xs(0, level="level").sort_values("crps_scaled").head(15)
        lines.append(f"## {exp}\n\nbest 15 at the top level by CRPS:\n\n{table(top)}\n")
    (root / "summary.md").write_text("\n".join(lines))
    print(f"collect: {root / 'summary.md'}", flush=True)


if __name__ == "__main__":
    for p in expand(sys.argv[1]):
        print(p)
