"""Raw downloads -> data/processed/<name>/ (see data.py for the layout). `make data`.

    python -m hierfm.prepare all | ru | europe

The source-specific parsing lives in src/build_*.py (written with the initial research,
docs/research/06-08); this module fixes the choices the experiments rely on:

    ru      ATS subjects + RES_<IPS> residuals under 7 IPS and the UES, 2017..2026; weekends and
            holidays (ATS does not publish them) filled from earlier days only (causal), the
            measured cells marked in observed.parquet so that scores skip the fill
    europe  France 12 regions, Germany 4 TSO zones, Denmark DK1-DK2 (22 nodes); temperature
            from the same weather points as ru, not from the Historical Forecast API: that one
            is stitched from the first hours of every run and would flatter the weather models

Temperature is written only when it covers the last two years of the load; otherwise the
dataset goes without it and the runs say so. Every step prints what it wrote.
"""

import importlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from .config import ROOT

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SRC = ROOT / "src"
BUILDERS = {}
MIN_TEMP_COVERAGE = 0.95


def builder(name):
    def wrap(fn):
        BUILDERS[name] = fn
        return fn
    return wrap


def _src(name):
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    return importlib.import_module(name)


# ==========================
# ===  Shared steps      ===
# ==========================

def collapse_single_children(out):
    """A node with one leaf child duplicates it (IPS_EAST over RES_EAST): S gets two equal rows and
    the error covariance becomes singular. The child is dropped and the parent takes its series."""
    hp = out / "hierarchy.json"
    h = json.loads(hp.read_text())
    kids = {}
    for n in h["nodes"]:
        if n.get("parent"):
            kids.setdefault(n["parent"], []).append(n["id"])
    renames = {}
    for parent, ch in kids.items():
        if len(ch) == 1 and ch[0] not in kids:
            renames[ch[0]] = parent
    if not renames:
        return
    h["nodes"] = [n for n in h["nodes"] if n["id"] not in renames]
    hp.write_text(json.dumps(h, ensure_ascii=False, indent=1))
    for f in ("load.parquet", "observed.parquet", "temp.parquet", "temp_forecast.parquet"):
        p = out / f
        if p.exists():
            df = pd.read_parquet(p)
            for child, parent in renames.items():
                if child in df and parent in df:
                    df = df.drop(columns=[child])
            df.rename(columns=renames).to_parquet(p)
    print(f"prepare: collapsed single-child nodes {renames}", flush=True)


def _aggregate_weather(wdir):
    """Node series from city points (download_weather.py --aggregate), for every source present."""
    for src in ("power", "archive", "prev_ecmwf_ifs025"):
        if (wdir / src).is_dir():
            subprocess.run([sys.executable, "-u", str(SRC / "download_weather.py"), "--out", str(wdir),
                            "--aggregate", src], check=True)


def _coverage(df, index):
    span = index[index >= index[-1] - pd.Timedelta(days=730)]
    part = df.reindex(span)
    return float(part.notna().to_numpy().mean()) if part.shape[1] else 0.0


def weather(out, kind, wdir):
    """temp.parquet (fact) and temp_forecast.parquet (issued forecast) when they cover enough."""
    bw = _src("build_weather")
    _aggregate_weather(wdir)
    ids = {n["id"] for n in json.loads((out / "hierarchy.json").read_text())["nodes"]}
    index = pd.read_parquet(out / "load.parquet").index
    fact = None
    for fname, target in ((bw.fact_file(wdir), "temp.parquet"), (bw.FORECAST_FILE, "temp_forecast.parquet")):
        p = wdir / fname
        if not p.exists():
            print(f"prepare: {target} skipped, {p.name} not aggregated yet", flush=True)
            continue
        df = bw.build_frame(p, kind, ids)
        df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index
        if target == "temp.parquet":
            fact = df
        elif fact is not None:
            # issued forecasts start 2024-10 (download_weather.PREV_FROM); before that the fact
            # stands in, which only touches training history and analogue days, never a test day
            first = df.dropna(how="all").index.min()
            hist = fact.reindex(fact.index[fact.index < first]).reindex(columns=df.columns)
            df = pd.concat([hist, df[df.index >= first]])
            print(f"prepare: temp forecast before {first:%Y-%m-%d} taken from the fact", flush=True)
        cov = _coverage(df, index)
        if cov < MIN_TEMP_COVERAGE:
            print(f"prepare: {target} skipped, covers {cov:.0%} of the last two years "
                  f"(need {MIN_TEMP_COVERAGE:.0%}); `make download` continues the weather", flush=True)
            (out / target).unlink(missing_ok=True)
            continue
        df.to_parquet(out / target)
        print(f"prepare: {target} written, coverage {cov:.1%}", flush=True)


# ==========================
# ===  Datasets          ===
# ==========================

@builder("ru")
def build_ru(raw, out):
    _src("build_ru").build(raw, out, causal=True)
    collapse_single_children(out)
    weather(out, "ru", raw / "weather")


@builder("europe")
def build_europe(raw, out):
    _src("build_europe").build(raw / "europe", out)
    collapse_single_children(out)
    # the same sources as ru: NASA POWER fact and issued IFS forecasts replace the Historical
    # Forecast API series that build_europe writes
    weather(out, "eu", raw / "weather")


def check(out):
    """Load through hierfm the way the experiments will, and print the shape of the result."""
    from . import data
    ds = data.load(out)
    measured = ds.obs[:, [ds.h.nodes.index(b) for b in ds.h.bottom]].mean()
    print(f"prepare: {out.name}: {ds.h.n} nodes ({ds.h.m} bottom, levels {ds.h.levels}), "
          f"{len(ds.index)} h from {ds.index[0]} to {ds.index[-1]}, bottom measured {measured:.1%}, "
          f"NaN {int(np.isnan(ds.y).sum())}, temp {'yes' if ds.temp is not None else 'no'}, "
          f"temp forecast {'yes' if ds.temp_fc is not None else 'no'}", flush=True)


def main(argv):
    names = list(BUILDERS) if not argv or argv[0] == "all" else argv
    missing = [n for n in names if n not in BUILDERS]
    if missing:
        sys.exit(f"prepare: unknown dataset {missing}; known {sorted(BUILDERS)}")
    for n in names:
        out = PROCESSED / n
        out.mkdir(parents=True, exist_ok=True)
        BUILDERS[n](RAW, out)
        check(out)


if __name__ == "__main__":
    main(sys.argv[1:])
