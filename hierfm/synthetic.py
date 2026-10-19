"""A synthetic grid: 1 system, R regions, R*K subregions, hourly; the smoke test and the Monte Carlo study.

Load follows daily and weekly cycles plus a temperature response; temperature shares a
regional weather factor, so errors of neighbours are correlated the way the paper assumes.
The strength of the common and regional factors against the local one is what the Monte Carlo
suite varies: it decides how much a weather prior can know about error correlation.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from . import hierarchy as hier


def make(folder, days=730, seed=0, regions=3, per_region=3, common_sd=0.35, regional_sd=0.4, local_sd=0.3):
    rng = np.random.default_rng(seed)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    idx = pd.date_range("2024-01-01", periods=24 * days, freq="h", tz="UTC")
    T = len(idx)
    hour, dow, doy = idx.hour.to_numpy(), idx.dayofweek.to_numpy(), idx.dayofyear.to_numpy()

    recs = [{"id": "SYS", "parent": None, "attrs": {}}]
    regions = [f"R{i + 1}" for i in range(regions)]
    bottom = []
    for r_i, r in enumerate(regions):
        recs.append({"id": r, "parent": "SYS", "attrs": {"tz": 3 + 2 * r_i}})
        for j in range(per_region):
            b = f"{r}_{j + 1}"
            bottom.append(b)
            recs.append({"id": b, "parent": r, "attrs": {"tz": 3 + 2 * r_i}})

    def ar1(n, phi, sd):
        return lfilter([1.0], [1.0, -phi], rng.normal(0, sd, n))

    seasonal = -12 * np.cos(2 * np.pi * doy / 365.25) + 4 * np.sin(2 * np.pi * (hour - 9) / 24)
    common = ar1(T, 0.995, common_sd)
    regional = {r: ar1(T, 0.99, regional_sd) for r in regions}
    load, temp = {}, {}
    for b in bottom:
        r = b.split("_")[0]
        t = 5 + seasonal + common + regional[r] + ar1(T, 0.9, local_sd)
        base = rng.uniform(800, 3000)
        daily = 1 + 0.18 * np.sin(2 * np.pi * (hour - 7) / 24) + 0.06 * np.sin(4 * np.pi * (hour - 3) / 24)
        weekly = np.where(dow >= 5, 0.9, 1.0)
        heat = 0.012 * np.clip(15 - t, 0, None) + 0.008 * np.clip(t - 22, 0, None)
        load[b] = base * daily * weekly * (1 + heat) * (1 + 0.01 * rng.standard_normal(T))
        temp[b] = t

    pd.DataFrame(load, index=idx).rename_axis("ds").to_parquet(folder / "load.parquet")
    pd.DataFrame(temp, index=idx).rename_axis("ds").to_parquet(folder / "temp.parquet")
    fc = {b: v + rng.normal(0, 1.0, T) for b, v in temp.items()}      # a forecast is the truth plus noise
    pd.DataFrame(fc, index=idx).rename_axis("ds").to_parquet(folder / "temp_forecast.parquet")
    hier.save(hier.from_nodes(["system", "region", "subregion"], recs), folder / "hierarchy.json")
    print(f"synthetic: {folder} · {len(bottom)} bottom nodes · {days} days", flush=True)
