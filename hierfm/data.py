"""Processed dataset: bottom-level load, its hierarchy and temperature, on one hourly UTC grid.

A processed folder holds:
    load.parquet           index ds (UTC), one column per bottom node, MW
    hierarchy.json         see hierarchy.py
    temp.parquet           index ds, one column per node that has its own weather (optional)
    temp_forecast.parquet  same layout, day-ahead temperature forecast issued before each day (optional)
    observed.parquet       index ds, bottom columns, True where the value was measured rather than
                           filled (ATS skips weekends and holidays); missing file = all measured

Aggregates are always rebuilt as sums of the bottom, so the truth is coherent by construction;
the published upper-level series (with their own losses and metering) are kept by prepare
scripts only for a sanity check.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import hierarchy as hier


@dataclass
class Dataset:
    h: hier.Hierarchy
    index: pd.DatetimeIndex
    y: np.ndarray                  # (T, n) load of every node, top-down order
    temp: np.ndarray | None        # (T, n) temperature of every node or None
    temp_fc: np.ndarray | None     # (T, n) temperature forecast or None
    obs: np.ndarray = None         # (T, n) True where the node value is measured; aggregates always are

    def pos(self, ts):
        return self.index.get_indexer([pd.Timestamp(ts)])[0]


def _node_frame(df, h, weights):
    """Fill nodes without their own column by the load-weighted mean of their children."""
    out = pd.DataFrame(index=df.index)
    for k in reversed(h.nodes):     # bottom first, so children exist when a parent needs them
        if k in df.columns:
            out[k] = df[k]
            continue
        kids = [c for c in h.nodes if h.parent[c] == k]
        w = np.array([weights.get(c, 1.0) for c in kids])
        out[k] = (out[kids].to_numpy() * w).sum(1) / w.sum()
    return out[h.nodes].to_numpy()


def load(folder):
    folder = Path(folder)
    h = hier.load(folder / "hierarchy.json")
    bottom = pd.read_parquet(folder / "load.parquet")[h.bottom]
    bottom = bottom.sort_index().asfreq("h")
    gaps = int(bottom.isna().sum().sum())
    if gaps:
        # short metering gaps only; long ones should be cut out by the prepare script
        bottom = bottom.interpolate(limit=6, limit_direction="both")
    y = h.aggregate(bottom.to_numpy())

    obs = np.ones_like(y, dtype=bool)
    if (folder / "observed.parquet").exists():
        # aggregates come from the system operator's own metering, only the bottom has filled cells
        o = pd.read_parquet(folder / "observed.parquet").reindex(bottom.index)[h.bottom].fillna(False)
        cols = [h.nodes.index(b) for b in h.bottom]
        obs[:, cols] = o.to_numpy(bool)

    mean_load = dict(zip(h.nodes, np.nanmean(y, 0)))
    temp = temp_fc = None
    if (folder / "temp.parquet").exists():
        t = pd.read_parquet(folder / "temp.parquet").reindex(bottom.index).interpolate(limit=6)
        temp = _node_frame(t, h, mean_load)
    if (folder / "temp_forecast.parquet").exists():
        t = pd.read_parquet(folder / "temp_forecast.parquet").reindex(bottom.index)
        temp_fc = _node_frame(t, h, mean_load)
    return Dataset(h, bottom.index, y, temp, temp_fc, obs)


def future_temp(ds, source):
    """Temperature a model may use for the forecast hours: issued forecast, the truth (an upper
    bound on what weather can give), or nothing."""
    if source == "none" or ds.temp is None:
        return None
    if source == "actual":
        return ds.temp
    if ds.temp_fc is None:
        # the truth in place of a missing forecast would leak the future into the inputs
        if not getattr(ds, "_warned_fc", False):
            print("data: no temperature forecast in this dataset; models run without future weather", flush=True)
            ds._warned_fc = True
        return None
    return ds.temp_fc


def origins(ds, cfg):
    """Forecast origins inside the test period, plus enough earlier ones to fill the longest window."""
    idx = ds.index
    start = pd.Timestamp(cfg.test_start, tz=idx.tz) if cfg.test_start else idx[cfg.context + 24 * cfg.max_window]
    end = pd.Timestamp(cfg.test_end, tz=idx.tz) if cfg.test_end else idx[-max(cfg.horizons) - 1]
    first = start - pd.Timedelta(days=cfg.max_window)
    cand = idx[(idx >= first) & (idx <= end) & (idx.hour == cfg.origin_hour)]
    cand = cand[:: max(1, cfg.origin_step // 24)]
    keep = [t for t in cand if ds.pos(t) >= cfg.context]
    return pd.DatetimeIndex(keep), start
