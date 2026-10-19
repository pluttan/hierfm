#!/usr/bin/env python3
"""Raw SO UPS / ATS files -> data/processed/ru in the hierfm format (see hierfm/data.py).

    python3 -u src/build_ru.py RAW_DIR OUT_DIR [--late] [--fill share|plan|none] [--no-residual] [--no-clean] [--causal]

RAW_DIR is data/raw (it holds so_ups/) or data/raw/so_ups itself. OUT_DIR gets:
    load.parquet       index ds (UTC, tz-aware, hourly), one column per bottom node, MW
    hierarchy.json     UES -> IPS -> region, format of hierfm/hierarchy.py
    observed.parquet   same shape as load, bool: True where the value is a real ATS measurement
    build_meta.json    what was built and the coverage numbers
Pandas and pyarrow are needed. The ATS tables are read with src/ats_reader.py (standard
library), so xlrd and openpyxl are not required. docs/research/06-data-russia.md explains the choices.
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ats_reader  # noqa: E402
import so_ups_hierarchy as sh  # noqa: E402

MSK = 3          # Moscow time is UTC+3 all year since 2014-10-26; ATS and SO UPS use it
IPS_KEYS = {530000: "CENTER", 550000: "SOUTH", 600000: "VOLGA", 610000: "SIBERIA",
            630000: "URAL", 840000: "NORTHWEST", 540000: "EAST"}
# SO UPS numbers regions differently from ATS in one case (Zabaykalsky Krai)
SO_TO_ATS = {76: 76200}
# From Monday 2017-11-20 to Saturday 2019-02-23 the hourly GenConsum series of SO UPS run one hour early
# against ATS and against the SO plan by subject (ForecastConsumSubRf). The day boundaries come from the
# best lag per day between the two SO sources (docs/research/06-data-russia.md, section 6.2).
SO_SHIFT = ("2017-11-20 00:00", "2019-02-23 23:00")


def so_root(raw_dir):
    raw_dir = Path(raw_dir)
    return raw_dir / "so_ups" if (raw_dir / "so_ups").exists() else raw_dir


# ------------------------------------------------------------------ readers
def read_ats(root, cache=True):
    """ATS fact_region -> wide frame: index = hour-beginning Moscow time (naive), columns = ATS code (int), MW.

    A day in the table has hours 1..24; hour h means the interval (h-1):00-h:00 of the Moscow day.
    ATS publishes working days only, so weekends and holidays are NaN after the pivot.
    """
    root = Path(root)
    pq = root / "tables" / "ats_hourly.parquet"
    files = sorted((root / "ats" / "xlsx").glob("*/*_fact_region.xls*"))
    if cache and pq.exists() and pq.stat().st_mtime >= max(f.stat().st_mtime for f in files):
        return pd.read_parquet(pq)
    rec, names, bad = [], {}, []
    for i, f in enumerate(files):
        code = int(f.name.split("_")[1])
        try:
            region, data = ats_reader.parse(f)
        except Exception as exc:  # noqa: BLE001 - keep going, report at the end
            bad.append((f.name, repr(exc)))
            continue
        names[code] = region
        rec.extend((code, d, h, v) for d, h, v in data)
        if i % 1000 == 0:
            print("ats", i, len(files), flush=True)
    if bad:
        print("unreadable ATS files:", len(bad), bad[:5], flush=True)
    df = pd.DataFrame(rec, columns=["code", "date", "hour", "mw"])
    df["ts"] = pd.to_datetime(df["date"], format="%d.%m.%Y") + pd.to_timedelta(df["hour"] - 1, unit="h")
    df = df.drop_duplicates(["code", "ts"], keep="last")
    wide = df.pivot(index="ts", columns="code", values="mw").sort_index()
    wide.columns = wide.columns.astype(int)
    wide.attrs["names"] = names
    if cache:
        (root / "tables").mkdir(exist_ok=True)
        wide.to_parquet(pq)
        (root / "tables" / "ats_names.json").write_text(
            json.dumps({str(k): v for k, v in names.items()}, ensure_ascii=False, indent=1))
    return wide


NODE = {1: "PZ1", 2: "PZ2", 8: "PZ8", **{k: f"IPS_{v}" for k, v in IPS_KEYS.items()}}


def read_genconsum(root):
    """SO UPS GenConsum -> frame index = hour-beginning Moscow time (naive), columns = (node, kind)."""
    rows = []
    for f in sorted((Path(root) / "br").glob("genconsum*_*.json")):
        for blk in json.loads(f.read_text()):
            key = blk["m_Item1"]
            if key is None:
                continue
            for r in blk["m_Item2"]:
                rows.append((NODE[key], r["M_DATE"][:10], r["INTERVAL"], r["E_USE_FACT"], r["E_USE_PLAN"]))
    df = pd.DataFrame(rows, columns=["node", "date", "hour", "fact", "plan"])
    df["ts"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"], unit="h")
    df = df.drop_duplicates(["node", "ts"], keep="last")
    fact = df.pivot(index="ts", columns="node", values="fact").sort_index()
    plan = df.pivot(index="ts", columns="node", values="plan").sort_index()
    # a zero is a missing value in these series (a power system cannot consume 0 MW)
    fact, plan = fact.where(fact > 0), plan.where(plan > 0)
    return fix_so_shift(fact), fix_so_shift(plan)


def fix_so_shift(frame):
    """Move the SO UPS values of the shifted period one hour later (label t holds the hour t+1)."""
    a, b = SO_SHIFT
    out = frame.copy()
    win = out.loc[a:b].index
    out.loc[win] = frame.shift(1, freq="h").reindex(win)
    return out


def read_plan(root):
    """SO UPS day-ahead consumption plan by subject -> wide frame (index Moscow hour-beginning, columns ATS code).

    Zone 1 row: [name, h0..h23, code]. Zone 2 (East) row: [code, name, h0..h23].
    Returns None when the files are not downloaded.
    """
    rows = []
    for zone, pat in (("Z1", "forecast_subrf_20*.json"), ("Z2", "forecast_subrf_z2_20*.json")):
        for f in sorted((Path(root) / "br").glob(pat)):
            for day, lst in json.loads(f.read_text()).items():
                for r in lst or []:
                    if not isinstance(r, list) or len(r) < 25:
                        continue
                    code, vals = (int(r[25]), r[1:25]) if zone == "Z1" else (int(r[0]), r[2:26])
                    code = SO_TO_ATS.get(code, code)
                    for h in range(24):
                        rows.append((code, f"{day} {h:02d}:00", vals[h]))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["code", "ts", "mw"])
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.drop_duplicates(["code", "ts"], keep="last")
    wide = df.pivot(index="ts", columns="code", values="mw").sort_index()
    return wide.where(wide > 0)


# ------------------------------------------------------------------ filling the days ATS does not publish
def hod_interp(frame):
    """Interpolate gaps in time inside each hour of day separately (Friday 15:00 to Monday 15:00)."""
    f = frame.to_frame() if isinstance(frame, pd.Series) else frame
    out = f.groupby(f.index.hour).transform(lambda s: s.interpolate(method="time", limit_area="inside"))
    return out.iloc[:, 0] if isinstance(frame, pd.Series) else out


def hod_ffill(frame):
    """Carry the last known value forward inside each hour of day (causal: uses only earlier days)."""
    f = frame.to_frame() if isinstance(frame, pd.Series) else frame
    out = f.groupby(f.index.hour).transform(lambda s: s.ffill())
    return out.iloc[:, 0] if isinstance(frame, pd.Series) else out


def smooth_fact(series, index, spike=0.12):
    """SO UPS fact on the grid. Isolated spikes (more than `spike` away from the 5-hour median) and the
    few missing hours (about 50 in 11 years) are replaced by linear interpolation."""
    s = series.reindex(index)
    m = s.rolling(5, center=True, min_periods=3).median()
    s = s.mask((s - m).abs() > spike * m)
    return s.interpolate(limit=6, limit_direction="both")


def fill_share(X, so_k, base=None, causal=False):
    """Fill NaN cells of the subjects of one IPS by IPS fact x ratio x share.

    ratio = sum(ATS subjects) / SO fact of the IPS; share = subject / sum(ATS subjects).
    Both are measured on days ATS publishes and interpolated in time inside each hour of day.
    The interpolation reads the next working day, so a weekend value depends on the Monday after it:
    a forecast issued on Monday 00:00 would see a trace of its own target in the context.
    causal=True carries the last working day forward instead and reads nothing from later days.
    """
    cols = list(X.columns)
    S = X.sum(axis=1, min_count=len(cols))
    spread = hod_ffill if causal else hod_interp
    rho = spread(S / so_k)
    W = spread(X.div(S, axis=0))
    W = W.div(W.sum(axis=1, min_count=len(cols)), axis=0)
    est = W.mul(so_k * rho, axis=0)
    return (X if base is None else base).fillna(est)


def fill_plan(X, plan, so_k):
    """Fill NaN cells by the SO day-ahead plan of the subject x its fact/plan ratio (interpolated in time).

    Subjects without a plan, or hours without one, fall back to fill_share.
    """
    out = X.copy()
    for c in X.columns:
        if plan is None or c not in plan.columns:
            continue
        p = plan[c].reindex(X.index).where(lambda s: s > 0)
        ratio = hod_interp(X[c] / p)
        out[c] = X[c].fillna(p * ratio)
    if out.isna().any().any():
        out = fill_share(X, so_k, base=out)
    return out


def clean_ats(X, low=0.3, high=3.0, day_hours=4, window_days=730):
    """Drop metering failures of ATS: non-positive values and hours far from the subject's own level.

    The level is the centred rolling median over `window_days`. A cell goes if it is below `low` or
    above `high` times that level. A day with at least `day_hours` such cells is dropped for the subject
    as a whole. ATS has failures that last months (Voronezh, April to November 2019).
    Returns the cleaned frame and the number of dropped cells per subject.
    """
    w = 24 * window_days
    med = X.rolling(w, center=True, min_periods=24 * 60).median()
    bad = (X <= 0) | (X < low * med) | (X > high * med)
    per_day = bad.groupby(X.index.normalize()).transform("sum")
    bad = bad | (per_day >= day_hours)
    dropped = (bad & X.notna()).sum()
    return X.mask(bad), dropped


# ------------------------------------------------------------------ hierarchy
def make_hierarchy(leaves, residual=True):
    """Hierarchy in the format of hierfm/hierarchy.py: UES -> 7 IPS -> subjects (+ one residual leaf per IPS).

    Node ids follow src/so_ups_hierarchy.py (UES, IPS_<KEY>, R<ATS code>, RES_<KEY>); the weather builder
    relies on them. A subject outside `leaves` is not a node: its load stays inside RES_<KEY>.
    """
    nodes = [{"id": "UES", "parent": None, "attrs": {"name": "ЕЭС России"}}]
    for k, name in sh.IPS.items():
        nodes.append({"id": f"IPS_{k}", "parent": "UES", "attrs": {"name": name}})
    order = list(sh.IPS)
    for code, (name, ips, tz) in sorted(sh.R.items(), key=lambda kv: (order.index(kv[1][1]), kv[0])):
        if code in leaves:
            nodes.append({"id": f"R{code}", "parent": f"IPS_{ips}",
                          "attrs": {"name": name, "ats_code": code, "tz": tz, "kind": "subject"}})
    if residual:
        for k in sh.IPS:
            nodes.append({"id": f"RES_{k}", "parent": f"IPS_{k}",
                          "attrs": {"name": f"Остаток {sh.IPS[k]}: факт СО минус сумма субъектов АТС",
                                    "tz": 3, "kind": "residual"}})
    return {"levels": ["UES", "IPS", "region"], "nodes": nodes}


# ------------------------------------------------------------------ build
def build(raw_dir, out_dir, include_late=False, residual=True, fill="share", clean=True, causal=False, log=print):
    """Raw files -> out_dir/{load.parquet, observed.parquet, hierarchy.json, build_meta.json}.

    include_late  also use the 7 subjects ATS publishes only from 2025-01 (East, Arkhangelsk, Komi);
                  the table then starts in 2025 instead of 2017.
    residual      add RES_<IPS> leaves = SO fact of the IPS minus the sum of its subjects.
    clean         drop metering failures of ATS before the fill (see clean_ats).
    fill          how to fill days without ATS data: "share" (IPS fact x ratio x share, default),
                  "plan" (SO day-ahead plan of the subject x ratio), "none" (leave NaN).
    causal        "share" carries ratio and share forward from the last working day (no look-ahead).
    """
    root, out = so_root(raw_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ats = read_ats(root)
    fact, _ = read_genconsum(root)
    plan = read_plan(root) if fill == "plan" else None
    first_valid = {c: ats[c].first_valid_index() for c in ats.columns}
    late = {c for c in ats.columns if first_valid[c] >= pd.Timestamp("2024-01-01")}
    unknown = sorted(set(ats.columns) - set(sh.R))
    if unknown:
        raise ValueError(f"ATS codes missing in so_ups_hierarchy.R: {unknown}")
    leaves = [c for c in sh.R if c in ats.columns and (include_late or c not in late)]
    t0 = max(first_valid[c] for c in leaves).normalize()
    t1 = min(ats[c].last_valid_index() for c in leaves)
    idx = pd.date_range(t0, t1, freq="h")
    X = ats.reindex(idx)[leaves]
    dropped = None
    if clean:
        X, dropped = clean_ats(X)
    else:
        X = X.where(lambda d: d > 0)                          # a zero is a metering failure
    observed = X.notna()
    so = {k: smooth_fact(fact[f"IPS_{k}"], idx) for k in sh.IPS}
    parts, obs_parts = [], []
    for k in sh.IPS:
        cols = [c for c in leaves if sh.R[c][1] == k]
        if cols:
            if fill == "share":
                Xk = fill_share(X[cols], so[k], causal=causal)
            elif fill == "plan":
                Xk = fill_plan(X[cols], plan, so[k])
            else:
                Xk = X[cols]
            parts.append(Xk)
        if residual:
            tot = Xk.sum(axis=1, min_count=len(cols)) if cols else 0.0
            res = (so[k] - tot).rename(f"RES_{k}")
            parts.append(res.to_frame())
            ok = observed[cols].all(axis=1) if cols else pd.Series(True, index=idx)
            obs_parts.append((ok & so[k].notna()).rename(f"RES_{k}").to_frame())
    load = pd.concat(parts, axis=1)
    load.columns = [f"R{c}" if isinstance(c, (int, np.integer)) else c for c in load.columns]
    obs = pd.concat([observed.rename(columns=lambda c: f"R{c}")] + obs_parts, axis=1)[load.columns]
    h = make_hierarchy(set(leaves), residual)
    bottom_ids = [n["id"] for n in h["nodes"] if n["id"].startswith("R")]   # R<code> and RES_<IPS>
    load = load[bottom_ids]
    obs = obs[bottom_ids]
    load.index = (load.index - pd.Timedelta(hours=MSK)).tz_localize("UTC")
    load.index.name = "ds"
    obs.index = load.index
    load.to_parquet(out / "load.parquet")
    obs.to_parquet(out / "observed.parquet")
    (out / "hierarchy.json").write_text(json.dumps(h, ensure_ascii=False, indent=1))
    meta = {
        "built": dt.date.today().isoformat(), "fill": fill, "include_late": include_late, "residual": residual,
        "first_utc": str(load.index[0]), "last_utc": str(load.index[-1]), "hours": len(load),
        "n_subjects": len(leaves), "n_bottom": len(bottom_ids),
        "share_observed_subject_cells": float(observed.to_numpy().mean()),
        "nan_cells_after_fill": int(load.isna().sum().sum()),
        "clean": clean, "causal": causal,
        "cells_dropped_by_clean": {f"R{c}": int(v) for c, v in dropped.items() if v > 0} if dropped is not None else {},
        "late_subjects_excluded": sorted(late) if not include_late else [],
        "note": "observed.parquet marks real ATS cells; RES_* is observed when all subjects of the IPS are",
    }
    (out / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    log("built", out, {k: meta[k] for k in ("first_utc", "last_utc", "hours", "n_bottom", "nan_cells_after_fill")})
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--late", action="store_true", help="use the 7 subjects that ATS publishes from 2025")
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--no-clean", action="store_true")
    ap.add_argument("--fill", choices=["share", "plan", "none"], default="share")
    ap.add_argument("--causal", action="store_true", help="fill weekends from earlier days only (no look-ahead)")
    a = ap.parse_args()
    build(a.raw_dir, a.out_dir, include_late=a.late, residual=not a.no_residual, fill=a.fill,
          clean=not a.no_clean, causal=a.causal)


if __name__ == "__main__":
    main()
