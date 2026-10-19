"""Stage 2: for every test origin draw sample paths, reconcile them by every method and score.

CPU only. Origins are independent, so a process pool takes them; one parquet per (model, horizon)
with one row per (origin, window, dependence, method, level).
"""

import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from . import covariance as cov
from . import data, dependence, metrics
from . import reconcile as rec
from .forecast import out_path

_G = {}   # read-only state of a worker, set by _init (macOS spawns workers, nothing is inherited)


def _init(state):
    _G.update(state)


def _median_index(levels):
    return int(np.argmin(np.abs(np.asarray(levels) - 0.5)))


def _window_errors(pos, h, w):
    """Errors of the base median over origins whose targets are already observed at `pos`."""
    ds, med, opos = _G["ds"], _G["med"], _G["opos"]
    sel = np.where((opos + h <= pos) & (opos >= pos - 24 * w))[0]
    if len(sel) == 0:
        return None
    return np.concatenate([ds.y[opos[j]:opos[j] + h] - med[j].T for j in sel])     # (N, n)


def _score_origin(k, cfg, model, h):
    ds, hh, rb = _G["ds"], _G["ds"].h, _G["R_bottom"]
    o = _G["test"][k]
    pos = ds.pos(o)
    f = np.load(out_path(cfg, model, h, o))
    levels, q = f["levels"].astype(float), f["q"].astype(float)
    y = ds.y[pos:pos + h]
    scale = metrics.mase_scale(ds.y[pos - cfg.context:pos], cfg.season)
    rng_base = cfg.seed * 1_000_003 + k

    hist_tpl = dependence.anomaly_template(ds.y, pos, h, cfg.samples, season=cfg.season)
    temp_tpl = None
    if ds.temp_fc is not None or ds.temp is not None:
        temp_tpl = dependence.anomaly_template(ds.y, pos, h, 30, season=cfg.season,
                                               temp_fc=ds.temp_fc if ds.temp_fc is not None else ds.temp)
    bottom_cols = [hh.nodes.index(b) for b in hh.bottom]
    levels_idx = {lv: hh.level_index(lv) for lv in range(len(hh.levels))}

    rows = []
    for w in cfg.windows:
        E = _window_errors(pos, h, w)
        if E is None or len(E) < 3:
            continue
        sd_b = E[:, bottom_cols].std(0, ddof=1) + 1e-9
        prior_R = cov.full_prior(hh, rb, sd_b)
        mats = {m: rec.method_matrix(m, hh, E, prior_R) for m in cfg.methods}
        corr_a = cov.to_corr(cov.shrunk(E, prior_R)[0])

        for di, dep in enumerate(cfg.dependence):
            rng = np.random.default_rng([rng_base, w, di])     # str hash is salted per process
            tpl = {"schaake_hist": hist_tpl, "schaake_temp": temp_tpl,
                   "schaake_err": E[: (len(E) // h) * h].reshape(-1, h, hh.n) if len(E) >= h else None}.get(dep)
            if dep.startswith("schaake") and tpl is None:
                continue
            base = dependence.draw(levels, q, cfg.samples, dep, rng, template=tpl, corr=corr_a)
            for m, (G, info) in mats.items():
                s = rec.apply(hh, G, base)
                med = np.median(s, 0)
                c = metrics.crps(s, y)
                ae = np.abs(med - y)
                for lv, idx in levels_idx.items():
                    row = {"origin": o, "window": w, "dependence": dep, "method": m, "level": lv,
                           "crps": c[:, idx].mean(), "mae": ae[:, idx].mean(),
                           "mase": (ae[:, idx] / scale[idx]).mean(),
                           "crps_scaled": (c[:, idx] / scale[idx]).mean(),
                           "lambda": info.get("lambda", np.nan), "n_err": len(E)}
                    for nominal in cfg.coverage:
                        cv, width = metrics.coverage(s[..., idx], y[:, idx], nominal)
                        row[f"cov{int(nominal * 100)}"] = cv.mean()
                        row[f"width{int(nominal * 100)}"] = (width / scale[idx]).mean()
                    if len(idx) > 1:
                        row["energy"] = metrics.energy_score(s[..., idx] / scale[idx], y[:, idx] / scale[idx])
                        row["variogram"] = metrics.variogram_score(s[..., idx] / scale[idx], y[:, idx] / scale[idx])
                    rows.append(row)
    return rows


def run(cfg, jobs=None, only_models=None):
    ds = data.load(cfg.processed)
    origins, test_start = data.origins(ds, cfg)
    hh = ds.h
    bottom_cols = [hh.nodes.index(b) for b in hh.bottom]

    # the weather prior is fixed before the test period, so it cannot peek at test data
    pre = ds.pos(test_start)
    if ds.temp is not None:
        R_bottom = cov.temp_anomaly_corr(ds.temp[max(0, pre - 24 * 365):pre, bottom_cols])
    else:
        R_bottom = cov.block_corr(hh, 0.3)
    R_bottom = np.nan_to_num(R_bottom)
    np.fill_diagonal(R_bottom, 1.0)

    jobs = jobs or max(1, (os.cpu_count() or 2) - 2)
    out_dir = Path(cfg.runs) / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in cfg.models:
        if only_models and spec.key not in only_models:
            continue
        for h in cfg.horizons:
            have = [o for o in origins if out_path(cfg, spec.key, h, o).exists()]
            if not have:
                print(f"evaluate: no forecasts for {spec.key} h{h}, skip", flush=True)
                continue
            first = np.load(out_path(cfg, spec.key, h, have[0]))
            mi = _median_index(first["levels"])
            med = np.stack([np.load(out_path(cfg, spec.key, h, o))["q"][:, :, mi] for o in have])
            test = [o for o in have if o >= test_start]
            state = dict(ds=ds, med=med, opos=np.array([ds.pos(o) for o in have]), test=test, R_bottom=R_bottom)
            print(f"evaluate: {spec.key} h{h} · {len(test)} test origins · {jobs} workers", flush=True)
            with Pool(jobs, initializer=_init, initargs=(state,)) as pool:
                parts = pool.map(partial(_score_origin, cfg=cfg, model=spec.key, h=h), range(len(test)),
                                 chunksize=4)
            df = pd.DataFrame([r for p in parts for r in p])
            df.insert(0, "model", spec.key)
            df.insert(1, "h", h)
            df.to_parquet(out_dir / f"{spec.key}_h{h}.parquet")
            print(f"evaluate: wrote {len(df)} rows", flush=True)
