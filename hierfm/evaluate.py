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
    # one BLAS thread per worker: with a worker per core, BLAS threads on top would oversubscribe
    # the machine many times over (64 workers x 64 threads on a 64-core node)
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
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


def members(spec):
    """Forecast folders behind a model: itself, or the members of an ensemble."""
    return spec.params["members"] if spec.name == "ensemble" else [spec.key]


def load_q(cfg, spec, h, o):
    """Quantiles of one origin; an ensemble averages its members' quantiles level by level
    (Vincentisation), which keeps the result a valid quantile function."""
    parts = [np.load(out_path(cfg, m, h, o)) for m in members(spec)]
    return parts[0]["levels"].astype(float), np.mean([p["q"].astype(float) for p in parts], 0)


def _score_origin(k, cfg, spec, h):
    ds, hh, rb = _G["ds"], _G["ds"].h, _G["R_bottom"]
    o = _G["test"][k]
    pos = ds.pos(o)
    levels, q = load_q(cfg, spec, h, o)
    y = ds.y[pos:pos + h]
    scale = metrics.mase_scale(ds.y[pos - min(cfg.context, cfg.history_hours):pos], cfg.season)
    rng_base = cfg.seed * 1_000_003 + k
    days_back = min(365, cfg.history_days or 365)
    strata = {"holiday": f"{o:%m-%d}" in cfg.holidays, "cold": bool(_G["cold"][k])}

    hist_tpl = dependence.anomaly_template(ds.y, pos, h, cfg.samples, days_back=days_back, season=cfg.season)
    temp_tpl = None
    fut = data.future_temp(ds, cfg.temp_source)
    if fut is not None:
        temp_tpl = dependence.anomaly_template(ds.y, pos, h, 30, days_back=days_back, season=cfg.season,
                                               temp_fc=fut)
    bottom_cols = [hh.nodes.index(b) for b in hh.bottom]
    levels_idx = {lv: hh.level_index(lv) for lv in range(len(hh.levels))}

    rows = []
    for w in cfg.windows:
        E = _window_errors(pos, h, w)
        if E is None or len(E) < 3:
            continue
        sd_b = E[:, bottom_cols].std(0, ddof=1) + 1e-9
        prior_R = cov.full_prior(hh, rb, sd_b, cfg.prior["own"])
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
                    row = {**cfg.tags, **strata, "origin": o, "window": w, "dependence": dep, "method": m,
                           "level": lv,
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
    R_bottom = bottom_prior(cfg, ds, test_start)

    jobs = jobs or max(1, (os.cpu_count() or 2) - 2)
    out_dir = Path(cfg.runs) / "scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in cfg.models:
        if only_models and spec.key not in only_models:
            continue
        for h in cfg.horizons:
            have = [o for o in origins if all(out_path(cfg, m, h, o).exists() for m in members(spec))]
            if not have:
                print(f"evaluate: no forecasts for {spec.key} h{h}, skip", flush=True)
                continue
            mi = _median_index(load_q(cfg, spec, h, have[0])[0])
            med = np.stack([load_q(cfg, spec, h, o)[1][:, :, mi] for o in have])
            test = [o for o in have if o >= test_start]
            state = dict(ds=ds, med=med, opos=np.array([ds.pos(o) for o in have]), test=test, R_bottom=R_bottom,
                         cold=cold_days(cfg, ds, test, h))
            print(f"evaluate: {spec.key} h{h} · {len(test)} test origins · {jobs} workers", flush=True)
            with Pool(jobs, initializer=_init, initargs=(state,)) as pool:
                parts = pool.map(partial(_score_origin, cfg=cfg, spec=spec, h=h), range(len(test)),
                                 chunksize=4)
            df = pd.DataFrame([r for p in parts for r in p])
            df.insert(0, "model", spec.key)
            df.insert(1, "h", h)
            df.to_parquet(out_dir / f"{spec.key}_h{h}.parquet")
            print(f"evaluate: wrote {len(df)} rows", flush=True)


def bottom_prior(cfg, ds, test_start):
    """Bottom correlation of method A, built only from data before the test period."""
    hh, pr = ds.h, cfg.prior
    bottom_cols = [hh.nodes.index(b) for b in hh.bottom]
    pre = ds.pos(test_start)
    temp = ds.temp[max(0, pre - 24 * 365):pre, bottom_cols] if ds.temp is not None else None
    kind = pr["kind"] if temp is not None or pr["kind"] in ("identity", "block") else "block"
    if kind == "identity":
        R = np.eye(hh.m)
    elif kind == "block":
        R = cov.block_corr(hh, pr["rho"])
    elif kind == "temp":
        R = cov.temp_anomaly_corr(temp)
    else:
        R = pr["mix"] * cov.temp_anomaly_corr(temp) + (1 - pr["mix"]) * cov.block_corr(hh, pr["rho"])
    R = np.nan_to_num(R)
    if pr.get("shuffle"):
        # deliberately wrong prior: the same numbers attached to the wrong nodes
        perm = np.random.default_rng(cfg.seed + 7).permutation(hh.m)
        R = R[perm][:, perm]
    np.fill_diagonal(R, 1.0)
    return R


def cold_days(cfg, ds, test, h):
    """Flags the coldest share of test origins by mean top-level temperature over the horizon."""
    if ds.temp is None or not test:
        return np.zeros(len(test), bool)
    t = np.array([ds.temp[ds.pos(o):ds.pos(o) + h, 0].mean() for o in test])
    return t <= np.quantile(t, cfg.cold_quantile)
