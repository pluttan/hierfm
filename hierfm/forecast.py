"""Stage 1: base forecasts of every node at every origin, one file per (model, horizon, origin).

The work is embarrassingly parallel, so each process takes every WORLD_SIZE-th task and pins
itself to one GPU. Under torchrun the rank comes from the environment; alone it is rank 0 of 1.
Finished files are skipped, so a killed run resumes where it stopped.
"""

import os
import time
from pathlib import Path

import numpy as np

from . import data, models


def rank_world():
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))


def pick_device(local_rank):
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return f"cuda:{local_rank}"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def out_path(cfg, model, h, origin):
    return Path(cfg.forecasts) / model / f"h{h}" / f"{origin:%Y%m%d%H}.npz"


def run(cfg, only_models=None, chunk=8):
    rank, world, local = rank_world()
    device = pick_device(local)
    ds = data.load(cfg.processed)
    origins, test_start = data.origins(ds, cfg)
    specs = [m for m in cfg.models if m.name != "ensemble" and (not only_models or m.key in only_models)]
    print(f"[rank {rank}/{world}] {device} · {len(origins)} origins from {origins[0]:%Y-%m-%d} · "
          f"test from {test_start:%Y-%m-%d} · context {_context(cfg)} · models {[m.key for m in specs]}",
          flush=True)

    for spec in specs:
        try:
            _run_model(cfg, ds, origins, spec, device, rank, world, chunk)
        except ImportError as exc:
            # packages are imported lazily, at the first prediction: a missing one drops this model only
            print(f"[rank {rank}] skip {spec.key}: {exc}", flush=True)


def _run_model(cfg, ds, origins, spec, device, rank, world, chunk):
    for h in cfg.horizons:
        todo = [o for o in origins if not out_path(cfg, spec.key, h, o).exists()]
        mine = todo[rank::world]
        if not mine:
            continue
        model = models.make(spec, device)
        if hasattr(model, "predict_at"):
            model.params.update(h=h, temp_source=cfg.temp_source)
            if cfg.history_days:
                model.params["train_days"] = cfg.history_days
            model.fit(ds, ds.pos(origins[0]))
        t0 = time.time()
        for i in range(0, len(mine), chunk):
            _predict_chunk(cfg, ds, model, spec.key, h, mine[i:i + chunk])
            done = min(i + chunk, len(mine))
            rate = done / max(time.time() - t0, 1e-9)
            print(f"[rank {rank}] {spec.key} h{h} {done}/{len(mine)} · {rate:.2f} origins/s", flush=True)


def _context(cfg):
    """A cold start shortens what a model may see, not only the covariance window."""
    return min(cfg.context, cfg.history_hours)


def _groups(h, level):
    """Node index groups for joint mode: everything at once, or one group per ancestor at `level`."""
    if level is None:
        return [np.arange(h.n)]
    top = [i for i, k in enumerate(h.nodes) if h.level[k] < level]
    by = {}
    for i, k in enumerate(h.nodes):
        if h.level[k] >= level:
            by.setdefault(h.ancestor_at(k, level), []).append(i)
    return ([np.array(top)] if top else []) + [np.array(v) for v in by.values()]


def _predict_chunk(cfg, ds, model, name, h, chunk):
    """Several origins share one model call: their contexts are stacked into one batch."""
    n, L = ds.h.n, _context(cfg)
    pos = [ds.pos(o) for o in chunk]
    if hasattr(model, "predict_at"):
        results = [model.predict_at(p, h, cfg.quantiles) for p in pos]
    else:
        fut = data.future_temp(ds, cfg.temp_source)
        use_temp = model.uses_temp and fut is not None
        results = []
        if model.params.get("joint", False):
            # joint mode: one call per origin and group, the nodes of a group are variates of one series
            for p in pos:
                q = None
                for g in _groups(ds.h, model.params.get("group_level")):
                    levels, qg = model.predict(ds.y[p - L:p, g].T, h, cfg.quantiles,
                                               ds.temp[p - L:p, g].T if use_temp else None,
                                               fut[p:p + h, g].T if use_temp else None,
                                               [ds.h.nodes[i] for i in g])
                    q = np.empty((n,) + qg.shape[1:]) if q is None else q
                    q[g] = qg
                results.append((levels, q))
        else:
            ctx = np.concatenate([ds.y[p - L:p].T for p in pos])                  # (k*n, L)
            tp = np.concatenate([ds.temp[p - L:p].T for p in pos]) if use_temp else None
            tf = np.concatenate([fut[p:p + h].T for p in pos]) if use_temp else None
            levels, q = model.predict(ctx, h, cfg.quantiles, tp, tf, ds.h.nodes * len(pos))
            results = [(levels, q[j * n:(j + 1) * n]) for j in range(len(pos))]

    for o, (levels, q) in zip(chunk, results):
        path = out_path(cfg, name, h, o)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, levels=np.asarray(levels, np.float32), q=np.asarray(q, np.float32))
        tmp.rename(path)                                  # a half-written file never looks finished
