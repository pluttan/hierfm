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
    return Path(cfg.runs) / "forecasts" / model / f"h{h}" / f"{origin:%Y%m%d%H}.npz"


def run(cfg, only_models=None, chunk=8):
    rank, world, local = rank_world()
    device = pick_device(local)
    ds = data.load(cfg.processed)
    origins, test_start = data.origins(ds, cfg)
    specs = [m for m in cfg.models if not only_models or m.key in only_models]
    print(f"[rank {rank}/{world}] {device} · {len(origins)} origins from {origins[0]:%Y-%m-%d} · "
          f"test from {test_start:%Y-%m-%d} · models {[m.key for m in specs]}", flush=True)

    for spec in specs:
        for h in cfg.horizons:
            todo = [o for o in origins if not out_path(cfg, spec.key, h, o).exists()]
            mine = todo[rank::world]
            if not mine:
                continue
            try:
                model = models.make(spec, device)
                if hasattr(model, "predict_at"):
                    model.params["h"] = h
                    model.fit(ds, ds.pos(origins[0]))
            except ImportError as exc:
                print(f"[rank {rank}] skip {spec.key}: {exc}", flush=True)
                break
            t0 = time.time()
            for i in range(0, len(mine), chunk):
                _predict_chunk(cfg, ds, model, spec.key, h, mine[i:i + chunk])
                done = min(i + chunk, len(mine))
                rate = done / max(time.time() - t0, 1e-9)
                print(f"[rank {rank}] {spec.key} h{h} {done}/{len(mine)} · {rate:.2f} origins/s", flush=True)


def _predict_chunk(cfg, ds, model, name, h, chunk):
    """Several origins share one model call: their contexts are stacked into one batch."""
    n, L = ds.h.n, cfg.context
    pos = [ds.pos(o) for o in chunk]
    if hasattr(model, "predict_at"):
        results = [model.predict_at(p, h, cfg.quantiles) for p in pos]
    else:
        ctx = np.concatenate([ds.y[p - L:p].T for p in pos])                  # (k*n, L)
        tp = tf = None
        if model.uses_temp and ds.temp is not None:
            tp = np.concatenate([ds.temp[p - L:p].T for p in pos])
            fut = ds.temp_fc if ds.temp_fc is not None else ds.temp
            tf = np.concatenate([fut[p:p + h].T for p in pos])
        if model.params.get("joint", False):
            # joint mode: one call per origin, the n nodes are variates of one series
            parts = [model.predict(ctx[j * n:(j + 1) * n], h, cfg.quantiles,
                                   None if tp is None else tp[j * n:(j + 1) * n],
                                   None if tf is None else tf[j * n:(j + 1) * n], ds.h.nodes)
                     for j in range(len(pos))]
            levels = parts[0][0]
            q = np.concatenate([p[1] for p in parts])
        else:
            levels, q = model.predict(ctx, h, cfg.quantiles, tp, tf, ds.h.nodes * len(pos))
        results = [(levels, q[j * n:(j + 1) * n]) for j in range(len(pos))]

    for o, (levels, q) in zip(chunk, results):
        path = out_path(cfg, name, h, o)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, levels=np.asarray(levels, np.float32), q=np.asarray(q, np.float32))
        tmp.rename(path)                                  # a half-written file never looks finished
