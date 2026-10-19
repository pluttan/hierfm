"""Shared helpers of the smoke tests: synthetic hourly series, timing, device choice.

Run every script as:  OMP_NUM_THREADS=4 nice -n 10 <venv>/bin/python -u script.py [--device cpu|mps]
"""

import argparse
import json
import time

import numpy as np

HORIZON = 48
CONTEXT = 1024          # about six weeks of hourly data
N_SERIES = 4            # variates of the multivariate / joint tests


def synth(n_series=N_SERIES, length=CONTEXT + HORIZON, seed=0):
    """Hourly load-like series: daily and weekly cycles, shared noise, own noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    common = rng.normal(0, 1, length).cumsum() * 0.02
    rows = []
    for i in range(n_series):
        base = 100 * (i + 1)
        daily = 0.20 * np.sin(2 * np.pi * (t - 6 - i) / 24)
        weekly = 0.08 * np.sin(2 * np.pi * t / 168)
        noise = rng.normal(0, 0.02, length)
        rows.append(base * (1 + daily + weekly + common + noise))
    return np.stack(rows).astype(np.float32)


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--repo", default=None, help="HF id or local directory with weights")
    return ap.parse_args()


def timeit(fn, repeats=3):
    """First call (includes compile and warm-up) and the median of the next calls, seconds."""
    t0 = time.perf_counter()
    out = fn()
    first = time.perf_counter() - t0
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        times.append(time.perf_counter() - t0)
    return out, first, float(np.median(times))


def report(name, device, first, median, extra):
    row = {"model": name, "device": device, "first_s": round(first, 3), "median_s": round(median, 3), **extra}
    print("RESULT " + json.dumps(row, ensure_ascii=False), flush=True)
