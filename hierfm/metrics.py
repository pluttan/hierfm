"""Scores of sample-path forecasts. Samples (K, H, n), truth (H, n)."""

import numpy as np


def crps(samples, y):
    """Fair CRPS per (step, node): E|X - y| - E|X - X'| / 2 with the K(K - 1) estimator of the
    second term (Ferro 2014); the K^2 version is biased toward smaller spread for small K.
    O(K log K) via sorting."""
    return crps_sorted(np.sort(samples, 0), y)


def crps_sorted(xs, y):
    """crps() on samples already sorted along the path axis: the evaluation sorts once per method
    and takes the median and the interval ends from the same array."""
    K = xs.shape[0]
    term1 = np.abs(xs - y[None]).mean(0)
    w = (2 * np.arange(1, K + 1) - K - 1)[:, None, None]
    term2 = (w * xs).sum(0) / (K * (K - 1))                # = E|X - X'| / 2 over distinct pairs
    return term1 - term2


def mase_scale(context, season):
    """Mean absolute seasonal difference of the context, per node: context (L, n)."""
    d = np.abs(context[season:] - context[:-season]).mean(0)
    d[d == 0] = 1.0
    return d


def coverage_cells(samples, y, level):
    """Hit (truth inside the central interval) and interval width per (step, node)."""
    a = (1 - level) / 2
    lo, hi = np.quantile(samples, [a, 1 - a], axis=0)
    return ((y >= lo) & (y <= hi)).astype(float), hi - lo


def sorted_quantile(xs, a):
    """np.quantile(samples, a, axis=0) (linear method) from samples sorted along axis 0."""
    pos = a * (xs.shape[0] - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, xs.shape[0] - 1)
    return xs[lo] + (pos - lo) * (xs[hi] - xs[lo])


def coverage(samples, y, level):
    hit, width = coverage_cells(samples, y, level)
    return hit.mean(0), width.mean(0)


def energy_score(samples, y):
    """Multivariate energy score over the node axis, averaged over steps: samples (K, H, d)."""
    K = samples.shape[0]
    t1 = np.linalg.norm(samples - y[None], axis=-1).mean(0)
    half = K // 2
    t2 = np.linalg.norm(samples[:half] - samples[half:2 * half], axis=-1).mean(0)
    return float((t1 - t2 / 2).mean())


def variogram_score(samples, y, p=0.5):
    """Variogram score of order p over nodes (Scheuerer & Hamill 2015), averaged over steps.
    One step at a time: (K, d, d) for 80 nodes and 200 paths is 10 MB, (K, H, d, d) would be 250."""
    d = y.shape[-1]
    # p = 0.5 is a square root: np.sqrt is several times faster than the general power
    power = np.sqrt if p == 0.5 else (lambda a: a ** p)
    out = []
    for t in range(y.shape[0]):
        ey = power(np.abs(y[t, :, None] - y[t, None, :]))
        ex = power(np.abs(samples[:, t, :, None] - samples[:, t, None, :])).mean(0)
        out.append(((ey - ex) ** 2).sum())
    return float(np.mean(out) / max(d * (d - 1), 1))


def pit(samples, y):
    return (samples <= y[None]).mean(0)
