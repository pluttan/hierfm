"""Scores of sample-path forecasts. Samples (K, H, n), truth (H, n)."""

import numpy as np


def crps(samples, y):
    """Fair CRPS per (step, node): E|X - y| - E|X - X'| / 2 with the K(K - 1) estimator of the
    second term (Ferro 2014); the K^2 version is biased toward smaller spread for small K.
    O(K log K) via sorting."""
    K = samples.shape[0]
    xs = np.sort(samples, 0)
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
    out = []
    for t in range(y.shape[0]):
        ey = np.abs(y[t, :, None] - y[t, None, :]) ** p
        ex = (np.abs(samples[:, t, :, None] - samples[:, t, None, :]) ** p).mean(0)
        out.append(((ey - ex) ** 2).sum())
    return float(np.mean(out) / max(d * (d - 1), 1))


def pit(samples, y):
    return (samples <= y[None]).mean(0)
