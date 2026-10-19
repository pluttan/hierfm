"""Scores of sample-path forecasts. Samples (K, H, n), truth (H, n)."""

import numpy as np


def crps(samples, y):
    """Energy form of CRPS per (step, node): E|X - y| - E|X - X'| / 2, O(K log K) via sorting."""
    K = samples.shape[0]
    xs = np.sort(samples, 0)
    term1 = np.abs(xs - y[None]).mean(0)
    w = (2 * np.arange(1, K + 1) - K - 1)[:, None, None]
    term2 = (w * xs).sum(0) / K ** 2                       # = E|X - X'| / 2
    return term1 - term2


def mase_scale(context, season):
    """Mean absolute seasonal difference of the context, per node: context (L, n)."""
    d = np.abs(context[season:] - context[:-season]).mean(0)
    d[d == 0] = 1.0
    return d


def coverage(samples, y, level):
    a = (1 - level) / 2
    lo, hi = np.quantile(samples, [a, 1 - a], axis=0)
    return ((y >= lo) & (y <= hi)).mean(0), (hi - lo).mean(0)


def energy_score(samples, y):
    """Multivariate energy score over the node axis, averaged over steps: samples (K, H, d)."""
    K = samples.shape[0]
    t1 = np.linalg.norm(samples - y[None], axis=-1).mean(0)
    half = K // 2
    t2 = np.linalg.norm(samples[:half] - samples[half:2 * half], axis=-1).mean(0)
    return float((t1 - t2 / 2).mean())


def variogram_score(samples, y, p=0.5):
    """Variogram score of order p over nodes (Scheuerer & Hamill 2015), averaged over steps."""
    ey = np.abs(y[:, :, None] - y[:, None, :]) ** p                      # (H, d, d)
    ex = (np.abs(samples[..., :, None] - samples[..., None, :]) ** p).mean(0)
    d = ey.shape[-1]
    return float(((ey - ex) ** 2).sum((1, 2)).mean() / max(d * (d - 1), 1))


def pit(samples, y):
    return (samples <= y[None]).mean(0)
