"""Sample paths from per-node quantiles, and the joint dependence between nodes (method B).

A pretrained model forecasts each node alone, so its quantiles say nothing about which node is
high when another one is high. Drawing every node independently makes errors cancel in sums and
the upper-level interval too narrow. The fix is to order the draws of all nodes together:

    independent     draws as they come (the problem)
    schaake_hist    ranks copied from K past days of load anomalies at the same hours
                    (long history, does not need past forecasts)
    schaake_err     ranks copied from the base forecast errors of the covariance window
    schaake_temp    like schaake_hist, but the K past days are the closest by temperature forecast
    gauss_copula    draw uniforms from a Gaussian copula with the method-A correlation
"""

import numpy as np
from scipy.stats import norm


def invert(levels, q, u):
    """Quantile function by linear interpolation, linear tails from the outer two quantiles.

    levels (Q,), q (n, H, Q), u (K, n) or (K, n, H) uniforms -> samples (K, H, n).
    Vectorised: the segment [k-1, k] of every uniform is found at once; below the first level
    and above the last the outer segment is extended, which is the linear tail.
    """
    levels = np.asarray(levels, float)
    K, n = u.shape[:2]
    H, Q = q.shape[1], q.shape[2]
    k = np.clip(np.searchsorted(levels, u, side="right"), 1, Q - 1)
    w = (u - levels[k - 1]) / (levels[k] - levels[k - 1])
    if u.ndim == 2:                       # one uniform per path and node: comonotone over the horizon
        k, w = k[:, :, None], w[:, :, None]
    cell = ((np.arange(n)[:, None] * H + np.arange(H)[None, :]) * Q)[None]  # flat (i, t) offset
    flat = q.reshape(-1)
    qa, qb = flat[cell + k - 1], flat[cell + k]
    return (qa + w * (qb - qa)).transpose(0, 2, 1)


def reorder(samples, template, rng):
    """Schaake shuffle: per (step, node) give the k-th path the sample whose rank the template has."""
    K, H, n = samples.shape
    T = template
    if len(T) != K:
        T = T[rng.integers(0, len(T), K)]
    T = T + 1e-9 * rng.standard_normal(T.shape) * (np.abs(T).mean() + 1)   # break ties of repeated days
    ranks = T.argsort(0).argsort(0)
    srt = np.sort(samples, 0)
    return np.take_along_axis(srt, ranks, 0)


def anomaly_template(y, pos, H, K, days_back=365, season=168, temp_fc=None, rng=None):
    """Load anomalies (y minus last week) of past days at the same hours: (K', H, n).

    With temp_fc, the K closest past days by mean temperature forecast are taken (analogue days).
    """
    starts = np.arange(pos - 24 * days_back, pos - H + 1, 24)
    starts = starts[starts - season >= 0]
    tpl = np.stack([y[s:s + H] - y[s - season:s - season + H] for s in starts])
    if temp_fc is not None:
        today = temp_fc[pos:pos + H].mean(0)
        dist = np.array([np.abs(temp_fc[s:s + H].mean(0) - today).mean() for s in starts])
        tpl = tpl[np.argsort(dist)[:K]]
    return tpl


def draw(levels, q, K, kind, rng, template=None, corr=None):
    """levels/q from a model -> (K, H, n) sample paths with the requested cross-node dependence."""
    n = q.shape[0]
    if kind == "gauss_copula":
        L = np.linalg.cholesky(corr + 1e-9 * np.eye(n))
        u = norm.cdf(rng.standard_normal((K, n)) @ L.T)
        return invert(levels, q, u)
    u = rng.random((K, n))
    s = invert(levels, q, u)
    if kind == "independent":
        return s
    return reorder(s, template, rng)
