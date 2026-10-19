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
    """
    levels = np.asarray(levels)
    K, n = u.shape[:2]
    H = q.shape[1]
    if u.ndim == 2:                       # one uniform per path and node: comonotone over the horizon
        u = np.repeat(u[:, :, None], H, 2)
    out = np.empty((K, H, n))
    lo_slope = (q[..., 1] - q[..., 0]) / (levels[1] - levels[0])
    hi_slope = (q[..., -1] - q[..., -2]) / (levels[-1] - levels[-2])
    for i in range(n):
        for t in range(H):
            x = u[:, i, t]
            v = np.interp(x, levels, q[i, t])
            low, high = x < levels[0], x > levels[-1]
            v[low] = q[i, t, 0] - lo_slope[i, t] * (levels[0] - x[low])
            v[high] = q[i, t, -1] + hi_slope[i, t] * (x[high] - levels[-1])
            out[:, t, i] = v
    return out


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
