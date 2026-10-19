"""Significance of score differences: Diebold–Mariano with the Harvey–Leybourne–Newbold correction,
and multiple comparisons with the best (MCB) through Nemenyi critical distances."""

import numpy as np
from scipy import stats


def diebold_mariano(loss_a, loss_b, h=1):
    """Two-sided test that two loss series have equal mean. Returns (statistic, p-value)."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    d = d[~np.isnan(d)]
    T = len(d)
    if T < 3:
        return np.nan, np.nan
    dc = d - d.mean()
    gamma = [dc @ dc / T] + [dc[k:] @ dc[:-k] / T for k in range(1, h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / T
    if var <= 0:
        return np.nan, np.nan
    dm = d.mean() / np.sqrt(var)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = dm * hln
    return float(stat), float(2 * stats.t.sf(abs(stat), T - 1))


def mcb(losses, alpha=0.05):
    """losses: DataFrame, rows = cases (origins), columns = methods. Lower is better.

    Returns mean ranks, the critical distance and which methods are not worse than the best.
    """
    clean = losses.dropna()
    ranks = clean.rank(axis=1).mean()
    k, N = clean.shape[1], clean.shape[0]
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6 * N))
    best = ranks.min()
    return ranks.sort_values(), cd, ranks[ranks <= best + cd].index.tolist()
