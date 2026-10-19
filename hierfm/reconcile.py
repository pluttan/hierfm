"""Linear projection reconciliation: y~ = S G y^ with G = (S' W^-1 S)^-1 S' W^-1.

Sample paths are reconciled one by one with the same G, which keeps a coherent joint
distribution (Panagiotelis et al., 2023). Every method differs only in W.
"""

import numpy as np

from . import covariance as cov


def projection(S, W):
    Winv_S = np.linalg.solve(W, S)                         # W^-1 S without forming W^-1
    return np.linalg.solve(S.T @ Winv_S, Winv_S.T)         # (m, n) = (S'W^-1S)^-1 S'W^-1


def bottom_up(h):
    G = np.zeros((h.m, h.n))
    G[np.arange(h.m), [h.nodes.index(b) for b in h.bottom]] = 1.0   # leaves may sit on any level
    return G


def method_matrix(method, h, E, prior_R=None, n_eff=None):
    """W (or G for bottom-up) for one method. E (N, n) are base forecast errors of the window,
    n_eff the number of forecast days behind them.

    mint_shrink is the published estimator (rows taken as independent); prior_shrink is method A
    (prior target and the n_eff correction). The two crossed variants separate the two effects:
    mint_shrink_neff (only the correction) and prior_shrink_raw (only the prior).

    Returns (G or None for 'base', info dict with the shrink weight when there is one).
    """
    S = h.S
    if method == "base":
        return None, {}
    if method == "bu":
        return bottom_up(h), {}
    if method == "ols":
        return projection(S, np.eye(h.n)), {}
    if method == "wls_struct":
        return projection(S, np.diag(S.sum(1))), {}
    var = E.var(0, ddof=1) + 1e-9
    if method == "wls_var":
        return projection(S, np.diag(var)), {}
    if method == "mint_sample":
        return projection(S, np.cov(E.T) + 1e-6 * np.diag(var)), {}
    shrink = {"mint_shrink": (np.eye(h.n), None), "mint_shrink_neff": (np.eye(h.n), n_eff),
              "prior_shrink": (prior_R, n_eff), "prior_shrink_raw": (prior_R, None)}
    if method in shrink:
        target, ne = shrink[method]
        W, lam = cov.shrunk(E, target, ne)
        return projection(S, W), {"lambda": lam}
    if method == "prior_only":
        sd = np.sqrt(var)
        return projection(S, prior_R * np.outer(sd, sd)), {"lambda": 1.0}
    raise ValueError(method)


def apply(h, G, samples):
    """samples (K, H, n) -> reconciled (K, H, n)."""
    if G is None:
        return samples
    return samples @ (h.S @ G).T
