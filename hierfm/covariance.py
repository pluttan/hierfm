"""Error covariance for reconciliation from a short window, shrunk toward a prior (method A).

Shrinkage works on the correlation scale: variances are cheap to estimate even from a week of
errors, correlations are not. The sample correlation is pulled toward a target correlation;
MinT-shrink uses the identity as the target, method A uses a prior built from the grid:

    bottom prior   R_b  from temperature-anomaly correlation (long weather history, no forecasts
                         needed) or from a block structure (same IPS)
    full prior     corr(S · D_b R_b D_b · S')  propagates it to aggregates through the hierarchy

The weight follows Schäfer & Strimmer (2005) for an arbitrary fixed target, so it falls on its
own as the window grows: with a long history method A turns into ordinary MinT.
"""

import numpy as np


def _standardise(E):
    E = E - E.mean(0)
    sd = E.std(0, ddof=1)
    sd[sd == 0] = 1.0
    return E / sd, sd


def shrink_weight(Z, target, n_eff=None):
    """Optimal weight toward a fixed correlation target, Schäfer & Strimmer (2005), eq. for target D.

    The formula assumes independent rows. Hourly errors of one forecast day are strongly
    autocorrelated, so a week of 168 rows carries the information of about 7 rows: with N rows
    the estimate of Var(r_ij) is too small by N / n_eff and the weight collapses (0.05 instead
    of about 0.7 on a 7-day window). n_eff, the number of forecast days in the window, rescales it;
    without it the function is the published estimator, kept as the MinT-shrink baseline.
    """
    N = len(Z)
    mean_w = Z.T @ Z / N                                   # mean of products z_ki z_kj
    mean_w2 = (Z ** 2).T @ (Z ** 2) / N                    # mean of their squares, no (N, n, n) tensor
    R = mean_w * N / (N - 1)
    var_r = (mean_w2 - mean_w ** 2) * N ** 2 / (N - 1) ** 3
    if n_eff:
        var_r = var_r * N / max(min(n_eff, N), 2)
    off = ~np.eye(Z.shape[1], dtype=bool)
    denom = ((R - target) ** 2)[off].sum()
    lam = var_r[off].sum() / denom if denom > 0 else 1.0
    return float(np.clip(lam, 0.0, 1.0)), R


def shrunk(E, target_corr, n_eff=None):
    """E (N, n) errors -> covariance shrunk toward target_corr, and the weight used."""
    Z, sd = _standardise(E)
    lam, R = shrink_weight(Z, target_corr, n_eff)
    C = lam * target_corr + (1 - lam) * R
    np.fill_diagonal(C, 1.0)
    return C * np.outer(sd, sd), lam


def to_corr(C):
    d = np.sqrt(np.clip(np.diag(C), 1e-12, None))
    return C / np.outer(d, d)


# ==========================
# ===  Priors            ===
# ==========================

def temp_anomaly_corr(temp_bottom, season=24):
    """Correlation of temperature anomalies between bottom nodes: shared weather, shared errors."""
    T = np.asarray(temp_bottom, float)
    T = T[~np.isnan(T).any(1)]
    anom = T[season:] - T[:-season]                        # day-to-day change removes the daily cycle
    return np.corrcoef(anom.T)


def block_corr(h, rho, level=1):
    """Bottom nodes under the same ancestor at `level` share correlation rho, others zero."""
    anc = [h.ancestor_at(b, level) for b in h.bottom]
    R = np.array([[rho if a == b else 0.0 for b in anc] for a in anc])
    np.fill_diagonal(R, 1.0)
    return R


def full_prior(h, R_bottom, bottom_sd, own=0.2):
    """Propagate a bottom correlation to every node through S, keep only the correlation.

    S D R D S' alone has rank m < n: it says an aggregate forecast errs exactly by the sum of the
    bottom errors, and as W it is singular. An aggregate forecast is made on its own, so each
    node also gets an own error share `own`.
    """
    D = np.diag(bottom_sd)
    R = to_corr(h.S @ D @ R_bottom @ D @ h.S.T)
    return (1 - own) * R + own * np.eye(h.n)


def prior_corr(h, kind, temp_bottom=None, rho=0.3, mix=0.5, bottom_sd=None, own=0.2):
    """kind: 'identity' (independent bottom), 'block', 'temp', 'temp_block' (average of the two)."""
    m = h.m
    if bottom_sd is None:
        bottom_sd = np.ones(m)
    if kind == "identity":
        Rb = np.eye(m)
    elif kind == "block":
        Rb = block_corr(h, rho)
    elif kind == "temp":
        Rb = temp_anomaly_corr(temp_bottom)
    elif kind == "temp_block":
        Rb = mix * temp_anomaly_corr(temp_bottom) + (1 - mix) * block_corr(h, rho)
    else:
        raise ValueError(kind)
    Rb = np.nan_to_num(Rb)
    np.fill_diagonal(Rb, 1.0)
    return full_prior(h, Rb, bottom_sd, own)
