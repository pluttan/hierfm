"""Common interface of a base forecaster.

Every model answers one question: given the recent history of a batch of series (and, if it
can use them, temperature history and forecast), return quantiles of the next h values.
Pretrained models return quantiles natively (Chronos-2 21 levels, TimesFM deciles); sample
paths are drawn from them later, in one place, so every model is treated the same way.
"""

import numpy as np
from scipy.stats import norm


class Model:
    name = "base"
    uses_temp = False
    NATIVE = None        # levels a model returns itself; None: it answers on any requested level
    # bumped when the files a model writes change meaning; forecast.py upgrades or redoes older files
    version = 1

    def __init__(self, device="cpu", **params):
        self.device = device
        self.params = params

    def fit(self, ds, train_end):
        """Optional: train on ds.y[:train_end]. Pretrained models skip it."""

    def predict(self, ctx, h, levels, temp_past=None, temp_future=None, node_ids=None):
        """ctx (b, L) -> (levels, q) with q of shape (b, h, len(levels)), levels ascending in (0, 1)."""
        raise NotImplementedError

    @classmethod
    def upgrade(cls, levels, q, old):
        """A file of an older version brought up to date, or None when only a new forecast can do it.

        Version 2 of a decile model: files of version 1 held the outer requested levels clamped to
        the outermost decile. Every decile is stored exactly on the requested grid, so the tails
        are rebuilt from the file itself, the same numbers a new forecast would give.
        """
        if old < 2 <= cls.version and cls.NATIVE is not None:
            levels = np.asarray(levels, float)
            idx = [int(np.argmin(np.abs(levels - lv))) for lv in cls.NATIVE]
            if np.allclose(levels[idx], cls.NATIVE):
                return to_levels(cls.NATIVE, q[..., idx], levels)
        return None


def to_levels(native_levels, q, levels):
    """Re-express native quantiles on the requested levels.

    Inside the native range: linear interpolation in level. Outside it: the line through the two
    outermost quantiles in normal scores, a normal tail with the local spread. Clamping instead
    (as before version 2) puts all the mass beyond the last decile on that decile, so a model with
    deciles 0.1..0.9 got a 95 % interval no wider than its 80 % one.
    """
    native_levels = np.asarray(native_levels, float)
    z = norm.ppf(native_levels)
    out = np.empty(q.shape[:-1] + (len(levels),), q.dtype)
    for j, lv in enumerate(levels):
        k = np.searchsorted(native_levels, lv)
        if k == 0:
            out[..., j] = q[..., 0] + (q[..., 1] - q[..., 0]) * (norm.ppf(lv) - z[0]) / (z[1] - z[0])
        elif k >= len(native_levels):
            out[..., j] = q[..., -1] + (q[..., -1] - q[..., -2]) * (norm.ppf(lv) - z[-1]) / (z[-1] - z[-2])
        else:
            a, b = native_levels[k - 1], native_levels[k]
            w = (lv - a) / (b - a)
            out[..., j] = (1 - w) * q[..., k - 1] + w * q[..., k]
    return np.sort(out, -1)   # some heads cross quantiles slightly
