"""Common interface of a base forecaster.

Every model answers one question: given the recent history of a batch of series (and, if it
can use them, temperature history and forecast), return quantiles of the next h values.
Pretrained models return quantiles natively (Chronos-2 21 levels, TimesFM deciles); sample
paths are drawn from them later, in one place, so every model is treated the same way.
"""

import numpy as np


class Model:
    name = "base"
    uses_temp = False

    def __init__(self, device="cpu", **params):
        self.device = device
        self.params = params

    def fit(self, ds, train_end):
        """Optional: train on ds.y[:train_end]. Pretrained models skip it."""

    def predict(self, ctx, h, levels, temp_past=None, temp_future=None, node_ids=None):
        """ctx (b, L) -> (levels, q) with q of shape (b, h, len(levels)), levels ascending in (0, 1)."""
        raise NotImplementedError


def to_levels(native_levels, q, levels):
    """Re-express native quantiles on the requested levels by linear interpolation in level.

    Levels outside the native range are clamped to the outermost native quantile: we do not
    invent tails a model did not produce.
    """
    native_levels = np.asarray(native_levels, float)
    out = np.empty(q.shape[:-1] + (len(levels),), q.dtype)
    for j, lv in enumerate(levels):
        k = np.searchsorted(native_levels, lv)
        if k == 0:
            out[..., j] = q[..., 0]
        elif k >= len(native_levels):
            out[..., j] = q[..., -1]
        else:
            a, b = native_levels[k - 1], native_levels[k]
            w = (lv - a) / (b - a)
            out[..., j] = (1 - w) * q[..., k - 1] + w * q[..., k]
    return np.sort(out, -1)   # some heads cross quantiles slightly
