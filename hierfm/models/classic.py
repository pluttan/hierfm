"""Two classic reference forecasters: seasonal naive and gradient boosting on weather and calendar.

Their quantiles come from the empirical distribution of their own errors, so they stand on the
same footing as pretrained models in the probabilistic comparison.
"""

import numpy as np

from ..data import future_temp
from .base import Model


class SeasonalNaive(Model):
    name = "snaive"

    def predict(self, ctx, h, levels, temp_past=None, temp_future=None, node_ids=None):
        s = int(self.params.get("season", 168))
        L = ctx.shape[1]
        if L < s:
            s = 24                    # a cold start shorter than the season: only the daily cycle is seen
        reps = int(np.ceil(h / s))
        point = np.tile(ctx[:, -s:], reps)[:, :h]
        # errors of the same rule over the context give the spread; a context of about one season
        # (a 7-day cold start) has no such errors, then the day-ahead rule stands in for them
        es = s if L - s >= 24 else 24
        err = ctx[:, es:] - ctx[:, :-es] if L > es else np.zeros((len(ctx), 1))
        err = err[:, -int(self.params.get("err_window", 24 * 56)):]
        qe = np.quantile(err, levels, axis=1).T                     # (b, Q)
        return np.asarray(levels), point[:, :, None] + qe[:, None, :]


class Boosting(Model):
    """One histogram gradient boosting per node (scikit-learn: same algorithm family as LightGBM,
    ships its own OpenMP, so no system library on a Mac or a cluster), direct multi-horizon.

    Features: hour of day, day of week, temperature at the target hour (forecast when available),
    load one week before the target hour, mean load of the last day before the origin.
    """
    name = "gbm"
    uses_temp = True

    def fit(self, ds, train_end):
        from sklearn.ensemble import HistGradientBoostingRegressor
        self.ds, self.models, self.resid = ds, {}, {}
        h = int(self.params.get("h", 24))
        start = max(24 * 7 + 24, train_end - 24 * int(self.params.get("train_days", 730)))
        origins = np.arange(start, train_end - h, 24)
        temp = future_temp(ds, self.params.get("temp_source", "forecast"))
        def new():
            return HistGradientBoostingRegressor(max_iter=int(self.params.get("trees", 400)), learning_rate=0.05,
                                                 max_leaf_nodes=31, min_samples_leaf=50, categorical_features=[0, 1])

        # in-sample residuals are too small and give too narrow intervals: the spread comes from
        # the last fifth held out, then the model is refitted on everything
        cut = int(len(origins) * 0.8) * h
        for i in range(ds.h.n):
            X, y = self._features(ds, i, origins, h, temp)
            early = new().fit(X[:cut], y[:cut])
            self.resid[i] = y[cut:] - early.predict(X[cut:])
            self.models[i] = new().fit(X, y)

    @staticmethod
    def _features(ds, i, origins, h, temp):
        """All rows of all origins at once (numpy), one row per origin and lead hour."""
        o = np.asarray(origins, int)[:, None]
        k = np.arange(h)[None, :]
        t = o + k                                                    # (m, h) target positions
        lag = np.where(t - 168 < o, t - 168, t - 336)                # a week back, never past the origin
        y = ds.y[:, i]
        cs = np.concatenate([[0.0], np.cumsum(y)])
        day_mean = (cs[o] - cs[o - 24]) / 24.0                       # mean of the day before the origin
        hour = ds.index.hour.to_numpy()[t]
        dow = ds.index.dayofweek.to_numpy()[t]
        tt = temp[t, i] if temp is not None else np.zeros(t.shape)
        X = np.stack([hour, dow, tt, y[lag], np.broadcast_to(day_mean, t.shape), np.broadcast_to(k, t.shape)],
                     axis=-1).reshape(-1, 6).astype(float)
        return X, y[t].reshape(-1)

    def predict_at(self, origin_pos, h, levels):
        """Boosting needs absolute positions (calendar, weather), so it is called per origin."""
        return self.predict_many([origin_pos], h, levels)[0]

    def predict_many(self, positions, h, levels):
        """One model call per node for a whole batch of origins: per-origin calls spend their time
        in OpenMP start-up, not in the trees (0.5 origins/s on 64 cores before this)."""
        ds = self.ds
        temp = future_temp(ds, self.params.get("temp_source", "forecast"))
        m = len(positions)
        q = np.empty((m, ds.h.n, h, len(levels)))
        for i in range(ds.h.n):
            X, _ = self._features(ds, i, positions, h, temp)
            point = self.models[i].predict(X).reshape(m, h)
            q[:, i] = point[:, :, None] + np.quantile(self.resid[i], levels)[None, None, :]
        return [(np.asarray(levels), q[j]) for j in range(m)]
