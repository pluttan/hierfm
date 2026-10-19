"""Two classic reference forecasters: seasonal naive and gradient boosting on weather and calendar.

Their quantiles come from the empirical distribution of their own errors, so they stand on the
same footing as pretrained models in the probabilistic comparison.
"""

import numpy as np

from .base import Model


class SeasonalNaive(Model):
    name = "snaive"

    def predict(self, ctx, h, levels, temp_past=None, temp_future=None, node_ids=None):
        s = int(self.params.get("season", 168))
        reps = int(np.ceil(h / s))
        point = np.tile(ctx[:, -s:], reps)[:, :h]
        # errors of the same rule over the context give the spread
        err = ctx[:, s:] - ctx[:, :-s]
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
        temp = ds.temp_fc if ds.temp_fc is not None else ds.temp
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
        rows, ys = [], []
        for o in origins:
            day_mean = ds.y[o - 24:o, i].mean()
            for k in range(h):
                t = o + k
                lag = ds.y[t - 168, i] if t - 168 < o else ds.y[t - 336, i]
                tt = temp[t, i] if temp is not None else 0.0
                rows.append((ds.index[t].hour, ds.index[t].dayofweek, tt, lag, day_mean, k))
                ys.append(ds.y[t, i])
        return np.array(rows, float), np.array(ys, float)

    def predict_at(self, origin_pos, h, levels):
        """Boosting needs absolute positions (calendar, weather), so it is called per origin."""
        ds = self.ds
        temp = ds.temp_fc if ds.temp_fc is not None else ds.temp
        q = np.empty((ds.h.n, h, len(levels)))
        for i in range(ds.h.n):
            X, _ = self._features(ds, i, [origin_pos], h, temp)
            point = self.models[i].predict(X)
            q[i] = point[:, None] + np.quantile(self.resid[i], levels)[None, :]
        return np.asarray(levels), q
