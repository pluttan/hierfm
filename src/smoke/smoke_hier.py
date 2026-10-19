"""hierarchicalforecast: which reconcilers and probabilistic modes work on a toy hierarchy.

Hierarchy: total -> 2 regions -> 3 + 2 bottom nodes, hourly, base model SeasonalNaive(24)
from statsforecast (gives fitted values for the residual-based methods).
"""
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import hierarchicalforecast
import statsforecast
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import BottomUp, TopDown, MiddleOut, MinTrace, ERM, OptimalCombination
from hierarchicalforecast.utils import aggregate
from statsforecast import StatsForecast
from statsforecast.models import SeasonalNaive, AutoETS

print("hierarchicalforecast", hierarchicalforecast.__version__, "statsforecast", statsforecast.__version__)
rng = np.random.default_rng(1)
H, L = 48, 24 * 28
ts = pd.date_range("2024-01-01", periods=L, freq="h")
rows = []
for region, nodes in {"R1": ["a", "b", "c"], "R2": ["d", "e"]}.items():
    for n in nodes:
        t = np.arange(L)
        y = 50 + 10 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 2, L) + rng.normal(0, 1, L).cumsum() * 0.05
        rows.append(pd.DataFrame({"region": region, "node": n, "ds": ts, "y": y}))
df = pd.concat(rows, ignore_index=True)
Y_df, S_df, tags = aggregate(df, [["region"], ["region", "node"]])
print("S shape", S_df.shape, "levels", {k: len(v) for k, v in tags.items()})

train = Y_df.groupby("unique_id").head(L - H)
test = Y_df.groupby("unique_id").tail(H)
sf = StatsForecast(models=[SeasonalNaive(season_length=24)], freq="h", n_jobs=1, fallback_model=None)
Y_hat = sf.forecast(df=train, h=H, fitted=True, level=[80, 95])
Y_fitted = sf.forecast_fitted_values()
print("base columns", list(Y_hat.columns))

recs = {
    "BottomUp": BottomUp(),
    "TopDown_fp": TopDown(method="forecast_proportions"),
    "MiddleOut": MiddleOut(middle_level="region", top_down_method="forecast_proportions"),
    "MinTrace_ols": MinTrace(method="ols"),
    "MinTrace_wls_struct": MinTrace(method="wls_struct"),
    "MinTrace_wls_var": MinTrace(method="wls_var"),
    "MinTrace_mint_shrink": MinTrace(method="mint_shrink"),
    "MinTrace_mint_cov": MinTrace(method="mint_cov"),
    "OptimalCombination": OptimalCombination(method="ols"),
    "ERM": ERM(method="closed", lambda_reg=1e-2),
}
res = {}
for name, rec in recs.items():
    t0 = time.perf_counter()
    try:
        hrec = HierarchicalReconciliation(reconcilers=[rec])
        out = hrec.reconcile(Y_hat_df=Y_hat, Y_df=Y_fitted, S_df=S_df, tags=tags)
        col = [c for c in out.columns if c.startswith("SeasonalNaive/") and "-lo-" not in c and "-hi-" not in c][0]
        merged = out.merge(test, on=["unique_id", "ds"])
        mae = float((merged[col] - merged["y"]).abs().mean())
        res[name] = ("ok", round(mae, 3), round(time.perf_counter() - t0, 3))
    except Exception as e:
        res[name] = ("FAIL", repr(e)[:160], 0)
for k, v in res.items():
    print("POINT", k, v)

# probabilistic: intervals_method normality / bootstrap / permbu
for method in ["normality", "bootstrap", "permbu"]:
    t0 = time.perf_counter()
    try:
        hrec = HierarchicalReconciliation(reconcilers=[MinTrace(method="mint_shrink")])
        out = hrec.reconcile(Y_hat_df=Y_hat, Y_df=Y_fitted, S_df=S_df, tags=tags, level=[80, 95],
                             intervals_method=method, num_samples=200, seed=0) \
            if method != "normality" else \
            hrec.reconcile(Y_hat_df=Y_hat, Y_df=Y_fitted, S_df=S_df, tags=tags, level=[80, 95],
                           intervals_method=method)
        cols = [c for c in out.columns if "MinTrace" in c]
        print("PROB", method, "ok", round(time.perf_counter() - t0, 3), "s", cols[:6])
    except Exception as e:
        print("PROB", method, "FAIL", repr(e)[:300])
