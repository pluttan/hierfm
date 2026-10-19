"""Chronos-2: quantiles, joint (multivariate) mode, covariates. No sample paths in the API."""
import sys
import numpy as np
import torch
from common import HORIZON, CONTEXT, synth, parse, timeit, report

args = parse()
from chronos import Chronos2Pipeline
data = synth()
past, truth = data[:, :CONTEXT], data[:, CONTEXT:]
repo = args.repo or "amazon/chronos-2"
pipe = Chronos2Pipeline.from_pretrained(repo, device_map=args.device)
print("quantile_levels of the model:", pipe.quantiles)

# 1. univariate: list of tensors (n_series, T) -> one task per series
def uni():
    return pipe.predict([torch.tensor(r) for r in past], prediction_length=HORIZON)
out, first, med = timeit(uni, args.repeats)
q = out[0]
print("univariate: list len", len(out), "item shape (n_variates, n_quantiles, H) =", tuple(q.shape))
report("chronos-2 univariate", args.device, first, med, {"shape": list(q.shape)})

# 2. joint: ONE task with a 2D target (n_variates, T): group attention across variates
def joint():
    return pipe.predict([{"target": past}], prediction_length=HORIZON)
out, first, med = timeit(joint, args.repeats)
print("joint: item shape", tuple(out[0].shape))
report("chronos-2 joint", args.device, first, med, {"shape": list(out[0].shape)})

# 3. predict_df with a covariate (known future)
import pandas as pd
ts = pd.date_range("2024-01-01", periods=CONTEXT + HORIZON, freq="h")
cov = np.sin(2 * np.pi * np.arange(CONTEXT + HORIZON) / 24).astype(np.float32)
ctx = pd.DataFrame({"item_id": 0, "timestamp": ts[:CONTEXT], "target": past[0], "temp": cov[:CONTEXT]})
fut = pd.DataFrame({"item_id": 0, "timestamp": ts[CONTEXT:], "temp": cov[CONTEXT:]})
df = pipe.predict_df(ctx, future_df=fut, prediction_length=HORIZON, quantile_levels=[0.1, 0.5, 0.9])
print("predict_df columns:", list(df.columns), "rows", len(df))
print("MAE of the median vs truth:", float(np.abs(df["0.5"].to_numpy() - truth[0]).mean()))
