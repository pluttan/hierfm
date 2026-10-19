"""TiRex-2 (NX-AI, multivariate, 9 quantiles). Package tirex-2, import name tirex2."""
import numpy as np
import torch
from common import HORIZON, CONTEXT, synth, parse, timeit, report

args = parse()
data = synth()
past, truth = data[:, :CONTEXT], data[:, CONTEXT:]
from tirex2 import TimeseriesType, load_model
model = load_model(args.repo or "NX-AI/TiRex-2", device=args.device)
ts_uni = [TimeseriesType(target=torch.tensor(r)[None], past_covariates=None, future_covariates=None) for r in past]
ts_joint = TimeseriesType(target=torch.tensor(past), past_covariates=None, future_covariates=None)

out, first, med = timeit(lambda: model.forecast(ts_uni, prediction_length=HORIZON, output_type="numpy"), args.repeats)
print("univariate: list", len(out), "item", out[0].shape)
report("tirex-2 univariate", args.device, first, med, {"shape": list(out[0].shape)})
out, first, med = timeit(lambda: model.forecast([ts_joint], prediction_length=HORIZON, output_type="numpy"), args.repeats)
print("joint (4 variates in one item):", out[0].shape)
print("MAE of the median:", float(np.abs(out[0][:, 4, :] - truth).mean()))
report("tirex-2 joint", args.device, first, med, {"shape": list(out[0].shape)})
