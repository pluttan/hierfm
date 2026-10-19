"""Toto-Open-Base-1.0 (Datadog): the only tested model that returns real sample paths (multivariate)."""
import numpy as np
import torch
from common import HORIZON, CONTEXT, synth, parse, timeit, report

args = parse()
data = synth()
past, truth = data[:, :CONTEXT], data[:, CONTEXT:]
from toto.data.util.dataset import MaskedTimeseries
from toto.inference.forecaster import TotoForecaster
from toto.model.toto import Toto

toto = Toto.from_pretrained(args.repo or "Datadog/Toto-Open-Base-1.0").to(args.device)
forecaster = TotoForecaster(toto.model)
series = torch.tensor(past).to(args.device)                      # (variates, time)
inputs = MaskedTimeseries(
    series=series,
    padding_mask=torch.ones_like(series, dtype=torch.bool),
    id_mask=torch.zeros_like(series),                            # one id: variates attend to each other
    timestamp_seconds=torch.zeros_like(series) + torch.arange(CONTEXT, device=args.device) * 3600,
    time_interval_seconds=torch.full((series.shape[0],), 3600, device=args.device),
)
def run():
    return forecaster.forecast(inputs, prediction_length=HORIZON, num_samples=64, samples_per_batch=64)
fc, first, med = timeit(run, max(1, args.repeats - 1))
s = fc.samples.cpu().numpy()
print("samples shape", s.shape, "(variates, H, num_samples) or with a batch axis in front")
print("MAE of the median:", float(np.abs(np.median(s, -1).reshape(truth.shape) - truth).mean()))
report("toto-open-base-1.0", args.device, first, med, {"samples": list(s.shape)})
