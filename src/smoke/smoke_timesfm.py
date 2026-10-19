"""TimesFM 2.5 (200M, torch): point + 10 columns (mean, q0.1..q0.9). No sample paths."""
import numpy as np
import torch
import timesfm
from common import HORIZON, CONTEXT, synth, parse, timeit, report

args = parse()
data = synth()
past, truth = data[:, :CONTEXT], data[:, CONTEXT:]
torch.set_float32_matmul_precision("high")
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
if args.device == "mps":
    model.model.to("mps")      # the package picks cuda/cpu itself; try to move manually
model.compile(timesfm.ForecastConfig(max_context=2048, max_horizon=256, normalize_inputs=True,
              use_continuous_quantile_head=True, force_flip_invariance=True, infer_is_positive=True,
              fix_quantile_crossing=True))
def run():
    return model.forecast(horizon=HORIZON, inputs=list(past))
(point, quant), first, med = timeit(run, args.repeats)
print("point", point.shape, "quantiles", quant.shape, "(col 0 = mean, then deciles 0.1..0.9)")
print("MAE point:", float(np.abs(point - truth).mean()))
report("timesfm-2.5", args.device, first, med, {"point": list(point.shape), "quant": list(quant.shape)})
