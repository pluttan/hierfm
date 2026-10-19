"""Moirai-2.0-R-small via uni2ts. Output of Moirai-2 is quantiles (9), not sample paths."""
import numpy as np
import torch
from common import HORIZON, CONTEXT, synth, parse, timeit, report

args = parse()
data = synth()
past, truth = data[:, :CONTEXT], data[:, CONTEXT:]
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
module = Moirai2Module.from_pretrained(args.repo or "Salesforce/moirai-2.0-R-small")
model = Moirai2Forecast(module=module, prediction_length=HORIZON, context_length=CONTEXT, target_dim=1,
                        feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
model.to(args.device)
print("forecast methods:", [m for m in dir(model) if "predict" in m or "forecast" in m or "sample" in m][:20])

# direct call: past_target (batch, time, 1), observed mask, is_pad mask
pt = torch.tensor(past)[:, :, None].to(args.device)
obs = torch.ones_like(pt, dtype=torch.bool)
pad = torch.zeros(pt.shape[:2], dtype=torch.bool, device=args.device)
def run():
    with torch.no_grad():
        return model(past_target=pt, past_observed_target=obs, past_is_pad=pad)
out, first, med = timeit(run, args.repeats)
print("output shape", tuple(out.shape), "(batch, n_quantiles, H) expected")
report("moirai-2.0-R-small", args.device, first, med, {"shape": list(out.shape)})
q = out.cpu().numpy()
print("MAE of the median (index 4):", float(np.abs(q[:, 4, :] - truth).mean()))
