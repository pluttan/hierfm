"""Wrappers of pretrained time series models. Imports are lazy: each lives in its own package
and a missing package must only disable that model, not the whole run.

Every wrapper is zero-shot by default. Covariates (temperature) are passed only to models that
accept them and only when the model params ask for it, so the ablation "with/without weather"
is one config switch.

Checked against the sources on 2026-10-19 (see docs/research/09-environment.md); lines marked
"CHECK 2026-10-19" say what was verified or fixed and where. Nothing here was run on a GPU yet.
"""

import numpy as np

from .base import Model, to_levels


def _batches(n, size):
    for i in range(0, n, size):
        yield slice(i, min(n, i + size))


# ==========================
# ===  Chronos-2         ===
# ==========================

class Chronos2(Model):
    name = "chronos2"
    uses_temp = True
    NATIVE = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
              0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]

    # CHECK 2026-10-19 (amazon-science/chronos-forecasting, src/chronos/chronos2/pipeline.py, 2.3.2):
    # predict_df(df, future_df, id_column, timestamp_column, target, prediction_length,
    # quantile_levels, batch_size, ...) matches the calls below. The result is long: id, timestamp,
    # target_name, "predictions", then one column per level named str(level). The 21 levels in NATIVE
    # equal chronos_config.quantiles of amazon/chronos-2. A list in `target` gives one block of rows per
    # target. No argument names changed; only shape checks were added.
    def _pipe(self):
        if not hasattr(self, "pipe"):
            from chronos import Chronos2Pipeline
            self.pipe = Chronos2Pipeline.from_pretrained(self.params.get("repo", "amazon/chronos-2"),
                                                         device_map=self.device)
        return self.pipe

    def predict(self, ctx, h, levels, temp_past=None, temp_future=None, node_ids=None):
        import pandas as pd
        pipe = self._pipe()
        b, L = ctx.shape
        ts = pd.date_range("2000-01-01", periods=L + h, freq="h")
        with_temp = self.params.get("temp", False) and temp_past is not None
        joint = self.params.get("joint", False)

        if joint:
            # all series of the batch are variates of one item: Chronos-2 attends across them
            ctx_df = pd.DataFrame({"id": 0, "timestamp": ts[:L], **{f"v{i}": ctx[i] for i in range(b)}})
            targets = [f"v{i}" for i in range(b)]
            fut = None
            if with_temp:
                ctx_df["temp"] = temp_past.mean(0)
                fut = pd.DataFrame({"id": 0, "timestamp": ts[L:], "temp": temp_future.mean(0)})
            out = pipe.predict_df(ctx_df, future_df=fut, prediction_length=h, quantile_levels=self.NATIVE,
                                  id_column="id", timestamp_column="timestamp", target=targets)
            q = np.stack([out[out["target_name"] == t][[str(x) for x in self.NATIVE]].to_numpy()
                          for t in targets])
            assert q.shape == (b, h, len(self.NATIVE)), q.shape   # rows of one target are in time order
            return np.asarray(levels), to_levels(self.NATIVE, q, levels)

        frames, futs = [], []
        for i in range(b):
            f = pd.DataFrame({"id": i, "timestamp": ts[:L], "target": ctx[i]})
            if with_temp:
                f["temp"] = temp_past[i]
                futs.append(pd.DataFrame({"id": i, "timestamp": ts[L:], "temp": temp_future[i]}))
            frames.append(f)
        out = pipe.predict_df(pd.concat(frames), future_df=pd.concat(futs) if futs else None,
                              prediction_length=h, quantile_levels=self.NATIVE,
                              id_column="id", timestamp_column="timestamp", target="target")
        out = out.sort_values(["id", "timestamp"])
        assert len(out) == b * h, (len(out), b, h)
        q = out[[str(x) for x in self.NATIVE]].to_numpy().reshape(b, h, len(self.NATIVE))
        return np.asarray(levels), to_levels(self.NATIVE, q, levels)


class ChronosBolt(Model):
    """Chronos-Bolt tiny/mini/small/base (9M..205M): the model-size axis of the experiments."""
    name = "chronos_bolt"
    NATIVE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # CHECK 2026-10-19 (chronos_bolt.py, base.py of chronos-forecasting 2.3.2): predict_quantiles(inputs,
    # quantile_levels, **kw) takes prediction_length through kw and returns (quantiles, mean) with
    # quantiles of shape (batch, prediction_length, n_levels): the order and shape used below are right.
    # device_map and torch_dtype are passed through to transformers (the base class renames
    # torch_dtype to dtype for transformers >= 4.56). The model predicts 64 steps natively; for
    # h = 168 it rolls forward on its own median, so the tail quantiles of long horizons are less reliable.
    def predict(self, ctx, h, levels, temp_past=None, temp_future=None, node_ids=None):
        import torch
        if not hasattr(self, "pipe"):
            from chronos import BaseChronosPipeline
            self.pipe = BaseChronosPipeline.from_pretrained(self.params.get("repo", "amazon/chronos-bolt-small"),
                                                            device_map=self.device, torch_dtype=torch.float32)
        qs = []
        for sl in _batches(len(ctx), int(self.params.get("batch", 512))):
            q, _ = self.pipe.predict_quantiles(torch.tensor(ctx[sl], dtype=torch.float32),
                                               prediction_length=h, quantile_levels=self.NATIVE)
            qs.append(q.cpu().numpy())
        return np.asarray(levels), to_levels(self.NATIVE, np.concatenate(qs), levels)


# ==========================
# ===  TimesFM 2.5 / 3.0 ===
# ==========================

class TimesFM25(Model):
    name = "timesfm25"
    NATIVE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # CHECK 2026-10-19 (google-research/timesfm, src/timesfm/timesfm_2p5/*, timesfm 3.0.2). Fixed here:
    # 1. ForecastConfig.per_core_batch_size defaults to 1 and the compiled decoder runs
    #    global_batch_size = per_core_batch_size * device_count series per step. With the default every
    #    series is a separate decoder call. It is now set to the "batch" parameter.
    # 2. The package hard-codes torch.device("cuda:0") and device_count = all visible GPUs. Under
    #    torchrun every rank would then sit on cuda:0 and pad its batches to a multiple of the card
    #    count. The device and device_count=1 are now set by hand (same trick as src/smoke/smoke_timesfm.py).
    # 3. compile() fixes max_horizon. A later call with a larger h raised "Horizon must be less than
    #    the max horizon". The model is now rebuilt when h exceeds the compiled value.
    # Unchanged and right: forecast(horizon, inputs) returns (point (n, h), quantiles (n, h, 10));
    # column 0 of the quantiles is the mean, columns 1..9 are the levels 0.1..0.9.
    def _model(self, L, h):
        batch = int(self.params.get("batch", 256))
        if getattr(self, "compiled_h", 0) < h:
            import timesfm
            import torch
            torch.set_float32_matmul_precision("high")
            kw = {}
            if "torch_compile" in self.params:
                kw["torch_compile"] = bool(self.params["torch_compile"])
            self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                self.params.get("repo", "google/timesfm-2.5-200m-pytorch"), **kw)
            if str(self.device).startswith("cuda"):
                self.model.model.device = torch.device(self.device)
                self.model.model.device_count = 1
                self.model.model.to(self.device)
            self.compiled_h = max(256, -(-h // 256) * 256)
            self.model.compile(timesfm.ForecastConfig(
                max_context=int(self.params.get("max_context", 2048)), max_horizon=self.compiled_h,
                per_core_batch_size=batch,
                normalize_inputs=True, use_continuous_quantile_head=True, force_flip_invariance=True,
                infer_is_positive=True, fix_quantile_crossing=True))
        return self.model

    def predict(self, ctx, h, levels, temp_past=None, temp_future=None, node_ids=None):
        model = self._model(ctx.shape[1], h)
        qs = []
        for sl in _batches(len(ctx), int(self.params.get("batch", 256))):
            _, quant = model.forecast(horizon=h, inputs=list(ctx[sl].astype(np.float32)))
            qs.append(np.asarray(quant)[:, :h, 1:])      # column 0 is the mean, then deciles
        q = np.concatenate(qs)
        return np.asarray(levels), to_levels(self.NATIVE, q, levels)


class TimesFM3(Model):
    """TimesFM 3.0: weights are non-commercial, fine for a paper; supports joint (multivariate) mode."""
    name = "timesfm3"
    uses_temp = True
    NATIVE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # CHECK 2026-10-19 (google-research/timesfm, src/timesfm3/torch/{timesfm3_forecaster,evaluator}.py,
    # package timesfm 3.0.2 installs the top-level module timesfm3). No change needed:
    # ModelConfig(checkpoint_path, per_core_batch_size, device) and TimesFM3Evaluator(config) exist;
    # predict_batch(contexts, horizon, past_only_covariates, past_future_covariates, ts_ids,
    # return_quantiles, use_symmetric_averaging, make_positive, sort_quantiles, use_znorm,
    # padding_mode, univariate) yields ForecastOutput(forecast, quantiles). A 1-D context gives
    # quantiles (h, 9), a 2-D context (n_variates, L) gives (n_variates, h, 9), levels 0.1 .. 0.9.
    # A past_future covariate must have length L + h. The model reads h from it, so padding_mode="none"
    # is right for h = 24 and 168. The evaluator caps a forward pass at 32 variates (max_variates in
    # config.json): a joint group larger than 31 (plus covariate rows) is cut into chunks, and the
    # chunks do not see each other. The context is cut to the last 15360 points.
    def _fc(self):
        if not hasattr(self, "fc"):
            from timesfm3 import ModelConfig, TimesFM3Evaluator
            self.fc = TimesFM3Evaluator(ModelConfig(
                checkpoint_path=self.params.get("repo", "google/timesfm-3.0-pytorch"),
                per_core_batch_size=int(self.params.get("batch", 32)), device=self.device))
        return self.fc

    def predict(self, ctx, h, levels, temp_past=None, temp_future=None, node_ids=None):
        fc = self._fc()
        with_temp = self.params.get("temp", False) and temp_past is not None
        if self.params.get("joint", False):
            kw = {}
            if with_temp:
                kw["past_future_covariates"] = [np.concatenate([temp_past, temp_future], 1).mean(0, keepdims=True)
                                                .astype(np.float32)]
            out = list(fc.predict_batch(contexts=[ctx.astype(np.float32)], horizon=h, return_quantiles=True,
                                        use_symmetric_averaging=False, **kw))[0]
            q = np.asarray(out.quantiles)                    # (b, h, 9)
        else:
            kw = {}
            if with_temp:
                kw["past_future_covariates"] = [np.concatenate([temp_past[i], temp_future[i]])[None]
                                                .astype(np.float32) for i in range(len(ctx))]
            outs = fc.predict_batch(list(ctx.astype(np.float32)), horizon=h, return_quantiles=True,
                                    use_symmetric_averaging=False, **kw)
            q = np.stack([np.asarray(o.quantiles) for o in outs])
        return np.asarray(levels), to_levels(self.NATIVE, q, levels)


# ==========================
# ===  Moirai-2          ===
# ==========================

class Moirai2(Model):
    """uni2ts pins numpy 1.26 and torch < 2.5, which have no Python 3.13 builds; it is installed
    with --no-deps (make install-moirai); if it does not run on current torch the model is skipped."""
    name = "moirai2"
    NATIVE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # CHECK 2026-10-19 (SalesforceAIResearch/uni2ts main, src/uni2ts/model/moirai2/forecast.py; PyPI
    # uni2ts 2.0.0 of 2025-11-04 contains uni2ts/model/moirai2). Changed: the model is now called
    # directly instead of through the gluonts PyTorchPredictor. Reasons: (a) the predictor needs a
    # ListDataset with a pandas Period and keeps gluonts and pandas versions in the loop, and
    # QuantileForecastGenerator takes float levels where its annotation says List[str], which works
    # only through the pydantic coercion of gluonts; (b) the direct call is what src/smoke/smoke_moirai.py
    # uses. forward(past_target (b, L, 1), past_observed_target (b, L, 1) bool, past_is_pad (b, L) bool)
    # returns (b, n_quantiles, h) in the order of module.quantile_levels = 0.1 .. 0.9 (config.json of
    # Salesforce/moirai-2.0-R-small). The scaling is inside the module, so no loc/scale is applied.
    # The model emits 4 patches of 16 = 64 steps at once; for h = 168 the code rolls forward itself.
    def predict(self, ctx, h, levels, temp_past=None, temp_future=None, node_ids=None):
        import torch
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        key = (ctx.shape[1], h)
        if getattr(self, "key", None) != key:
            if not hasattr(self, "module"):
                self.module = Moirai2Module.from_pretrained(
                    self.params.get("repo", "Salesforce/moirai-2.0-R-small"))
            self.net = Moirai2Forecast(module=self.module, prediction_length=h, context_length=ctx.shape[1],
                                       target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
            self.net.to(self.device).eval()
            self.key = key
        qs = []
        for sl in _batches(len(ctx), int(self.params.get("batch", 64))):
            x = np.asarray(ctx[sl], np.float32)
            seen = np.isfinite(x)
            pt = torch.from_numpy(np.where(seen, x, 0.0))[:, :, None].to(self.device)
            obs = torch.from_numpy(seen)[:, :, None].to(self.device)
            pad = torch.zeros(pt.shape[:2], dtype=torch.bool, device=self.device)
            with torch.no_grad():
                out = self.net(past_target=pt, past_observed_target=obs, past_is_pad=pad)
            qs.append(out.permute(0, 2, 1).float().cpu().numpy())      # (b, nq, h) -> (b, h, nq)
        return np.asarray(levels), to_levels(self.NATIVE, np.concatenate(qs), levels)
