"""Wrappers of pretrained time series models. Imports are lazy: each lives in its own package
and a missing package must only disable that model, not the whole run.

Every wrapper is zero-shot by default. Covariates (temperature) are passed only to models that
accept them and only when the model params ask for it, so the ablation "with/without weather"
is one config switch.
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
        q = out[[str(x) for x in self.NATIVE]].to_numpy().reshape(b, h, len(self.NATIVE))
        return np.asarray(levels), to_levels(self.NATIVE, q, levels)


class ChronosBolt(Model):
    """Chronos-Bolt tiny/mini/small/base (9M..205M): the model-size axis of the experiments."""
    name = "chronos_bolt"
    NATIVE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

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

    def _model(self, L, h):
        if not hasattr(self, "model"):
            import timesfm
            import torch
            torch.set_float32_matmul_precision("high")
            self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                self.params.get("repo", "google/timesfm-2.5-200m-pytorch"))
            self.model.compile(timesfm.ForecastConfig(
                max_context=int(self.params.get("max_context", 2048)), max_horizon=max(256, h),
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

    def predict(self, ctx, h, levels, temp_past=None, temp_future=None, node_ids=None):
        import pandas as pd
        from gluonts.dataset.common import ListDataset
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
        key = (ctx.shape[1], h)
        if getattr(self, "key", None) != key:
            module = Moirai2Module.from_pretrained(self.params.get("repo", "Salesforce/moirai-2.0-R-small"))
            model = Moirai2Forecast(module=module, prediction_length=h, context_length=ctx.shape[1],
                                    target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
            self.predictor = model.create_predictor(batch_size=int(self.params.get("batch", 64)),
                                                    device=self.device)
            self.key = key
        start = pd.Period("2000-01-01 00:00", freq="h")
        data = ListDataset([{"start": start, "target": row} for row in ctx], freq="h")
        q = np.stack([np.stack([f.quantile(lv) for lv in self.NATIVE], -1)
                      for f in self.predictor.predict(data)])
        return np.asarray(levels), to_levels(self.NATIVE, q, levels)
