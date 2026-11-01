"""Stage 4: the cross-run tables of the paper, from runs/<suite>/all_scores.parquet.

    python -m hierfm paper [SUITE ...]        default: ru europe synthetic -> docs/results/<suite>.md

Every table reads only the columns and the experiment it needs (pyarrow filters), so a suite of
tens of millions of rows fits a laptop. Skill is the relative CRPS gain over the base forecasts,
in percent: 100 * (1 - CRPS(method) / CRPS(base)), positive is better.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pads

from . import stats
from .config import ROOT
from .report import table

PRETRAINED = ("chronos2", "chronos2_temp", "chronos2_joint", "timesfm25", "timesfm3", "timesfm3_temp",
              "timesfm3_joint", "moirai2")
METHODS = ["bu", "ols", "wls_struct", "wls_var", "mint_sample", "mint_shrink", "mint_shrink_neff",
           "prior_shrink", "prior_shrink_raw", "prior_only"]
OURS, REFERENCE = "prior_shrink", "mint_shrink"
MAIN_WINDOW = 28
KEYS = ["model", "h", "level", "window", "dependence", "method"]


# ==========================
# ===  Loading           ===
# ==========================
class Suite:
    def __init__(self, name):
        self.name = name
        self.path = ROOT / "runs" / name / "all_scores.parquet"
        self.ds = pads.dataset(self.path)
        self.cols = set(self.ds.schema.names)
        exp = self.ds.to_table(columns=["experiment"]).column("experiment")
        self.experiments = sorted(set(exp.to_pylist()))

    def load(self, experiment, columns, **eq):
        """Rows of one experiment, only the given columns; eq filters by value or list of values."""
        cols = [c for c in dict.fromkeys(columns) if c in self.cols]
        f = pads.field("experiment") == experiment
        for k, v in eq.items():
            if k not in self.cols:
                continue
            f &= pads.field(k).isin(list(v)) if isinstance(v, (list, tuple, set)) else pads.field(k) == v
        return self.ds.to_table(columns=cols, filter=f).to_pandas()


# ==========================
# ===  Skill             ===
# ==========================
def skill(df, by, value="crps_scaled"):
    """Mean score per group and method, then the gain of every method over base in percent.

    The base of a group is the same model, horizon, level, window and dependence, so the gain is
    only the reconciliation. Groups are averaged after the ratio: every model and level weighs the same.
    """
    keys = [k for k in KEYS if k in df.columns]
    extra = [b for b in by if b not in keys]
    m = df.groupby(keys + extra, observed=True)[value].mean().unstack("method")
    if "base" not in m:
        return pd.DataFrame()
    g = 100 * (1 - m.div(m["base"], axis=0))
    g = g.drop(columns="base")
    out = g.groupby([b for b in by if b != "method"], observed=True).mean()
    return out[[c for c in METHODS if c in out]]


def wins(df, value="crps_scaled", ours=OURS, reference=REFERENCE, by=("window",)):
    """Diebold–Mariano per model, horizon, level and window: ours vs reference, counted by outcome."""
    rows = []
    # synthetic grids share their origin dates, so every case is tested on its own
    keys = ["model", "h", "level", "window", "dependence"] + (["case"] if "case" in df else [])
    for k, g in df[df.method.isin([ours, reference])].groupby(keys, observed=True):
        p = g.pivot_table(index="origin", columns="method", values=value)
        if ours not in p or reference not in p:
            continue
        stat, pv = stats.diebold_mariano(p[ours], p[reference], h=max(3, int(k[1]) // 24))
        if np.isnan(pv):
            continue
        res = "ours better" if pv < 0.05 and stat < 0 else "reference better" if pv < 0.05 else "no difference"
        rows.append({**dict(zip(keys, k)), "result": res})
    if not rows:
        return pd.DataFrame()
    r = pd.DataFrame(rows)
    return r.groupby(list(by))["result"].value_counts().unstack(fill_value=0)


def _pre(df):
    return df[df.model.isin(PRETRAINED)] if df.model.isin(PRETRAINED).any() else df


# ==========================
# ===  Sections          ===
# ==========================
COLS = KEYS + ["origin", "crps_scaled", "mase", "cov80", "cov95", "width80", "width95", "energy",
               "variogram", "lambda", "n_eff", "holiday", "cold", "case"]


def sec_main(s, out):
    df = s.load("main", COLS)
    ind = df[df.dependence == "independent"]
    w = MAIN_WINDOW if MAIN_WINDOW in ind.window.unique() else int(ind.window.min())

    base = ind[(ind.method == "base") & (ind.window == w)]
    t = base.groupby(["model", "h", "level"])[["crps_scaled", "mase", "cov80", "cov95"]].mean()
    out.append(("Base forecasts: scaled CRPS, MASE and coverage by model, horizon and level", t))

    for h, g in _pre(ind).groupby("h"):
        out.append((f"Skill over base by covariance window, pretrained models, all levels, h{h}",
                    skill(g, ["window", "method"])))
        out.append((f"Skill over base by level, window {w} d, pretrained models, h{h}",
                    skill(g[g.window == w], ["level", "method"])))
    out.append(("Skill over base by model, window 7 d and window 28 d, all levels",
                skill(ind[ind.window.isin([7, w])], ["model", "h", "window", "method"])))

    out.append((f"Diebold–Mariano, {OURS} vs {REFERENCE}, per model, horizon and level, by window",
                wins(ind)))
    out.append((f"Diebold–Mariano, {OURS} vs mint_shrink_neff, by window",
                wins(ind, reference="mint_shrink_neff")))

    lam = ind[ind.method.isin(["mint_shrink", "mint_shrink_neff", "prior_shrink", "prior_shrink_raw"])]
    out.append(("Shrinkage weight lambda by window (mean over origins and models)",
                lam.groupby(["window", "method"])["lambda"].mean().unstack("method")))

    dep = _pre(df[(df.window == w) & df.method.isin(["base", REFERENCE, OURS])])
    t = dep.groupby(["h", "level", "dependence", "method"])[["crps_scaled", "cov80", "cov95", "energy",
                                                              "variogram"]].mean()
    out.append((f"Dependence of the sample paths, window {w} d, pretrained models", t))

    st = []
    for col in ("holiday", "cold"):
        if col in ind:
            sk = skill(_pre(ind[ind.window == w]), [col, "method"])
            sk.index = [f"{col}={v}" for v in sk.index]
            st.append(sk)
    if st:
        out.append((f"Skill over base in hard strata (holidays, coldest days), window {w} d", pd.concat(st)))


def sec_by_case(s, out, exp, title, models=None):
    """Skill and base CRPS of one experiment, split by its grid or case keys."""
    df = s.load(exp, COLS + [c for c in s.cols if c.startswith(
        ("history_days", "context", "temp_source", "prior_", "samples", "seed", "synthetic_"))])
    if df.empty:
        return
    ind = df[df.dependence == "independent"] if (df.dependence == "independent").any() else df
    if models:
        ind = ind[ind.model.isin(models)]
    by = ["case"]
    base = ind[ind.method == "base"].groupby(by + ["model", "h", "level"])["crps_scaled"].mean()
    if len(base):
        out.append((f"{title}: base scaled CRPS", base.unstack("level")))
    sk = skill(ind, by + ["window", "method"])
    if len(sk):
        out.append((f"{title}: skill over base by window, all levels", sk))
    if {"energy", "cov80"} <= set(df.columns) and exp in ("samples", "seeds"):
        t = df.groupby(by + ["dependence", "method"])[["crps_scaled", "energy", "cov80", "cov95"]].mean()
        out.append((f"{title}: scores by dependence", t))


def sec_synthetic(s, out):
    keys = ["synthetic_common_sd", "synthetic_regional_sd", "synthetic_regions"]
    df = s.load("study", COLS + keys)
    ind = df[df.dependence == "independent"]
    for k in keys:
        out.append((f"Synthetic grids: skill over base by {k}, all windows", skill(ind, [k, "method"])))
    sk = skill(ind, keys[:2] + ["method"])
    if OURS in sk and REFERENCE in sk:
        d = (sk[OURS] - sk[REFERENCE]).unstack(keys[1])
        out.append((f"Synthetic grids: {OURS} minus {REFERENCE}, skill points, common x regional weather", d))
    out.append(("Synthetic grids: skill over base by window", skill(ind, ["window", "method"])))
    out.append((f"Synthetic grids: Diebold–Mariano {OURS} vs {REFERENCE} by window", wins(ind)))


EXPERIMENTS = {
    "cold_start": "Cold start (history in days)",
    "context": "Model input length",
    "model_size": "Model size (Chronos-Bolt)",
    "joint_scope": "Joint forecasting scope",
    "weather": "Weather input",
    "ensemble": "Ensembles of pretrained models",
    "prior": "Prior of method A",
    "samples": "Number of sample paths",
    "seeds": "Sampling seed",
}


def build(name):
    s = Suite(name)
    out = []
    if "main" in s.experiments:
        sec_main(s, out)
    for exp, title in EXPERIMENTS.items():
        if exp in s.experiments:
            sec_by_case(s, out, exp, title)
    if "study" in s.experiments:
        sec_synthetic(s, out)
    dest = ROOT / "docs" / "results" / f"{name}.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    parts = [f"# Results: suite {name}\n\nSkill: 100 * (1 - CRPS(method) / CRPS(base)), percent, "
             f"positive is better. Scores are scaled by the seasonal naive error of each node.\n"]
    for title, t in out:
        if t is None or len(t) == 0:
            continue
        parts.append(f"## {title}\n\n{table(t.round(3) if hasattr(t, 'round') else t, '.3f')}\n")
    dest.write_text("\n".join(parts))
    print(f"paper: {dest} ({len(out)} tables)", flush=True)


def main(names):
    for n in names or ["ru", "europe", "synthetic"]:
        if (ROOT / "runs" / n / "all_scores.parquet").exists():
            build(n)
        else:
            print(f"paper: no runs/{n}/all_scores.parquet, skip", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
