"""Numbers of the paper (figures and tables) from the score archives, written as small CSVs.
usage: python -I paper/build/numbers.py RESULTS_DIR OUT_DIR
RESULTS_DIR holds runs/<suite>/all_scores.parquet (the cluster archive); OUT_DIR gets the CSVs,
which are committed, so the figures and the text build without the 2 GB archive.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # the hierfm package
from hierfm import paper as P  # noqa: E402

SHOWN = ["bu", "wls_var", "mint_shrink", "mint_shrink_neff", "prior_shrink"]
COLS = P.KEYS + ["origin", "crps_scaled", "cov80", "cov95", "lambda", "case"]


def main_numbers(s, out):
    df = s.load("main", COLS)
    pre = P._pre(df)
    sk = P.skill(pre, ["h", "dependence", "window", "method"])
    sk[SHOWN].reset_index().assign(suite=s.name).to_csv(out / f"dep_skill_{s.name}.csv", index=False)

    base = df[(df.method == "base") & (df.window == P.MAIN_WINDOW) & (df.dependence == "independent")]
    t = base.groupby(["model", "h", "level"])[["crps_scaled", "cov80", "cov95"]].mean().reset_index()
    t.assign(suite=s.name).to_csv(out / f"base_{s.name}.csv", index=False)

    lam = df[(df.dependence == "independent") & df.method.isin(["mint_shrink", "mint_shrink_neff", "prior_shrink"])]
    lam.groupby(["window", "method"])["lambda"].mean().unstack("method").reset_index().assign(
        suite=s.name).to_csv(out / f"lambda_{s.name}.csv", index=False)

    # Diebold-Mariano of the coherent methods against base, Schaake by history, the main window
    sh = pre[(pre.dependence == "schaake_hist") & (pre.window == P.MAIN_WINDOW)]
    rows = []
    for ours in ("mint_shrink_neff", "wls_var", "bu", "mint_shrink", "prior_shrink"):
        w = P.wins(sh, ours=ours, reference="base", by=("h",))
        for h, r in w.iterrows():
            rows.append({"suite": s.name, "method": ours, "h": h, **r.to_dict()})
    w = P.wins(sh, ours="mint_shrink_neff", reference="mint_shrink", by=("h",))
    for h, r in w.iterrows():
        rows.append({"suite": s.name, "method": "neff_vs_mint_shrink", "h": h, **r.to_dict()})
    pd.DataFrame(rows).fillna(0).to_csv(out / f"dm_{s.name}.csv", index=False)


def with_base(sk, df, keys):
    """Skill table plus the scaled CRPS of the base forecasts (mean over levels) of each row."""
    b = df[(df.method == "base") & (df.dependence == "independent")]
    b = b.groupby(keys + ["level"])["crps_scaled"].mean().groupby(keys).mean().rename("base_crps")
    return sk[SHOWN].reset_index().merge(b.reset_index(), on=keys, how="left")


def weak_numbers(s, out):
    """Skill on weak bases: short model input, small models, cold start, gradient boosting."""
    parts = []
    for exp in ("context", "model_size", "cold_start"):
        if exp not in s.experiments:
            continue
        df = s.load(exp, COLS + [c for c in s.cols if c.startswith(("history_days", "context"))])
        if exp == "cold_start":
            df = df[df.model != "gbm"]
        sk = P.skill(df, ["case", "model", "dependence", "window", "method"])
        parts.append(with_base(sk, df, ["case", "model"]).assign(experiment=exp))
    df = s.load("main", COLS, model=["gbm", "snaive", "chronos2", "timesfm25", "timesfm3"])
    sk = P.skill(df, ["model", "h", "dependence", "window", "method"])
    parts.append(with_base(sk, df, ["model", "h"]).assign(experiment="main"))
    pd.concat(parts).to_csv(out / f"weak_{s.name}.csv", index=False)


def synthetic_numbers(s, out):
    keys = ["synthetic_common_sd", "synthetic_regional_sd"]
    df = s.load("study", P.COLS + keys + ["synthetic_regions"])
    ind = df[df.dependence == "independent"]
    sk = P.skill(ind, keys + ["method"])
    sk.reset_index().to_csv(out / "synthetic_grid.csv", index=False)
    P.skill(ind, ["window", "method"]).reset_index().to_csv(out / "synthetic_window.csv", index=False)
    P.skill(ind, ["synthetic_regions", "method"]).reset_index().to_csv(out / "synthetic_regions.csv", index=False)


def run(results, out):
    P.ROOT = Path(results)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("ru", "europe"):
        s = P.Suite(name)
        main_numbers(s, out)
        weak_numbers(s, out)
        print(f"numbers: {name} done", flush=True)
    synthetic_numbers(P.Suite("synthetic"), out)
    print("numbers: synthetic done", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    run(*sys.argv[1:])
