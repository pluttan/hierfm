"""Stage 3: the tables the paper needs, from runs/scores/*.parquet, as CSV and Markdown."""

from pathlib import Path

import pandas as pd

from . import stats

MAIN_WINDOW_DAYS = 28


def _md(df, path, title):
    path.write_text(f"# {title}\n\n{df.to_markdown(floatfmt='.4f')}\n")


def run(cfg, reference="mint_shrink", ours="prior_shrink"):
    src = Path(cfg.runs) / "scores"
    out = Path(cfg.runs) / "report"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_parquet(p) for p in sorted(src.glob("*.parquet"))])
    window = MAIN_WINDOW_DAYS if MAIN_WINDOW_DAYS in df.window.unique() else df.window.min()
    cov_cols = [c for c in df.columns if c.startswith("cov")]

    # accuracy by level at the main window, independent draws: methods side by side
    ind = df[(df.window == window) & (df.dependence == "independent")]
    acc = ind.groupby(["model", "h", "level", "method"])[["mase", "crps_scaled"] + cov_cols].mean()
    acc.to_csv(out / "accuracy_by_level.csv")
    _md(acc.unstack("method")["mase"], out / "accuracy_by_level.md", f"MASE by level, window {window} d")

    # window length: the first claim of the paper, ours vs MinT-shrink at the top level
    win = df[(df.dependence == "independent") & (df.method.isin([reference, ours, "wls_var", "bu"]))]
    win = win.groupby(["model", "h", "level", "window", "method"])["mase"].mean().unstack("method")
    win.to_csv(out / "window_length.csv")
    _md(win, out / "window_length.md", "MASE by covariance window length")

    # calibration: the second claim, coverage by dependence and level
    cal = df[(df.window == window) & (df.method.isin(["base", ours]))]
    cal = cal.groupby(["model", "h", "level", "dependence", "method"])[cov_cols + ["crps_scaled"]].mean()
    cal.to_csv(out / "calibration.csv")
    _md(cal, out / "calibration.md", f"Coverage and CRPS by dependence, window {window} d")

    # significance: ours vs reference per level, and MCB over all methods at the top level
    rows = []
    for (model, h, lv, w), g in ind.groupby(["model", "h", "level", "window"]):
        p = g.pivot_table(index="origin", columns="method", values="crps_scaled")
        if ours in p and reference in p:
            stat, pv = stats.diebold_mariano(p[ours], p[reference], h=max(1, h // 24))
            rows.append({"model": model, "h": h, "level": lv, "window": w, "dm": stat, "p": pv,
                         "ours_better": p[ours].mean() < p[reference].mean()})
    dm = pd.DataFrame(rows)
    dm.to_csv(out / "dm_test.csv", index=False)
    if len(dm):
        _md(dm.set_index(["model", "h", "level", "window"]), out / "dm_test.md",
            f"Diebold–Mariano, {ours} vs {reference}, CRPS")

    lines = []
    for (model, h), g in ind[ind.level == 0].groupby(["model", "h"]):
        ranks, cd, ok = stats.mcb(g.pivot_table(index="origin", columns="method", values="crps_scaled"))
        lines.append(f"## {model} h{h}\n\ncritical distance {cd:.3f}; not worse than best: {', '.join(ok)}\n\n"
                     f"{ranks.to_frame('mean rank').to_markdown(floatfmt='.3f')}\n")
    (out / "mcb.md").write_text("# MCB, top level, CRPS\n\n" + "\n".join(lines))
    print(f"report: {out}", flush=True)
