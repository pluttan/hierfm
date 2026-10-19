#!/usr/bin/env python3
"""How well do the fill methods of build_ru.py rebuild subject load on days ATS does not publish?

Test: hide two consecutive working days (Tue and Wed, Mon and Thu stay) for all subjects, rebuild them
with each method and compare with the hidden truth. A weekend gap has the same length, but its level
differs from the neighbours, so this is an optimistic check. Output: MAPE in percent.

    nice -n 19 python3 -u src/check_ru_fill.py RAW_DIR [FROM] [TO]
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_ru as b  # noqa: E402
import so_ups_hierarchy as sh  # noqa: E402

warnings.filterwarnings("ignore")
lo = sys.argv[2] if len(sys.argv) > 2 else "2023-01-01"
hi = sys.argv[3] if len(sys.argv) > 3 else "2026-09-30"
root = b.so_root(sys.argv[1])
ats_all = b.read_ats(root)
ats = ats_all.loc["2019-01-01":]
fact, _ = b.read_genconsum(root)
plan = b.read_plan(root)
ipsof = {c: v[1] for c, v in sh.R.items()}
core = [c for c in ats.columns if c in sh.R and ats_all[c].first_valid_index() < pd.Timestamp("2018-01-01")]
idx = pd.date_range(ats.index.min(), ats.index.max(), freq="h")
X, _ = b.clean_ats(ats_all.reindex(pd.date_range(ats_all.index.min(), ats_all.index.max(), freq="h"))[core])
X = X.reindex(idx)
days = X.notna().any(axis=1).groupby(X.index.normalize()).first()
dset = set(days[days].index)
# candidate pairs: Tue+Wed in the test window, Mon and Thu observed
cand = [d for d in days[days].index if lo <= str(d.date()) <= hi and d.weekday() == 1
        and (d + pd.Timedelta(days=1)) in dset and (d - pd.Timedelta(days=1)) in dset
        and (d + pd.Timedelta(days=2)) in dset]
hide = cand[::4]
print("hidden pairs:", len(hide), "first", hide[0].date(), "last", hide[-1].date())
mask = pd.Series(False, index=idx)
for d in hide:
    mask[d:d + pd.Timedelta(days=2) - pd.Timedelta(hours=1)] = True
Xh = X.copy()
Xh.loc[mask.to_numpy(), :] = np.nan
res = {}
for name in ("share", "share_causal", "plan", "prev_day", "avg_neighbours"):
    est = pd.DataFrame(index=idx, columns=core, dtype=float)
    for ips in sh.IPS:
        cols = [c for c in core if ipsof[c] == ips]
        if not cols:
            continue
        so_k = b.smooth_fact(fact[f"IPS_{ips}"], idx)
        if name == "share":
            est[cols] = b.fill_share(Xh[cols], so_k)
        elif name == "share_causal":
            est[cols] = b.fill_share(Xh[cols], so_k, causal=True)
        elif name == "plan":
            est[cols] = b.fill_plan(Xh[cols], plan, so_k)
        elif name == "prev_day":
            # previous observed day, same hour: the Monday before the gap
            e = Xh[cols].copy()
            for d in hide:
                for k in (0, 1):
                    src = d - pd.Timedelta(days=1)
                    tgt = d + pd.Timedelta(days=k)
                    e.loc[tgt:tgt + pd.Timedelta(hours=23), cols] = X.loc[src:src + pd.Timedelta(hours=23), cols].to_numpy()
            est[cols] = e
        else:
            e = Xh[cols].copy()
            for d in hide:
                for k in (0, 1):
                    tgt = d + pd.Timedelta(days=k)
                    a = X.loc[d - pd.Timedelta(days=1):d - pd.Timedelta(days=1) + pd.Timedelta(hours=23), cols].to_numpy()
                    c = X.loc[d + pd.Timedelta(days=2):d + pd.Timedelta(days=2) + pd.Timedelta(hours=23), cols].to_numpy()
                    e.loc[tgt:tgt + pd.Timedelta(hours=23), cols] = (a + c) / 2
            est[cols] = e
    m = mask.to_numpy()
    ape = ((est[m] - X[m]).abs() / X[m])
    res[name] = ape
    print(f"\n== {name}: MAPE %, all subjects pooled {100 * np.nanmean(ape.to_numpy()):.2f}, "
          f"median subject {100 * ape.mean().median():.2f}, worst subject "
          f"{100 * ape.mean().max():.2f} ({ape.mean().idxmax()})")
    per_ips = {}
    for ips in sh.IPS:
        cols = [c for c in core if ipsof[c] == ips]
        if cols:
            tot_est, tot = est.loc[m, cols].sum(axis=1), X.loc[m, cols].sum(axis=1)
            per_ips[ips] = round(float(100 * ((tot_est - tot).abs() / tot).mean()), 2)
    print("   IPS-level MAPE %:", per_ips)
    ue, ut = est.loc[m].sum(axis=1), X.loc[m].sum(axis=1)
    print("   all-subjects sum MAPE %:", round(float(100 * ((ue - ut).abs() / ut).mean()), 2))
    # error correlation between subjects: does the fill keep a realistic dependence?
    err = (est[m] - X[m]) / X[m]
    cor = err.corr().to_numpy()
    print("   mean off-diagonal correlation of relative errors between subjects:",
          round(float((cor.sum() - len(cor)) / (len(cor) * (len(cor) - 1))), 3))
