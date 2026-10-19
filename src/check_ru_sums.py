#!/usr/bin/env python3
"""Checks of the Russian hierarchy on the raw tables: coverage, weekday gaps, hour alignment, sums, membership.

    nice -n 19 python3 -u src/check_ru_sums.py RAW_DIR [cover|days|align|sums|member|plan] ...

Run on pcomp (pandas and numpy are there; no installs needed). Output is plain text.
"""

import sys
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_ru as b  # noqa: E402
import so_ups_hierarchy as sh  # noqa: E402

warnings.filterwarnings("ignore")
pd.set_option("display.width", 220, "display.max_columns", 40, "display.max_rows", 300)
what = sys.argv[2:] or ["cover", "days", "align", "sums", "member"]
root = b.so_root(sys.argv[1])
ats = b.read_ats(root)
fact, plan = b.read_genconsum(root)
ips_of = {c: v[1] for c, v in sh.R.items()}


def ips_sum(frame, ips, min_count=1):
    cols = [c for c in frame.columns if ips_of.get(c) == ips]
    return frame[cols].sum(axis=1, min_count=min_count if min_count else 1), cols


if "cover" in what:
    print("== ATS coverage per code")
    names = ats.attrs.get("names", {})
    out = []
    for c in ats.columns:
        s = ats[c].dropna()
        out.append((c, names.get(c, "")[:32], s.index.min(), s.index.max(), len(s),
                    int((s <= 0).sum()), round(s.mean(), 1)))
    print(pd.DataFrame(out, columns=["code", "name", "first", "last", "hours", "nonpos", "mean_mw"]).to_string())
    print("codes in ATS not in hierarchy:", sorted(set(ats.columns) - set(sh.R)))
    print("codes in hierarchy not in ATS:", sorted(set(sh.R) - set(ats.columns)))

if "days" in what:
    print("== share of calendar days present (any code, median over codes), per year and weekday")
    day = ats.notna().groupby(ats.index.normalize()).any().sum(axis=1) >= 30
    d = pd.DataFrame({"present": day})
    d["year"] = d.index.year
    d["wd"] = d.index.weekday
    full = pd.date_range(d.index.min(), d.index.max(), freq="D")
    dd = pd.Series(day).reindex(full).fillna(False)
    t = pd.DataFrame({"p": dd.astype(float), "year": full.year, "wd": full.weekday})
    print(t.pivot_table(index="year", columns="wd", values="p", aggfunc="mean").round(2))
    print("hours per present day (min, max):", int(ats.notna().groupby(ats.index.normalize()).sum().max().max()))

if "align" in what:
    print("== hour alignment: corr of d(ATS sum of IPS) and d(SO fact) for shifts of ATS by k hours (2019-2024 working days)")
    for ips in ("CENTER", "VOLGA", "URAL", "SIBERIA"):
        s, _ = ips_sum(ats, ips)
        f = fact[f"IPS_{ips}"]
        res = {}
        for k in range(-3, 4):
            j = pd.concat([s.shift(k, freq="h").rename("x"), f.rename("f")], axis=1).dropna()
            j = j.loc["2019-01-01":"2024-12-31"]
            res[k] = round(float(np.corrcoef(j["x"].diff().dropna(), j["f"].diff().dropna())[0, 1]), 4)
        print(ips, res)

if "sums" in what:
    print("== ratio sum(ATS subjects of IPS) / SO IPS fact: mean per year (working days only, all subjects present)")
    rows = {}
    for ips in list(sh.IPS):
        s, cols = ips_sum(ats, ips)
        n = ats[cols].notna().sum(axis=1)
        ok = n == len(cols)
        f = fact[f"IPS_{ips}"]
        r = (s[ok] / f.reindex(s.index)[ok]).dropna()
        rows[ips] = r.groupby(r.index.year).agg(["mean", "std"]).round(4).stack()
        print(ips, "n subjects", len(cols), "hours with all present", int(ok.sum()))
    print(pd.DataFrame(rows).T.to_string())
    print("== UES: sum of ATS subjects / sum of SO IPS facts, by year (working days, all subjects present)")
    k6 = ["CENTER", "SOUTH", "VOLGA", "URAL", "NORTHWEST", "SIBERIA"]
    core = [c for c in ats.columns if c in sh.R and ats[c].first_valid_index() < pd.Timestamp("2024-01-01")]
    s6 = ats[core].sum(axis=1, min_count=len(core))
    f6 = fact[[f"IPS_{k}" for k in k6]].sum(axis=1, min_count=6)
    r = (s6 / f6.reindex(s6.index)).dropna()
    print("core (67 subjects) / 6 IPS:", r.groupby(r.index.year).agg(["mean", "std"]).round(4).T.to_dict())
    allc = [c for c in ats.columns if c in sh.R]
    s7 = ats[allc].sum(axis=1, min_count=len(allc))
    f7 = fact[[f"IPS_{k}" for k in sh.IPS]].sum(axis=1, min_count=7)
    r = (s7 / f7.reindex(s7.index)).dropna()
    print("all (74 subjects) / 7 IPS:", r.groupby(r.index.year).agg(["mean", "std"]).round(4).T.to_dict())

if "member" in what:
    # Move each subject to every other IPS in turn. The right assignment keeps the ratio
    # sum(ATS)/SO fact of both IPS nearly constant, so the sum of squared relative spread is lowest.
    for lo, hi in (("2022-01-01", "2024-12-31"), ("2025-01-01", "2026-09-30")):
        ks = [k for k in sh.IPS if k != "EAST"] if lo.startswith("2022") else list(sh.IPS)
        cols = [c for c in ats.columns if c in sh.R]
        x = ats.loc[lo:hi, cols].dropna(axis=1, thresh=5000)
        x = x.dropna()
        F = pd.DataFrame({k: fact[f"IPS_{k}"] for k in ks}).reindex(x.index).dropna()
        x = x.reindex(F.index)
        day = x.index.normalize()
        # daily means remove hour-of-day effects; the ratio is then compared day by day
        xd, Fd = x.groupby(day).mean(), F.groupby(day).mean()

        def cost(assign):
            c = 0.0
            for k in ks:
                mem = [r for r in xd.columns if assign[r] == k]
                if not mem:
                    return 1e9
                r = xd[mem].sum(axis=1) / Fd[k]
                c += (r.std() / r.mean()) ** 2
            return c

        base = {r: sh.R[r][1] for r in xd.columns}
        c0 = cost(base)
        print(f"== membership by swap test, working days {lo[:4]}-{hi[:4]}, days {len(xd)}, base cost {c0:.6f}")
        rows = []
        for r in xd.columns:
            alt = {}
            for k in ks:
                a2 = dict(base)
                a2[r] = k
                alt[k] = cost(a2)
            best = min(alt, key=alt.get)
            rows.append((r, sh.R[r][0][:30], base[r], best, round(alt[best] / c0, 3),
                         round(alt[base[r]] / c0, 3), round(float(xd[r].mean()))))
        df = pd.DataFrame(rows, columns=["code", "name", "assigned", "best", "cost_best/base", "cost_assigned/base", "mean_mw"])
        print(df[df["assigned"] != df["best"]].sort_values("cost_best/base").to_string())

if "plan" in what:
    pl = b.read_plan(root)
    print("plan frame:", None if pl is None else pl.shape)
    if pl is not None:
        print("plan codes not in ATS:", sorted(set(pl.columns) - set(ats.columns)))
        print("ATS codes without plan:", sorted(set(ats.columns) - set(pl.columns)))
        com = [c for c in ats.columns if c in pl.columns]
        r = (ats[com] / pl[com].reindex(ats.index)).dropna(how="all")
        print("median ratio fact/plan per code:", r.median().round(3).to_dict())
        print("plan coverage: first/last", pl.index.min(), pl.index.max(), "weekend share", float(pl.index.weekday.isin([5, 6]).mean()))
