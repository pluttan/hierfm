#!/usr/bin/env python3
"""Checks of the SO UPS / ATS tables: coverage, gaps, sums, alignment of hours.

    nice venv/bin/python -u src/check_so_ups.py ues      # SO UPS: price zones and IPS
    nice venv/bin/python -u src/check_so_ups.py ats      # ATS: subjects of the RF
    nice venv/bin/python -u src/check_so_ups.py plan     # SO plan by subject
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

T = Path(os.environ.get("SO_UPS_ROOT", Path(__file__).resolve().parents[1] / "data/raw/so_ups")) / "tables"
pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 200)


def ues():
    d = pd.read_csv(T / "ues_hourly.csv.gz", parse_dates=["ts"])
    for c in ("fact_mw", "plan_mw", "gen_fact_mw", "gen_plan_mw"):
        print("zeros in", c, int((d[c] == 0).sum()))
    d[["fact_mw", "plan_mw", "gen_fact_mw", "gen_plan_mw"]] = d[["fact_mw", "plan_mw", "gen_fact_mw", "gen_plan_mw"]].replace(0, np.nan)
    d = d[d["ts"] <= pd.Timestamp.now().floor("h")]
    print("== coverage by node (fact_mw), zeros treated as missing")
    g = d.groupby("node")
    print(pd.DataFrame({
        "first_ts": g["ts"].min(), "last_ts": g["ts"].max(), "rows": g.size(),
        "nan_fact": g["fact_mw"].apply(lambda s: s.isna().sum()),
        "first_valid": g.apply(lambda x: x.loc[x["fact_mw"].notna(), "ts"].min()),
        "last_valid": g.apply(lambda x: x.loc[x["fact_mw"].notna(), "ts"].max()),
        "zeros": g["fact_mw"].apply(lambda s: (s == 0).sum())}))
    f = d.pivot(index="ts", columns="node", values="fact_mw")
    p = d.pivot(index="ts", columns="node", values="plan_mw")
    f = f.loc["2015-01-01":]
    ips6 = ["IPS_CENTER", "IPS_SOUTH", "IPS_VOLGA", "IPS_URAL", "IPS_NORTHWEST", "IPS_SIBERIA"]
    print("\n== annual mean MW and TWh by node")
    print(f.groupby(f.index.year).mean().round(0).T)
    print("\n== sum checks (fact): relative difference, per year")
    z1 = f[["IPS_CENTER", "IPS_SOUTH", "IPS_VOLGA", "IPS_URAL", "IPS_NORTHWEST"]].sum(axis=1, min_count=5)
    chk = pd.DataFrame({
        "PZ1 - sum5": f["PZ1"] - z1,
        "PZ2 - Siberia": f["PZ2"] - f["IPS_SIBERIA"],
        "UES1 (PZ1+PZ2) - sum6": f["PZ1"] + f["PZ2"] - f[ips6].sum(axis=1, min_count=6),
        "PZ8 - East": f["PZ8"] - f["IPS_EAST"]})
    rel = chk.div(f["PZ1"] + f["PZ2"], axis=0)
    print(chk.groupby(chk.index.year).agg(["mean", "std", "min", "max"]).round(1).T)
    print("\nrelative |diff| quantiles over all hours")
    print(rel.abs().quantile([0.5, 0.9, 0.99, 1.0]).round(5).T)
    print("\n== plan vs fact, MAPE by node and year (SO day-ahead plan as the reference)")
    ape = ((p - f).abs() / f).loc["2015-01-01":]
    print((ape.groupby(ape.index.year).mean() * 100).round(2).T)
    print("\n== steps: largest hour-to-hour relative jumps per node")
    for c in f.columns:
        s = f[c].dropna()
        j = (s.diff().abs() / s.shift()).sort_values(ascending=False).head(3)
        print(c, [(str(i), round(v, 3)) for i, v in j.items()])
    print("\n== monthly mean of (sum of 5 IPS - PZ1), MW, to find changes of composition")
    m = (-chk["PZ1 - sum5"]).resample("MS").mean().round(0)
    print(m.to_string())
    print("\n== monthly mean of East and Siberia around 2018-2019, MW")
    print(f[["IPS_EAST", "IPS_SIBERIA"]].loc["2018-06":"2019-06"].resample("MS").mean().round(0))
    print("\n== monthly mean of South, 2022-2025, MW")
    print(f[["IPS_SOUTH"]].loc["2022-01":"2025-03"].resample("MS").mean().round(0).T.to_string())
    print("\n== monthly mean of Russian UES, MW (first zone + Siberia + East)")
    tot = f["PZ1"] + f["PZ2"] + f["IPS_EAST"]
    print(tot.resample("YS").mean().round(0))
    # hour offset between the sum of 7 IPS and PZ1+PZ2 is zero by construction; the label
    # of the hour is checked against the daily shape (minimum at night, MSK)
    prof = f["PZ1"].groupby(f.index.hour).mean()
    print("\n== mean profile of PZ1 by label hour (MSK):", prof.round(0).to_dict())


def ats():
    d = pd.read_csv(T / "regions_fact.csv.gz", parse_dates=["ts"])
    print("subjects:", d["code"].nunique(), "rows:", len(d), "span:", d["ts"].min(), d["ts"].max())
    g = d.groupby(["code", "region"])
    cov = g["ts"].agg(["min", "max", "size"])
    full = pd.date_range("2015-01-01", d["ts"].max().normalize() + pd.Timedelta(hours=23), freq="h")
    cov["expected_in_span"] = [((full >= a) & (full <= b)).sum() for a, b in zip(cov["min"], cov["max"])]
    cov["missing_in_span"] = cov["expected_in_span"] - cov["size"]
    print(cov.to_string())
    w = d.pivot(index="ts", columns="code", values="mw")
    print("\ncoverage per year: number of subjects with data, mean share of hours present")
    yr = w.groupby(w.index.year)
    print(pd.DataFrame({"hours": yr.size(), "subjects": yr.apply(lambda x: x.notna().any().sum()),
                        "share_present": yr.apply(lambda x: x.notna().mean().mean())}).round(3))
    days = d.assign(day=d["ts"].dt.normalize()).groupby("day")["code"].nunique()
    print("\n== days present in ATS files (any subject):", len(days), "from", days.index.min().date(), "to", days.index.max().date())
    print("by weekday (0=Mon):", days.groupby(days.index.dayofweek).size().to_dict())
    print("days per year:", days.groupby(days.index.year).size().to_dict())
    allwd = pd.bdate_range(days.index.min(), days.index.max())
    miss = allwd.difference(days.index)
    print("Mon-Fri days not present:", len(miss), " per year:", pd.Series(miss.year).value_counts().sort_index().to_dict())
    print("hours per day (min, max):", d.groupby(["code", d["ts"].dt.normalize()]).size().agg(["min", "max"]).to_dict())
    print("longest run of consecutive absent calendar days, per year:")
    cal = pd.date_range(days.index.min(), days.index.max())
    absent = pd.Series(~cal.isin(days.index), index=cal)
    runs = (absent != absent.shift()).cumsum()
    ln = absent.groupby(runs).agg(["first", "size"])
    ln["start"] = absent.index.to_series().groupby(runs).first().values
    ln = ln[ln["first"]]
    print(ln.groupby(ln["start"].dt.year)["size"].max().to_dict())
    print("\nnegative or zero values:", int((d["mw"] <= 0).sum()))
    bad = d[d["mw"] <= 0]
    print(bad.groupby(["code", "region"]).size().sort_values(ascending=False).head(10).to_dict())
    print("annual total, GW mean (sum over subjects present):")
    print((w.groupby(w.index.year).sum(min_count=1).sum(axis=1) / w.groupby(w.index.year).size() / 1000).round(1))


def plan():
    d = pd.read_csv(T / "regions_plan.csv.gz", parse_dates=["ts"])
    print(d.groupby("zone").agg(subjects=("code", "nunique"), first=("ts", "min"), last=("ts", "max"), rows=("mw", "size")))
    print(d.groupby(["zone", "code", "region"]).size().unstack(0).head(100))


def cross():
    """ATS subjects against SO UPS series: hour alignment, time zone, residual of IPS."""
    from so_ups_hierarchy import R
    a = pd.read_csv(T / "regions_fact.csv.gz", parse_dates=["ts"])
    u = pd.read_csv(T / "ues_hourly.csv.gz", parse_dates=["ts"]).replace(0, np.nan)
    w = a.pivot(index="ts", columns="code", values="mw")
    uf = u.pivot(index="ts", columns="node", values="fact_mw")
    up = u.pivot(index="ts", columns="node", values="plan_mw")
    print("== time zone: hour of the minimum and of the maximum of the mean day profile (label hour as published)")
    for code in (46, 22, 65, 50, 25, 5, 8):
        if code in w:
            s = w[code].dropna().loc["2025-01-01":] if code in (5, 8) else w[code].dropna().loc["2023-01-01":]
            pr = s.groupby(s.index.hour).mean()
            print(code, R[code][0], "UTC+%d" % R[code][2], "min at", int(pr.idxmin()), "max at", int(pr.idxmax()))
    print("\n== hour alignment: correlation of hourly changes, ATS sum of subjects of IPS Center vs SO IPS Center")
    ctr = [c for c, v in R.items() if v[1] == "CENTER" and c in w]
    s_ats = w[ctr].sum(axis=1, min_count=len(ctr)).loc["2022-01-01":"2024-12-31"]
    so = uf["IPS_CENTER"].loc["2022-01-01":"2024-12-31"]
    for lag in (-2, -1, 0, 1, 2):
        x = s_ats.diff()
        y = so.shift(-lag).diff()
        print("shift of SO by", lag, "hours: corr", round(float(x.corr(y)), 4),
              " mean |diff| MW", round(float((s_ats - so.shift(-lag)).abs().mean()), 1))
    print("\n== residual = SO IPS - sum of ATS subjects, MW, per year (subjects of a year with all hours present)")
    rows = []
    for k in ("CENTER", "NORTHWEST", "VOLGA", "SOUTH", "URAL", "SIBERIA", "EAST"):
        cs = [c for c, v in R.items() if v[1] == k and c in w]
        for y in range(2015, 2027):
            if not (w.index.year == y).any():
                continue
            sub = w.loc[str(y), cs]
            full = sub.dropna(axis=1, how="all")
            if full.shape[1] == 0:
                continue
            ssum = full.sum(axis=1, min_count=full.shape[1])
            res = uf["IPS_" + k].loc[str(y)] - ssum
            rows.append((k, y, full.shape[1], len(cs), round(float(res.mean()), 0), round(float(res.std()), 0),
                         round(float((res / uf["IPS_" + k].loc[str(y)]).mean() * 100), 2),
                         round(float((res / uf["IPS_" + k].loc[str(y)]).min() * 100), 2),
                         round(float((res / uf["IPS_" + k].loc[str(y)]).max() * 100), 2)))
    print(pd.DataFrame(rows, columns=["IPS", "year", "n_subj_present", "n_subj_listed", "mean_MW", "sd_MW",
                                      "mean_%", "min_%", "max_%"]).to_string())
    print("\n== SO plan by subject (64) sum against SO plan by IPS")
    p = pd.read_csv(T / "regions_plan.csv.gz", parse_dates=["ts"])
    p["code"] = p["code"].replace({76: 76200})  # Zabaykalsky: code 76 at SO, 76200 at ATS
    pz = p[p["zone"] == "Z1"].pivot_table(index="ts", columns="code", values="mw", aggfunc="last")
    print("codes in plan Z1:", pz.shape[1], "hours:", pz.shape[0], "first", pz.index.min(), "last", pz.index.max())
    miss = [c for c in R if c not in pz.columns and R[c][1] != "EAST"]
    print("ATS codes without plan series:", [(c, R[c][0]) for c in miss])
    pm = {k: [c for c, v in R.items() if v[1] == k and c in pz.columns] for k in ("CENTER", "NORTHWEST", "VOLGA", "SOUTH", "URAL", "SIBERIA")}
    for k, cs in pm.items():
        pl = pz[cs].sum(axis=1, min_count=len(cs))
        ref = up["IPS_" + k].reindex(pl.index)
        fr = uf["IPS_" + k].reindex(pl.index)
        print(k, "plan(subj sum)/plan(IPS) mean", round(float((pl / ref).mean()), 4),
              " MAPE(plan subj sum vs fact IPS) %", round(float(((pl - fr).abs() / fr).mean() * 100), 2))
    print("\n== subject level: MAPE of SO plan against ATS fact by year (SO plan in MW, ATS fact in MW)")
    f = w.reindex(pz.index)
    ape = ((pz - f).abs() / f).dropna(axis=1, how="all")
    print("median over subjects of the yearly MAPE, %:")
    print((ape.groupby(ape.index.year).mean() * 100).median(axis=1).round(2))
    ratio = (pz.groupby(pz.index.year).mean() / f.groupby(f.index.year).mean()).dropna(axis=1, how="all")
    print("median over subjects of mean(plan)/mean(fact) by year:")
    print(ratio.median(axis=1).round(3))
    print("quantiles of the ratio over subjects (all years):")
    print(ratio.stack().quantile([0.05, 0.25, 0.5, 0.75, 0.95]).round(3).to_dict())
    k = pd.Series({y: pz.index.year.isin([y]).sum() for y in sorted(set(pz.index.year))})
    sc = pz.copy()
    for y in sorted(set(pz.index.year)):
        m = pz.index.year == y
        sc.loc[m] = pz.loc[m] / ratio.loc[y].reindex(pz.columns).values
    ape2 = ((sc - f).abs() / f).dropna(axis=1, how="all")
    print("median over subjects of the yearly MAPE after scaling the plan by its yearly mean ratio, %:")
    print((ape2.groupby(ape2.index.year).mean() * 100).median(axis=1).round(2))


if __name__ == "__main__":
    for w in sys.argv[1:] or ["ues"]:
        globals()[w]()
