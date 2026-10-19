import numpy as np, pandas as pd, sys
f="/Volumes/pr/bmstu/papers/soglasovanie-prognozov/data/raw/opsd/time_series_60min_singleindex_2020-10-06.csv"
hdr=pd.read_csv(f,nrows=0).columns
cols=["utc_timestamp"]+[c for c in hdr if ("load_actual_entsoe_transparency" in c or "load_forecast_entsoe_transparency" in c)]
df=pd.read_csv(f,usecols=cols,parse_dates=["utc_timestamp"]).set_index("utc_timestamp")
nodes=sorted({c.split("_load_")[0] for c in df.columns})
err={}
for n in nodes:
    a=df.get(f"{n}_load_actual_entsoe_transparency"); fc=df.get(f"{n}_load_forecast_entsoe_transparency")
    if a is None or fc is None: continue
    e=(fc-a)/a
    e=e[(a>0)]
    err[n]=e
E=pd.DataFrame(err)
E=E.clip(-0.5,0.5)
print("nodes with both series:",E.shape[1]); cov=E.notna().mean().round(2)
print("coverage share:",dict(cov[cov>0]))
print("MAPE %:",(E.abs().mean()*100).round(2).to_dict())
E.to_pickle("/tmp/cov05sim/E.pkl")
