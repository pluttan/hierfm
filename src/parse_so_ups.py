#!/usr/bin/env python3
"""Turn the raw files of download_so_ups.py into tidy tables.

Output (data/raw/so_ups/tables/, csv.gz, time = Moscow time, hour-beginning label):
  ues_hourly.csv.gz       node, ts, fact_mw, plan_mw, gen_fact_mw, gen_plan_mw   (SO UPS, GenConsum)
  regions_fact.csv.gz     code, region, ts, mw                                    (ATS fact_region)
  regions_plan.csv.gz     zone, code, region, ts, mw                              (SO UPS day-ahead plan)

Needs numpy/pandas and xlrd (old .xls files of ATS); run it with the project venv on the mac:
    nice venv/bin/python -u src/parse_so_ups.py
"""

import datetime as dt
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

ROOT = Path(os.environ.get("SO_UPS_ROOT", Path(__file__).resolve().parents[1] / "data/raw/so_ups"))
OUT = ROOT / "tables"
NODE = {1: "PZ1", 2: "PZ2", 8: "PZ8", 530000: "IPS_CENTER", 550000: "IPS_SOUTH",
        600000: "IPS_VOLGA", 610000: "IPS_SIBERIA", 630000: "IPS_URAL",
        840000: "IPS_NORTHWEST", 540000: "IPS_EAST"}
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def parse_genconsum():
    rows = []
    for f in sorted((ROOT / "br").glob("genconsum*_*.json")):
        for blk in json.loads(f.read_text()):
            key = blk["m_Item1"]
            if key is None:
                continue
            for r in blk["m_Item2"]:
                rows.append((NODE[key], r["M_DATE"][:10], r["INTERVAL"], r["E_USE_FACT"],
                             r["E_USE_PLAN"], r["GEN_FACT"], r["GEN_PLAN"]))
    df = pd.DataFrame(rows, columns=["node", "date", "hour", "fact_mw", "plan_mw",
                                     "gen_fact_mw", "gen_plan_mw"])
    df["ts"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"], unit="h")
    df = df.drop(columns=["date", "hour"]).drop_duplicates(["node", "ts"], keep="last")
    return df.sort_values(["node", "ts"])[["node", "ts", "fact_mw", "plan_mw",
                                           "gen_fact_mw", "gen_plan_mw"]]


def xlsx_rows(path):
    """Minimal reader of the first sheet of an xlsx: list of row lists."""
    with zipfile.ZipFile(path) as z:
        sst = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS):
                sst.append("".join(t.text or "" for t in si.iter("{%s}t" % NS["m"])))
        sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    out = []
    for row in sheet.find("m:sheetData", NS):
        vals = {}
        for c in row:
            col = re.match(r"[A-Z]+", c.get("r")).group(0)
            v = c.find("m:v", NS)
            if v is None:
                continue
            vals[col] = sst[int(v.text)] if c.get("t") == "s" else float(v.text)
        out.append([vals.get(k) for k in "ABC"])
    return out


def xls_rows(path):
    import xlrd
    sh = xlrd.open_workbook(path).sheet_by_index(0)
    return [sh.row_values(i)[:3] for i in range(sh.nrows)]


def parse_ats_file(path):
    rows = xlsx_rows(path) if path.suffix == ".xlsx" else xls_rows(path)
    region = None
    out = []
    for r in rows:
        if r[0] == "Субъект РФ:":
            region = r[1]
        elif isinstance(r[1], float) and isinstance(r[0], str) and re.fullmatch(r"\d\d\.\d\d\.\d{4}", r[0]) \
                and isinstance(r[2], float):
            out.append((r[0], int(r[1]), r[2]))
    return region, out


def parse_ats():
    recs, names = [], {}
    files = sorted((ROOT / "ats" / "xlsx").glob("*/*"))
    for i, f in enumerate(files):
        code = int(f.name.split("_")[1])
        region, data = parse_ats_file(f)
        names[code] = region
        recs.extend((code, d, h, v) for d, h, v in data)
        if i % 500 == 0:
            print("ats", i, len(files), flush=True)
    df = pd.DataFrame(recs, columns=["code", "date", "hour", "mw"])
    df["ts"] = pd.to_datetime(df["date"], format="%d.%m.%Y") + pd.to_timedelta(df["hour"] - 1, unit="h")
    df["region"] = df["code"].map(names)
    df = df.drop_duplicates(["code", "ts"], keep="last")
    return df.sort_values(["code", "ts"])[["code", "region", "ts", "mw"]]


def parse_plan():
    rows = []
    for zone, pat in (("Z1", "forecast_subrf_20*.json"), ("Z2", "forecast_subrf_z2_20*.json")):
        for f in sorted((ROOT / "br").glob(pat)):
            for day, lst in json.loads(f.read_text()).items():
                for r in lst or []:
                    if not isinstance(r, list) or len(r) < 25:
                        continue
                    code = int(r[25]) if len(r) > 25 and r[25] is not None else -1
                    for h in range(24):
                        rows.append((zone, code, r[0], f"{day} {h:02d}:00", r[1 + h]))
    df = pd.DataFrame(rows, columns=["zone", "code", "region", "ts", "mw"])
    df["ts"] = pd.to_datetime(df["ts"])
    return df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1:] or ["ues", "ats", "plan"]
    if "ues" in what:
        d = parse_genconsum()
        d.to_csv(OUT / "ues_hourly.csv.gz", index=False)
        print("ues", d.shape, flush=True)
    if "ats" in what:
        d = parse_ats()
        d.to_csv(OUT / "regions_fact.csv.gz", index=False)
        print("ats", d.shape, flush=True)
    if "plan" in what:
        d = parse_plan()
        d.to_csv(OUT / "regions_plan.csv.gz", index=False)
        print("plan", d.shape, flush=True)


if __name__ == "__main__":
    main()
