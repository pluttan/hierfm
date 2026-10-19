#!/usr/bin/env python3
"""Проверка иерархии Европы: покрытие рядов и сходимость сумм.

Читает файлы из download_europe.py, переводит всё в часовой шаг (среднее МВт),
строит матрицу суммирования и сравнивает сумму зон с национальным рядом.
Только стандартная библиотека. Запуск: nice python3.12 -u check_europe_hierarchy.py RAW_DIR [OUT_JSON]
"""
import csv
import datetime as dt
import json
import math
import os
import sys
from collections import defaultdict

FR_REG = {  # код INSEE -> короткое имя
    "84": "ARA", "27": "BFC", "53": "BRE", "24": "CVL", "44": "GES", "32": "HDF",
    "28": "NOR", "75": "NAQ", "76": "OCC", "52": "PDL", "93": "PAC", "11": "IDF",
}
DE_ZONES = ["50Hertz", "Amprion", "TenneT", "TransnetBW"]


def hour_of(ts):
    return ts[:13]  # 'YYYY-MM-DDTHH'


def to_hourly(pairs):
    acc = defaultdict(lambda: [0.0, 0])
    for ts, v in pairs:
        if v is None:
            continue
        a = acc[hour_of(ts)]
        a[0] += v
        a[1] += 1
    return {h: s / n for h, (s, n) in acc.items()}


def read_two_col(path):
    out = []
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            try:
                out.append((row[0], float(row[1])))
            except (ValueError, IndexError):
                pass
    return out


def load_all(raw):
    s = {}
    for z in ["DE"] + DE_ZONES:
        s[f"SMARD_{z}"] = to_hourly(read_two_col(f"{raw}/smard/smard_load_{z}.csv"))
    # Energinet
    dk = defaultdict(list)
    with open(f"{raw}/energinet/energinet_consumption_settlement.csv") as f:
        for r in csv.DictReader(f):
            if r["GrossConsumptionMWh"]:
                dk[r["PriceArea"]].append((r["HourUTC"][:13] if "T" in r["HourUTC"] else r["HourUTC"], float(r["GrossConsumptionMWh"])))
    for a, lst in dk.items():
        s[f"ENERGINET_{a}"] = {h: v for h, v in lst}
    # ODRE: def + tr, определённые данные важнее
    reg = defaultdict(list)
    nat = []
    for kind in ("def", "tr"):
        p = f"{raw}/odre/odre_regional_{kind}.csv"
        if os.path.exists(p):
            with open(p, encoding="utf-8-sig") as f:
                for r in csv.DictReader(f, delimiter=","):
                    if r["consommation"] not in ("", None):
                        reg[(kind, r["code_insee_region"])].append((r["date_heure"], float(r["consommation"])))
        p = f"{raw}/odre/odre_national_{kind}.csv"
        if os.path.exists(p):
            with open(p, encoding="utf-8-sig") as f:
                for r in csv.DictReader(f):
                    if r["consommation"] not in ("", None):
                        nat.append((kind, r["date_heure"], float(r["consommation"])))
    for code, name in FR_REG.items():
        d = to_hourly(reg.get(("tr", code), []))
        d.update(to_hourly(reg.get(("def", code), [])))
        s[f"ODRE_{name}"] = d
    dn = to_hourly([(t, v) for k, t, v in nat if k == "tr"])
    dn.update(to_hourly([(t, v) for k, t, v in nat if k == "def"]))
    s["ODRE_FR"] = dn
    # energy-charts
    d = f"{raw}/energy_charts"
    if os.path.isdir(d):
        for fn in sorted(os.listdir(d)):
            if fn.startswith("load_") and fn.endswith(".csv"):
                s["EC_" + fn[5:-4].upper()] = to_hourly(read_two_col(os.path.join(d, fn)))
    return s


def span(d):
    k = sorted(d)
    return (k[0], k[-1], len(k)) if k else (None, None, 0)


def compare(a, b):
    """Сравнение рядов a (сумма зон) и b (национальный): средняя доля, MAPE, максимум."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    rel = [(a[k] - b[k]) / b[k] for k in keys if b[k]]
    mape = sum(abs(x) for x in rel) / len(rel)
    mean = sum(rel) / len(rel)
    srt = sorted(abs(x) for x in rel)
    p99 = srt[int(0.99 * (len(srt) - 1))]
    return {"n_common_hours": len(keys), "mean_rel_diff": mean, "mape": mape, "p99_abs_rel": p99,
            "max_abs_rel": srt[-1]}


def sum_series(parts, require_all=True):
    keys = None
    for p in parts:
        keys = set(p) if keys is None else keys & set(p)
    return {k: sum(p[k] for p in parts) for k in sorted(keys)}


def main():
    raw = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else None
    s = load_all(raw)
    rep = {"spans": {k: span(v) for k, v in s.items()}}
    rep["checks"] = {}
    # 1. Германия: сумма 4 зон против общей нагрузки SMARD и против energy-charts
    de_sum = sum_series([s[f"SMARD_{z}"] for z in DE_ZONES])
    rep["checks"]["DE zones vs SMARD_DE"] = compare(de_sum, s["SMARD_DE"])
    if "EC_DE" in s:
        rep["checks"]["DE SMARD_DE vs EC_DE"] = compare(s["SMARD_DE"], s["EC_DE"])
        rep["checks"]["DE zones vs EC_DE"] = compare(de_sum, s["EC_DE"])
    # 2. Дания
    if "ENERGINET_DK1" in s and "ENERGINET_DK2" in s:
        dk_sum = sum_series([s["ENERGINET_DK1"], s["ENERGINET_DK2"]])
        if "EC_DK" in s:
            rep["checks"]["DK DK1+DK2 vs EC_DK"] = compare(dk_sum, s["EC_DK"])
    # 3. Франция
    fr_sum = sum_series([s[f"ODRE_{n}"] for n in FR_REG.values()])
    rep["checks"]["FR 12 regions vs ODRE_FR"] = compare(fr_sum, s["ODRE_FR"])
    if "EC_FR" in s:
        rep["checks"]["FR 12 regions vs EC_FR"] = compare(fr_sum, s["EC_FR"])
        rep["checks"]["FR ODRE_FR vs EC_FR"] = compare(s["ODRE_FR"], s["EC_FR"])
    # 4. Пропуски в общем окне 3 уровней
    parts = [s[f"SMARD_{z}"] for z in DE_ZONES] + [s["ENERGINET_DK1"], s["ENERGINET_DK2"]] + \
            [s[f"ODRE_{n}"] for n in FR_REG.values()]
    common = set.intersection(*[set(p) for p in parts])
    if common:
        c = sorted(common)
        rep["bottom18_common"] = {"first": c[0], "last": c[-1], "hours": len(c)}
        # доля пропущенных часов внутри окна
        t0 = dt.datetime.strptime(c[0], "%Y-%m-%dT%H")
        t1 = dt.datetime.strptime(c[-1], "%Y-%m-%dT%H")
        total = int((t1 - t0).total_seconds() // 3600) + 1
        rep["bottom18_common"]["missing_share_in_window"] = 1 - len(c) / total
    # матрица суммирования EU-3: 22 узла x 18 листьев
    leaves = [f"FR_{n}" for n in FR_REG.values()] + [f"DE_{z}" for z in DE_ZONES] + ["DK_DK1", "DK_DK2"]
    nodes = [("TOTAL", leaves), ("FR", leaves[:12]), ("DE", leaves[12:16]), ("DK", leaves[16:])] + [(l, [l]) for l in leaves]
    rep["S_shape"] = [len(nodes), len(leaves)]
    rep["S_rank"] = len(leaves)
    txt = json.dumps(rep, indent=1, ensure_ascii=False, default=str)
    if out:
        with open(out, "w") as f:
            f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
