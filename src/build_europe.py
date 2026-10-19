#!/usr/bin/env python3
"""Сборка европейской иерархии в формат hierfm (load.parquet, hierarchy.json, temp.parquet).

Вход: каталог raw с файлами download_europe.py (smard, energinet, odre, energy_charts, openmeteo).
Выход: каталог процессед по формату hierfm/data.py.

Иерархия по умолчанию (extended=False), три уровня: EUROPE -> страна -> зона.
    FR: 12 регионов ODRE, DE: 4 зоны оператора сети (SMARD), DK: DK1 и DK2 (Energinet).
Расширение (extended=True): к ним добавляются страны energy-charts как листья уровня "страна".

Правила.
- Шаг часовой, время UTC. Подчасовые ряды усредняются (МВт), метка часа это его начало.
- Узел со своими зонами никогда не берётся из другого источника: сумма зон и есть страна.
- ODRE: данные "définitives" главнее "temps réel" там, где совпадают.
- Нули и отрицательные значения считаются пропусками. Дыры до max_gap часов заполняются
  линейной интерполяцией. Дыры до long_gap часов заполняются средним значением того же часа
  неделей раньше и неделей позже. Остальное остаётся пропуском, окно режется по общему покрытию.
- По умолчанию из расширения исключены ME, MK (нули, пропуски) и IE (дыры до 99 часов).

    python3 src/build_europe.py RAW_DIR OUT_DIR [--extended] [--start 2020-01-01] [--end ...]
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from hierfm import hierarchy as hier

FR_REG = {"84": "ARA", "27": "BFC", "53": "BRE", "24": "CVL", "44": "GES", "32": "HDF",
          "28": "NOR", "75": "NAQ", "76": "OCC", "52": "PDL", "93": "PAC", "11": "IDF"}
DE_ZONES = ["50Hertz", "Amprion", "TenneT", "TransnetBW"]
DK_ZONES = ["DK1", "DK2"]
ZONED = ("FR", "DE", "DK")
# Страны energy-charts, которые по умолчанию не берём: нули и пропуски (см. docs/research/07).
EXCLUDE_DEFAULT = ("ME", "MK", "IE")
# Часовой пояс, смещение от UTC в зимнее время (для суточных профилей).
TZ = {"PT": 0, "IE": 0, "GR": 2, "FI": 2, "EE": 2, "LV": 2, "LT": 2, "BG": 2, "RO": 2}
# Координаты: те же, что у download_europe.POINTS (погода берётся оттуда по имени файла).


def hourly(s):
    """Ряд с метками времени (UTC, любой шаг) в средний МВт за час."""
    s = s.dropna()
    s.index = s.index.floor("h")
    return s.groupby(level=0).mean()


def read_two_col(path):
    df = pd.read_csv(path, parse_dates=[0])
    s = df.set_index(df.columns[0])[df.columns[1]].astype(float)
    s.index = pd.DatetimeIndex(s.index).tz_convert("UTC") if s.index.tz else s.index.tz_localize("UTC")
    return hourly(s)


def read_smard(raw):
    return {z: read_two_col(raw / "smard" / f"smard_load_{z}.csv") for z in DE_ZONES}


def read_energinet(raw):
    df = pd.read_csv(raw / "energinet" / "energinet_consumption_settlement.csv",
                     usecols=["HourUTC", "PriceArea", "GrossConsumptionMWh"])
    df["HourUTC"] = pd.to_datetime(df["HourUTC"], utc=True)
    out = {}
    for a in DK_ZONES:
        d = df[df.PriceArea == a].set_index("HourUTC")["GrossConsumptionMWh"].astype(float)
        out[a] = hourly(d)
    return out


def read_odre(raw):
    """Регионы Франции. 'def' перекрывает 'tr' в общих часах."""
    parts = {}
    for kind in ("tr", "def"):
        p = raw / "odre" / f"odre_regional_{kind}.csv"
        df = pd.read_csv(p, encoding="utf-8-sig", dtype={"code_insee_region": str},
                         usecols=["date_heure", "code_insee_region", "consommation"])
        df["date_heure"] = pd.to_datetime(df["date_heure"], utc=True)
        for code, name in FR_REG.items():
            d = df[df.code_insee_region == code].set_index("date_heure")["consommation"].astype(float)
            parts.setdefault(name, {})[kind] = hourly(d)
    return {n: v["def"].combine_first(v["tr"]) for n, v in parts.items()}


def read_countries(raw, skip):
    out = {}
    for p in sorted((raw / "energy_charts").glob("load_*.csv")):
        c = p.stem[5:].upper()
        if c not in ZONED and c not in skip:
            out[c] = read_two_col(p)
    return out


def read_temp(raw, kind, name):
    p = raw / "openmeteo" / kind / f"t2m_{name}.csv"
    return read_two_col(p) if p.exists() else None


def build(raw_dir, out_dir, extended=False, start=None, end=None, max_gap=6, long_gap=168, exclude=EXCLUDE_DEFAULT):
    raw, out = Path(raw_dir) / "europe" if (Path(raw_dir) / "europe").is_dir() else Path(raw_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cols = {}
    for n, s in read_odre(raw).items():
        cols[f"FR_{n}"] = s
    for z, s in read_smard(raw).items():
        cols[f"DE_{z}"] = s
    for z, s in read_energinet(raw).items():
        cols[f"DK_{z}"] = s
    zone_leaves = list(cols)
    countries = read_countries(raw, exclude) if extended else {}
    cols.update(countries)

    df = pd.DataFrame(cols).sort_index()
    df = df.where(df > 0)                                    # нули и минусы это сбой данных
    df = df.asfreq("h")
    gaps_before = df.isna().sum()
    df = df.interpolate(limit=max_gap, limit_area="inside")
    weekly = (df.shift(168) + df.shift(-168)) / 2
    weekly = weekly.fillna(df.shift(168)).fillna(df.shift(-168))
    for k in df:
        na = df[k].isna() & df[k].ffill().notna() & df[k].bfill().notna()   # только внутренние дыры
        grp = (na != na.shift()).cumsum()
        size = na.groupby(grp).transform("sum")
        fill = na & (size <= long_gap)
        df.loc[fill, k] = weekly.loc[fill, k]
    lo, hi = df.dropna().index[[0, -1]]                      # общее окно: все листья в наличии
    # внутри окна оставшиеся дыры длиннее max_gap не режем, а выдаём в отчёт
    if start:
        lo = max(lo, pd.Timestamp(start, tz="UTC"))
    if end:
        hi = min(hi, pd.Timestamp(end, tz="UTC"))
    df = df.loc[lo:hi]
    df.index.name = "ds"
    left = df.isna().sum()
    # берём самый длинный отрезок, где нет пропусков во всех листьях
    ok = df.notna().all(axis=1)
    if not ok.all():
        grp = (ok != ok.shift()).cumsum()
        runs = ok[ok].groupby(grp[ok]).size()
        best = runs.idxmax()
        keep = ok & (grp == best)
        df = df.loc[keep.idxmax():keep[keep].index[-1]]
        left = df.isna().sum()

    # иерархия
    recs = [{"id": "EUROPE", "parent": None, "attrs": {}}]
    for c in ZONED:
        recs.append({"id": c, "parent": "EUROPE", "attrs": {"tz": 1}})
    for c in countries:
        recs.append({"id": c, "parent": "EUROPE", "attrs": {"tz": TZ.get(c, 1)}})
    for z in zone_leaves:
        recs.append({"id": z, "parent": z.split("_")[0], "attrs": {"tz": 1, "source": _source(z)}})
    h = hier.from_nodes(["europe", "country", "zone"], recs)
    h.nodes  # порядок проверен from_nodes
    bottom = h.bottom
    df = df[bottom]
    df.to_parquet(out / "load.parquet")
    hier.save(h, out / "hierarchy.json")

    # погода: реанализ как фактическая температура, архив прогнозов как прогноз
    def tname(b):
        return b.split("_", 1)[1] if b.startswith("DK_") else b
    t_act, t_fc = {}, {}
    for b in bottom:
        a = read_temp(raw, "archive", tname(b))
        f = read_temp(raw, "histfc", tname(b))
        if a is not None:
            t_act[b] = a
        if f is not None:
            t_fc[b] = f
    if t_act:
        pd.DataFrame(t_act).reindex(df.index).interpolate(limit=max_gap).rename_axis("ds").to_parquet(out / "temp.parquet")
    if t_fc:
        pd.DataFrame(t_fc).reindex(df.index).rename_axis("ds").to_parquet(out / "temp_forecast.parquet")

    # сходимость: сумма зон против опубликованного национального ряда (справка, не вход)
    meta = {"extended": extended, "n_nodes": h.n, "n_bottom": h.m, "levels": h.levels,
            "first": str(df.index[0]), "last": str(df.index[-1]), "hours": len(df),
            "interpolated_hours_per_leaf": {k: int(gaps_before[k] - left.get(k, 0)) for k in bottom
                                            if gaps_before[k] != left.get(k, 0)},
            "temp_leaves": len(t_act), "temp_forecast_leaves": len(t_fc),
            "temp_forecast_first": str(pd.DataFrame(t_fc).index.min()) if t_fc else None}
    (out / "build_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"europe: {out} · {h.m} листьев · {h.n} узлов · {df.index[0]} .. {df.index[-1]} · {len(df)} ч", flush=True)
    return meta


def _source(z):
    return {"FR": "ODRE", "DE": "SMARD", "DK": "Energinet"}[z.split("_")[0]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--extended", action="store_true")
    ap.add_argument("--start")
    ap.add_argument("--end")
    a = ap.parse_args()
    build(a.raw_dir, a.out_dir, a.extended, a.start, a.end)
