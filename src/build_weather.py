#!/usr/bin/env python3
"""Температурные файлы temp.parquet и temp_forecast.parquet для папки data/processed/<имя>.

Вход: узловые ряды из download_weather.py --aggregate (UTC, градусы Цельсия)
  DIR/node_temperature_archive.csv.gz            факт ERA5-Land
  DIR/node_temperature_prev_ecmwf_ifs025.csv.gz  прогноз IFS 0.25 за 24 часа до часа события
Выход: PROCESSED/temp.parquet и PROCESSED/temp_forecast.parquet.
Формат hierfm/data.py: индекс ds (UTC, шаг час), столбец на узел, значения в градусах Цельсия.

Идентификаторы узлов в таблице погоды (REG_*, IPS_*, UES, DK1) отличаются от идентификаторов
иерархий (R<код АТС>, DK_DK1). Функция rename_map строит соответствие. Столбцы, которых нет
в иерархии, отбрасываются. Лист без своей погоды получает погоду родителя
(RES_* берёт погоду ОЭС, Еврейская АО берёт погоду ОЭС Востока), иначе hierfm.data.load
не сможет заполнить узел.

Запуск (нужны pandas и pyarrow, на маке или кластере):
  python3 src/build_weather.py --hier data/processed/ru/hierarchy.json --kind ru \
      --weather data/raw/weather --out data/processed/ru
  python3 src/build_weather.py --hier data/processed/europe/hierarchy.json --kind eu \
      --weather data/raw/weather --out data/processed/europe
Параметр --check печатает покрытие и не пишет файлы.
"""
import argparse
import csv
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_FILE = "node_temperature_archive.csv.gz"
FORECAST_FILE = "node_temperature_prev_ecmwf_ifs025.csv.gz"

# узел погоды -> код АТС там, где идентификаторы расходятся с автоматическим правилом
# (REG_* находится по русскому имени в src/so_ups_hierarchy.py)
RU_BY_WEATHER_ID = {
    "REG_moscow": 46, "REG_leningrad": 41, "REG_chuvashia": 97, "REG_krasnodar": 3,
    "REG_tyumen": 71, "REG_kemerovo": 32, "REG_sakha_south": 98,
}
# лист иерархии без собственной погоды -> узел, чью погоду он берёт
RU_FALLBACK = {"R99": "IPS_EAST"}
EU_ALIAS = {"DK1": "DK_DK1", "DK2": "DK_DK2"}


def ru_names():
    """Русское имя из so_ups_hierarchy.R -> код АТС."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from so_ups_hierarchy import R  # noqa: PLC0415
    return {v[0]: k for k, v in R.items()}


def rename_map(kind, weather_ids, names):
    """Словарь: идентификатор в таблице погоды -> идентификатор в иерархии.

    names: русское имя узла погоды (node_name или ats_name) -> идентификатор погоды, берётся из
    data/meta/weather_points.csv. Для kind eu и для UES/IPS_* имена совпадают, кроме DK1 и DK2.
    """
    out = {}
    if kind == "eu":
        for w in weather_ids:
            out[w] = EU_ALIAS.get(w, w)
        return out
    by_name = ru_names()
    for w in weather_ids:
        if w == "UES" or w.startswith("IPS_"):
            out[w] = w
        elif w in RU_BY_WEATHER_ID:
            out[w] = f"R{RU_BY_WEATHER_ID[w]}"
        elif w.startswith("REG_"):
            nm = names.get(w, "")
            if nm in by_name:
                out[w] = f"R{by_name[nm]}"
            else:
                out[w] = None   # нет в иерархии (например, Калининград)
    return out


def point_names():
    names = {}
    with (ROOT / "data/meta/weather_points.csv").open(encoding="utf8") as f:
        for r in csv.DictReader(f):
            names.setdefault(r["node_id"], r["ats_name"] or r["node_name"])
    return names


def read_node_csv(path):
    with gzip.open(path, "rt", newline="") as f:
        rd = csv.reader(f)
        head = next(rd)
        rows = [(r[0], [float(x) if x else float("nan") for x in r[1:]]) for r in rd]
    return head[1:], rows


def build_frame(path, kind, hier_ids):
    import numpy as np  # noqa: PLC0415
    import pandas as pd  # noqa: PLC0415
    cols, rows = read_node_csv(path)
    idx = pd.DatetimeIndex([r[0] for r in rows], name="ds")
    df = pd.DataFrame(np.array([r[1] for r in rows], dtype="float32"), index=idx, columns=cols)
    mp = rename_map(kind, cols, point_names())
    df = df[[c for c in cols if mp.get(c)]].rename(columns=mp)
    df = df[[c for c in df.columns if c in hier_ids]]
    if kind == "ru":
        for leaf, src in RU_FALLBACK.items():
            if leaf in hier_ids and leaf not in df and src in df:
                df[leaf] = df[src]
        for k in hier_ids:
            if k.startswith("RES_") and k not in df and "IPS_" + k[4:] in df:
                df[k] = df["IPS_" + k[4:]]
    full = pd.date_range(df.index.min(), df.index.max(), freq="h", name="ds")
    return df.reindex(full)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hier", required=True, help="hierarchy.json процессной папки")
    ap.add_argument("--kind", choices=["ru", "eu"], required=True)
    ap.add_argument("--weather", default=str(ROOT / "data/raw/weather"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    nodes = json.load(open(a.hier, encoding="utf8"))["nodes"]
    ids = {n["id"] for n in nodes}
    kids = {n["parent"] for n in nodes if n["parent"]}
    leaves = ids - kids
    for fname, target in ((ARCHIVE_FILE, "temp.parquet"), (FORECAST_FILE, "temp_forecast.parquet")):
        p = Path(a.weather) / fname
        if not p.exists():
            print("нет файла", p, "- пропуск", target)
            continue
        df = build_frame(p, a.kind, ids)
        lack = sorted(leaves - set(df.columns))
        print(target, "строк", len(df), "столбцов", df.shape[1], "от", df.index[0], "до", df.index[-1],
              "доля пропусков %.3f" % float(df.isna().to_numpy().mean()),
              "листья без погоды:", lack or "нет")
        if not a.check:
            Path(a.out).mkdir(parents=True, exist_ok=True)
            df.to_parquet(Path(a.out) / target)


if __name__ == "__main__":
    main()
