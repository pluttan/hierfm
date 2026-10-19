#!/usr/bin/env python3
"""Скачивание часовой температуры воздуха по точкам из data/meta/weather_points.csv.

Источники Open-Meteo, ключ не нужен:
  archive  archive-api.open-meteo.com, модель era5_land (реанализ ERA5-Land, 2 м). Факт.
  prev     previous-runs-api.open-meteo.com, поле temperature_2m_previous_day1:
           прогноз, выпущенный за 24 часа до часа события. Честный прогноз задним числом.

Только стандартная библиотека. Запуск на маке:
  nice python3.12 -u src/download_weather.py --out DIR [--plan "archive:2026,2025;prev:2026,2025"]

Файлы: DIR/archive/ГОД/ТОЧКА.csv и DIR/prev_МОДЕЛЬ/ГОД/ТОЧКА.csv, колонки time (UTC) и значение.
Повторный запуск пропускает готовые файлы, поэтому скрипт можно запускать каждый день.

Лимиты бесплатного API (open-meteo.com/en/terms): 600 вызовов в минуту, 5000 в час,
10000 в день, 300000 в месяц. Вызов на год по одной точке считается дробно, примерно как
дни/14 вызовов. Скрипт держит паузу по числу условных вызовов. При ответе о дневном лимите
он пишет статус и выходит с кодом 75. Тогда повторите запуск на следующие сутки (UTC).

Сборка узловых рядов (взвешенное среднее по городам узла):
  python3.12 -u src/download_weather.py --out DIR --aggregate archive
  python3.12 -u src/download_weather.py --out DIR --aggregate prev_ecmwf_ifs025
"""
import argparse
import csv
import datetime as dt
import gzip
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POINTS = ROOT / "data/meta/weather_points.csv"
UA = "Mozilla/5.0 (research; soglasovanie-prognozov)"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
PREV = "https://previous-runs-api.open-meteo.com/v1/forecast"
FIRST_DAY = dt.date(2018, 1, 1)
ARCHIVE_LAG = 7          # дней: ERA5-Land отстаёт примерно на 5 суток
PREV_MODEL = "ecmwf_ifs025"
PREV_START = {"ecmwf_ifs025": dt.date(2024, 3, 1), "gfs_global": dt.date(2021, 5, 1),
              "icon_global": dt.date(2024, 3, 1)}
SEC_PER_UNIT = 0.12      # пауза на один условный вызов: около 500 в минуту
DEFAULT_PLAN = ("archive:2026,2025;prev:2026,2025;archive:2024;prev:2024;"
                "archive:2023,2022,2021,2020,2019,2018")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def load_points():
    seen = {}
    with POINTS.open(encoding="utf8") as f:
        for r in csv.DictReader(f):
            seen.setdefault(r["point_id"], (float(r["lat"]), float(r["lon"])))
    return seen


class DailyLimit(Exception):
    pass


def fetch(url):
    """GET с разбором ответов о лимитах. Возвращает словарь JSON."""
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            try:
                body = json.load(e)
            except Exception:  # noqa: BLE001
                body = {}
            reason = str(body.get("reason", "")) or f"HTTP {e.code}"
            low = reason.lower()
            if "daily" in low:
                raise DailyLimit(reason)
            if "hourly" in low:
                now = time.time()
                wait = 3600 - now % 3600 + 15
                log("hourly limit:", reason, "- sleep", int(wait), "s")
                time.sleep(wait)
            elif "minutely" in low or e.code == 429:
                log("minute limit:", reason, "- sleep 70 s")
                time.sleep(70)
            elif e.code >= 500:
                log("server error", e.code, "- sleep", 10 * (attempt + 1))
                time.sleep(10 * (attempt + 1))
            else:
                raise RuntimeError(reason)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            log("network error", e, "- sleep", 10 * (attempt + 1))
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("too many retries: " + url)


def stage_range(kind, year, model):
    today = dt.datetime.now(dt.timezone.utc).date()
    a = dt.date(year, 1, 1)
    b = dt.date(year, 12, 31)
    if kind == "archive":
        a = max(a, FIRST_DAY)
        b = min(b, today - dt.timedelta(days=ARCHIVE_LAG))
    else:
        a = max(a, PREV_START[model])
        b = min(b, today - dt.timedelta(days=1))
    return a, b


def download_stage(kind, year, points, out, model, state):
    a, b = stage_range(kind, year, model)
    if a > b:
        log("stage", kind, year, "empty range")
        return
    sub = "archive" if kind == "archive" else f"prev_{model}"
    field = "temperature_2m" if kind == "archive" else "temperature_2m_previous_day1"
    units = max(1.0, ((b - a).days + 1) / 14.0)
    done = 0
    for pid, (lat, lon) in points.items():
        f = out / sub / str(year) / f"{pid}.csv"
        if f.exists() and f.stat().st_size > 100:
            continue
        q = {"latitude": lat, "longitude": lon, "start_date": a.isoformat(), "end_date": b.isoformat(),
             "hourly": field, "timezone": "UTC", "temperature_unit": "celsius"}
        if kind == "archive":
            q["models"] = "era5_land"
            url = ARCHIVE + "?" + urllib.parse.urlencode(q)
        else:
            q["models"] = model
            url = PREV + "?" + urllib.parse.urlencode(q)
        d = fetch(url)
        h = d["hourly"]
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        with tmp.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["time", field])
            for t, v in zip(h["time"], h[field]):
                w.writerow([t, "" if v is None else v])
        tmp.replace(f)
        state["calls"] += 1
        state["units"] += units
        done += 1
        if done % 25 == 0:
            log(f"{kind} {year}: {done} files, total calls {state['calls']}, units~{state['units']:.0f}")
        time.sleep(units * SEC_PER_UNIT)
    log(f"stage {kind} {year} finished, new files {done}")


def run(args):
    out = Path(args.out)
    points = load_points()
    if args.limit_points:
        points = dict(list(points.items())[:args.limit_points])
    log("points", len(points), "out", out)
    state = {"calls": 0, "units": 0.0}
    plan = []
    for part in args.plan.split(";"):
        kind, years = part.split(":")
        plan += [(kind, int(y)) for y in years.split(",")]
    try:
        for kind, year in plan:
            download_stage(kind, year, points, out, args.model, state)
    except DailyLimit as e:
        log("DAILY LIMIT:", e, "- stop; run again after 00:00 UTC. calls", state["calls"],
            "units~", int(state["units"]))
        return 75
    log("plan finished, calls", state["calls"], "units~", int(state["units"]))
    return 0


def aggregate(args):
    """Узловые ряды: взвешенное среднее по городам узла. Часы UTC, сетка по первому и последнему часу."""
    out = Path(args.out)
    src = out / args.aggregate
    rows = list(csv.DictReader(POINTS.open(encoding="utf8")))
    series = {}
    for pid in {r["point_id"] for r in rows}:
        data = {}
        for f in sorted(src.glob(f"*/{pid}.csv")):
            with f.open() as fh:
                next(fh)
                for line in fh:
                    t, v = line.rstrip("\n").split(",")
                    data[t] = float(v) if v else float("nan")
        series[pid] = data
    stamps = sorted({t for d in series.values() for t in d})
    idx = {t: i for i, t in enumerate(stamps)}
    arrs = {}
    for pid, d in series.items():
        a = array("f", [float("nan")]) * len(stamps)
        for t, v in d.items():
            a[idx[t]] = v
        arrs[pid] = a
    nodes = {}
    for r in rows:
        nodes.setdefault(r["node_id"], []).append((r["point_id"], float(r["weight"])))
    names = sorted(nodes)
    dest = out / f"node_temperature_{args.aggregate}.csv.gz"
    with gzip.open(dest, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["time"] + names)
        for i, t in enumerate(stamps):
            line = [t]
            for n in names:
                s = ws = 0.0
                for pid, wt in nodes[n]:
                    v = arrs[pid][i]
                    if v == v:
                        s += wt * v
                        ws += wt
                line.append(f"{s / ws:.2f}" if ws > 0 else "")
            w.writerow(line)
    log("written", dest, "rows", len(stamps), "nodes", len(names))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data/raw/weather"))
    ap.add_argument("--plan", default=DEFAULT_PLAN)
    ap.add_argument("--model", default=PREV_MODEL, help="модель для prev: ecmwf_ifs025, gfs_global, icon_global")
    ap.add_argument("--limit-points", type=int, default=0, help="только первые N точек (для проверки)")
    ap.add_argument("--aggregate", help="archive или prev_МОДЕЛЬ: собрать узловые ряды вместо скачивания")
    args = ap.parse_args()
    sys.exit(aggregate(args) if args.aggregate else run(args))


if __name__ == "__main__":
    main()
