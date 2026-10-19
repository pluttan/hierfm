#!/usr/bin/env python3
"""Скачивание европейской иерархии нагрузки без ключей API.

Источники (все открытые, без токена):
  smard       SMARD (Bundesnetzagentur): нагрузка Германии и 4 зон ОПС, час.
  energinet   Energinet: валовое потребление DK1 и DK2, час.
  odre        RTE eco2mix (ODRE): потребление 12 регионов и всей Франции, 30 мин / 15 мин.
  echarts     energy-charts.info: нагрузка (ряд "Load") по странам, из данных ENTSO-E.
  openmeteo   Open-Meteo: температура 2 м (реанализ) и архив прогнозов по точкам.

Только стандартная библиотека. Запуск на маке:
  nice python3.12 -u download_europe.py --out DIR [--only smard,odre] [--start 2020-01-01] [--end 2026-10-01]
Повторный запуск пропускает готовые файлы. Ключ --force скачивает заново.
Скрипт пишет manifest.json со списком файлов, размерами и sha256.
"""
import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

UA = "Mozilla/5.0 (research; soglasovanie-prognozov)"

SMARD_REGIONS = ["DE", "50Hertz", "Amprion", "TenneT", "TransnetBW"]
SMARD_FILTER = 410  # Stromverbrauch: Gesamt (Netzlast)

ODRE_BASE = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets"

ECHARTS_COUNTRIES = [
    "at", "be", "bg", "ch", "cz", "de", "dk", "ee", "es", "fi", "fr", "gr", "hr",
    "hu", "ie", "it", "lt", "lu", "lv", "me", "mk", "nl", "no", "pl", "pt", "ro",
    "rs", "se", "si", "sk",
]

# Приближённые координаты. Для регионов Франции это префектуры, для зон ОПС и стран
# это опорные города. Это грубая замена весов по населению.
POINTS = {
    "FR_ARA": (45.76, 4.84), "FR_BFC": (47.32, 5.04), "FR_BRE": (48.11, -1.68),
    "FR_CVL": (47.90, 1.91), "FR_GES": (48.58, 7.75), "FR_HDF": (50.63, 3.06),
    "FR_NOR": (49.44, 1.10), "FR_NAQ": (44.84, -0.58), "FR_OCC": (43.60, 1.44),
    "FR_PDL": (47.22, -1.55), "FR_PAC": (43.30, 5.37), "FR_IDF": (48.86, 2.35),
    "DE_50Hertz": (52.52, 13.40), "DE_Amprion": (50.94, 6.96),
    "DE_TenneT": (51.31, 9.48), "DE_TransnetBW": (48.78, 9.18),
    "DK1": (56.16, 10.20), "DK2": (55.68, 12.57),
    "AT": (48.21, 16.37), "BE": (50.85, 4.35), "BG": (42.70, 23.32),
    "CH": (46.95, 7.45), "CZ": (50.08, 14.44), "EE": (59.44, 24.75),
    "ES": (40.42, -3.70), "FI": (60.17, 24.94), "GR": (37.98, 23.73),
    "HR": (45.81, 15.98), "HU": (47.50, 19.04), "IE": (53.35, -6.26),
    "IT": (41.90, 12.50), "LT": (54.69, 25.28), "LV": (56.95, 24.11),
    "LU": (49.61, 6.13), "NL": (52.37, 4.90), "NO": (59.91, 10.75),
    "PL": (52.23, 21.01), "PT": (38.72, -9.14), "RO": (44.43, 26.10),
    "RS": (44.79, 20.45), "SE": (59.33, 18.07), "SI": (46.06, 14.51),
    "SK": (48.15, 17.11),
}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def get(url, retries=6, pause=2.0, timeout=120):
    """GET с повтором. На ответ 429 ждёт дольше."""
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
            if body.startswith(b"Too Many Requests"):
                raise urllib.error.HTTPError(url, 429, "text 429", None, None)
            return body
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (400, 404):
                raise
            wait = pause * (2 ** i) + (10 if e.code == 429 else 0)
        except Exception as e:  # сеть, таймаут
            last = e
            wait = pause * (2 ** i)
        log("повтор", i + 1, url[:110], repr(last)[:80], "жду", wait)
        time.sleep(wait)
    raise RuntimeError(f"не удалось скачать {url}: {last}")


def write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(tmp, mode) as f:
        f.write(data)
    os.replace(tmp, path)


def ms(date_str):
    d = dt.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp() * 1000)


# ---------------------------------------------------------------- SMARD
def dl_smard(out, start, end, force):
    d = os.path.join(out, "smard")
    t0, t1 = ms(start), ms(end)
    for reg in SMARD_REGIONS:
        path = os.path.join(d, f"smard_load_{reg}.csv")
        if os.path.exists(path) and not force:
            log("smard пропуск", reg)
            continue
        idx = json.loads(get(f"https://www.smard.de/app/chart_data/{SMARD_FILTER}/{reg}/index_hour.json"))
        stamps = sorted(idx["timestamps"])
        # недельный блок начинается раньше нужной даты: берём блок, который её содержит
        first = max([s for s in stamps if s <= t0] or [stamps[0]])
        rows = {}
        for s in [x for x in stamps if x >= first and x < t1]:
            url = f"https://www.smard.de/app/chart_data/{SMARD_FILTER}/{reg}/{SMARD_FILTER}_{reg}_hour_{s}.json"
            js = json.loads(get(url))
            for t, v in js["series"]:
                if t0 <= t < t1 and v is not None:
                    rows[t] = v
            time.sleep(0.15)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["timestamp_utc", "load_mw"])
        for t in sorted(rows):
            w.writerow([dt.datetime.fromtimestamp(t / 1000, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), rows[t]])
        write(path, buf.getvalue())
        log("smard", reg, len(rows), "строк")


# ------------------------------------------------------------ Energinet
def dl_energinet(out, start, end, force):
    path = os.path.join(out, "energinet", "energinet_consumption_settlement.csv")
    if os.path.exists(path) and not force:
        log("energinet пропуск")
        return
    q = urllib.parse.urlencode({
        "start": start + "T00:00", "end": end + "T00:00", "limit": 0,
        "sort": "HourUTC asc",
        "columns": "HourUTC,PriceArea,GrossConsumptionMWh,GridLossTransmissionMWh,"
                   "GridLossInterconnectorsMWh,GridLossDistributionMWh,PowerToHeatMWh",
    })
    js = json.loads(get("https://api.energidataservice.dk/dataset/ProductionConsumptionSettlement?" + q))
    recs = js["records"]
    cols = list(recs[0].keys())
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for r in recs:
        w.writerow([r[c] for c in cols])
    write(path, buf.getvalue())
    log("energinet", len(recs), "строк")


# ----------------------------------------------------------------- ODRE
def odre_export(dataset, where, select):
    q = urllib.parse.urlencode({"select": select, "where": where, "timezone": "UTC",
                                "delimiter": ",", "order_by": "date_heure"})
    return get(f"{ODRE_BASE}/{dataset}/exports/csv?{q}", timeout=600)


def dl_odre(out, start, end, force):
    d = os.path.join(out, "odre")
    where = f'date_heure>="{start}T00:00:00Z" AND date_heure<"{end}T00:00:00Z"'
    jobs = [
        ("eco2mix-regional-cons-def", "regional_def", "date_heure,code_insee_region,libelle_region,nature,consommation"),
        ("eco2mix-regional-tr", "regional_tr", "date_heure,code_insee_region,libelle_region,nature,consommation"),
        ("eco2mix-national-cons-def", "national_def", "date_heure,nature,consommation"),
        ("eco2mix-national-tr", "national_tr", "date_heure,nature,consommation"),
    ]
    for ds, name, sel in jobs:
        path = os.path.join(d, f"odre_{name}.csv")
        if os.path.exists(path) and not force:
            log("odre пропуск", name)
            continue
        try:
            data = odre_export(ds, where, sel)
        except Exception as e:
            log("odre ОШИБКА", ds, e)
            continue
        write(path, data)
        log("odre", name, len(data), "байт")
        time.sleep(1)


# ------------------------------------------------------------ energy-charts
def dl_echarts(out, start, end, force):
    d = os.path.join(out, "energy_charts")
    y0, y1 = int(start[:4]), int(end[:4])
    fails = []
    for c in ECHARTS_COUNTRIES:
        path = os.path.join(d, f"load_{c}.csv")
        if os.path.exists(path) and not force:
            log("echarts пропуск", c)
            continue
        rows = {}
        ok = True
        for y in range(y0, y1 + 1):
            a = max(start, f"{y}-01-01")
            b = min(end, f"{y}-12-31")
            if a > b:
                continue
            url = f"https://api.energy-charts.info/public_power?country={c}&start={a}&end={b}"
            try:
                js = json.loads(get(url, pause=4.0))
            except Exception as e:
                log("echarts ОШИБКА", c, y, str(e)[:100])
                fails.append((c, y, str(e)[:100]))
                ok = False
                break
            ser = {p["name"]: p["data"] for p in js.get("production_types", [])}
            ld = ser.get("Load")
            if ld is None:
                fails.append((c, y, "нет ряда Load"))
                continue
            for t, v in zip(js["unix_seconds"], ld):
                if v is not None:
                    rows[t] = v
            time.sleep(3.0)
        if not rows or not ok:
            continue
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["timestamp_utc", "load_mw"])
        for t in sorted(rows):
            w.writerow([dt.datetime.fromtimestamp(t, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), rows[t]])
        write(path, buf.getvalue())
        log("echarts", c, len(rows), "строк")
    if fails:
        write(os.path.join(d, "failures.json"), json.dumps(fails, ensure_ascii=False, indent=1))


# ------------------------------------------------------------- Open-Meteo
def dl_openmeteo(out, start, end, force):
    d = os.path.join(out, "openmeteo")
    last = (dt.date.fromisoformat(end) - dt.timedelta(days=1)).isoformat()
    for kind, base in (("archive", "https://archive-api.open-meteo.com/v1/archive"),
                       ("histfc", "https://historical-forecast-api.open-meteo.com/v1/forecast")):
        s = start if kind == "archive" else max(start, "2022-01-01")
        for name, (lat, lon) in POINTS.items():
            path = os.path.join(d, kind, f"t2m_{name}.csv")
            if os.path.exists(path) and not force:
                continue
            q = urllib.parse.urlencode({"latitude": lat, "longitude": lon, "start_date": s,
                                        "end_date": last, "hourly": "temperature_2m",
                                        "timezone": "UTC"})
            try:
                js = json.loads(get(f"{base}?{q}", pause=5.0))
            except Exception as e:
                log("openmeteo ОШИБКА", kind, name, str(e)[:100])
                continue
            h = js["hourly"]
            buf = io.StringIO()
            w = csv.writer(buf)
            w.writerow(["timestamp_utc", "temperature_2m_c"])
            for t, v in zip(h["time"], h["temperature_2m"]):
                w.writerow([t + ":00Z", v])
            write(path, buf.getvalue())
            time.sleep(1.5)
        log("openmeteo готово", kind)


def manifest(out):
    items = []
    for root, _, files in os.walk(out):
        for f in sorted(files):
            if f == "manifest.json" or f.endswith(".part"):
                continue
            p = os.path.join(root, f)
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            items.append({"path": os.path.relpath(p, out), "bytes": os.path.getsize(p), "sha256": h.hexdigest()})
    write(os.path.join(out, "manifest.json"),
          json.dumps({"created": dt.datetime.now(dt.timezone.utc).isoformat(), "files": items}, indent=1))
    log("manifest:", len(items), "файлов")


SOURCES = {"smard": dl_smard, "energinet": dl_energinet, "odre": dl_odre,
           "echarts": dl_echarts, "openmeteo": dl_openmeteo}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default=",".join(SOURCES))
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-10-01", help="не включая")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    for s in a.only.split(","):
        log("== источник", s)
        SOURCES[s](a.out, a.start, a.end, a.force)
    manifest(a.out)


if __name__ == "__main__":
    sys.exit(main())
