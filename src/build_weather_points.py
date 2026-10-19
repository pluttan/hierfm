#!/usr/bin/env python3
"""Строит таблицу точек погоды data/meta/weather_points.csv.

Для каждого города из weather_nodes.py запрашивает геокодер Open-Meteo
(geocoding-api.open-meteo.com, данные GeoNames): координаты и население.
Вес города в узле равен населению города, поделённому на сумму по узлу.
Для узла с детьми берутся города всех потомков. Вес пропорционален населению города.
Тогда температура ОЭС равна среднему по городам, взвешенному по населению.

Только стандартная библиотека. Лёгкий скрипт (около 280 коротких запросов).
Результат геокодера кэшируется в data/meta/geocode_cache.json, повторный запуск запросов не делает.

    python3 src/build_weather_points.py
"""
import csv
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import weather_nodes as wn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data/meta"
CACHE = META / "geocode_cache.json"
API = "https://geocoding-api.open-meteo.com/v1/search"
DEFAULT_POP = 50000  # если GeoNames не знает население


def geocode(name, lat, lon, cache):
    key = f"{name}|{lat}|{lon}"
    if key in cache:
        return cache[key]
    q = urllib.parse.urlencode({"name": name, "count": 20, "language": "en", "format": "json"})
    best = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(f"{API}?{q}", timeout=30) as r:
                res = json.load(r).get("results", [])
            break
        except Exception as e:  # noqa: BLE001
            print("retry", name, e, flush=True)
            time.sleep(5 * (attempt + 1))
            res = []
    for x in res:
        d = math.hypot(x["latitude"] - lat, (x["longitude"] - lon) * math.cos(math.radians(lat)))
        if d > 0.8:
            continue
        score = (x.get("population") or 0, -d)
        if best is None or score > best[0]:
            best = (score, x)
    if best is None:
        out = {"found": False, "lat": lat, "lon": lon, "pop": 0, "geoname": None, "admin1": ""}
    else:
        x = best[1]
        out = {"found": True, "lat": round(x["latitude"], 3), "lon": round(x["longitude"], 3),
               "pop": x.get("population") or 0, "geoname": x.get("id"), "admin1": x.get("admin1", "")}
    cache[key] = out
    time.sleep(0.4)
    return out


def main():
    META.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    nodes = wn.all_nodes()
    by_id = {n["id"]: n for n in nodes}
    children = {}
    for n in nodes:
        children.setdefault(n["parent"], []).append(n["id"])

    def leaves(nid):
        if not children.get(nid):
            return [nid]
        return [leaf for c in children[nid] for leaf in leaves(c)]

    info = {}
    for n in nodes:
        for name, lat, lon in n["cities"]:
            g = geocode(name, lat, lon, cache)
            info[(n["id"], name)] = g
    CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1))

    rows, missing = [], []
    for n in nodes:
        pts = []
        for leaf in leaves(n["id"]):
            for name, lat, lon in by_id[leaf]["cities"]:
                g = info[(leaf, name)]
                pop = g["pop"] or DEFAULT_POP
                if not g["found"] or not g["pop"]:
                    missing.append((leaf, name, g["found"]))
                pts.append((leaf, name, g["lat"], g["lon"], pop, g))
        total = sum(p[4] for p in pts)
        for leaf, name, la, lo, pop, g in pts:
            pid = f"{la:.2f}_{lo:.2f}"
            rows.append(dict(node_id=n["id"], level=n["level"], parent=n["parent"], hier=n["hier"],
                             node_name=n["name"], leaf_id=leaf, city=name, lat=f"{la:.3f}", lon=f"{lo:.3f}",
                             weight=f"{pop / total:.6f}", population=g["pop"], point_id=pid,
                             has_load=n["has_load"], ats_name=n["ats"] or ""))
    cols = list(rows[0])
    out = META / "weather_points.csv"
    with out.open("w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)
    uniq = sorted({r["point_id"] for r in rows})
    print("rows", len(rows), "nodes", len(nodes), "unique points", len(uniq))
    print("without GeoNames match or population (default weight used):", missing)


if __name__ == "__main__":
    main()
