#!/usr/bin/env python3
"""Download public hourly electricity-consumption data of Russia.

Sources (all open, no login):
  1. br.so-ups.ru   System Operator (SO UPS) public web API "CommonInfo".
       GenConsum      hourly fact and plan of consumption and generation,
                      price zones 1, 2, 8 and 7 IPS (UES of Russia).
       ForecastConsumSubRf  day-ahead consumption plan by subject of the RF (hourly).
       ForecastConsum  day-ahead consumption plan by IPS (hourly).
  2. atsenergo.ru   ATS (market operator): monthly xlsx per subject of the RF with the
       hourly total actual consumption of the subject, MW ("fact_region").

Runs anywhere with the standard library only (python 3.9+); used on the mac and on pcomp. Repeatable: existing files are skipped,
the current period is always refreshed. Pauses between requests keep the load low.

    python3 -u src/download_so_ups.py dict genconsum subrf ats --start 2015
    python3 -u src/download_so_ups.py ats --start 2019 --pause 0.4

The certificate chain of these sites is Russian (not in the macOS store), so the
check is switched off: the data are public, nothing secret is sent.
"""

import argparse
import datetime as dt
import json
import os
import re
import signal
import socket
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

BR = "https://br.so-ups.ru/webapi/api"
ATS = "https://www.atsenergo.ru"
UA = "Mozilla/5.0 (research; hourly load data for forecast reconciliation)"
CTX = ssl._create_unverified_context()

# IPS ids from /api/dict/GetDictionaries
OES = {"CENTER": 530000, "SOUTH": 550000, "VOLGA": 600000, "SIBERIA": 610000,
       "URAL": 630000, "NORTHWEST": 840000}
OES_EAST = 540000
PAUSE = 0.3


# pcomp resolves names over TCP (resolv.conf: options use-vc) and a lookup sometimes hangs for minutes.
# One lookup per host is enough for a download run, so the answers are kept in memory.
_GAI = socket.getaddrinfo
_GAI_CACHE = {}


def _cached_getaddrinfo(host, *args, **kwargs):
    key = (host, args, tuple(sorted(kwargs.items())))
    if key not in _GAI_CACHE:
        _GAI_CACHE[key] = _GAI(host, *args, **kwargs)
    return _GAI_CACHE[key]


socket.getaddrinfo = _cached_getaddrinfo


def _hung(*_):
    raise TimeoutError("request hung (TLS or the server did not answer)")


def http(url, data=None, tries=5, timeout=90):
    """GET or POST with retries; returns bytes. A watchdog alarm stops a hang that the socket
    timeout does not cover."""
    last = None
    signal.signal(signal.SIGALRM, _hung)
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers={
                "User-Agent": UA, "X-Requested-With": "XMLHttpRequest"})
            signal.alarm(timeout + 30)
            try:
                with urllib.request.urlopen(req, context=CTX, timeout=timeout) as r:
                    body = r.read()
            finally:
                signal.alarm(0)
            time.sleep(PAUSE)
            return body
        except Exception as exc:  # noqa: BLE001 - retry on any network error
            last = exc
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"failed {url}: {last}")


def save(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(body)
    tmp.replace(path)


def do_dict(root):
    save(root / "br" / "dictionaries.json", http(f"{BR}/dict/GetDictionaries"))
    print("dict ok", flush=True)


def years(start, end):
    return range(start, end.year + 1)


def do_genconsum(root, start, end):
    """Fact and plan by price zone and IPS. One request per calendar year."""
    for y in years(start, end):
        d0, d1 = f"{y}.01.01", f"{y}.12.31"
        if y == end.year:
            d1 = end.strftime("%Y.%m.%d")
        out = root / "br" / f"genconsum_{y}.json"
        if not out.exists() or y == end.year:
            q = "priceZone[]=1&priceZone[]=2&" + "&".join(
                f"oesTerritory[]={i}" for i in OES.values())
            url = f"{BR}/CommonInfo/GenConsum?{q}&startDate={d0}&endDate={d1}"
            save(out, http(url))
        out2 = root / "br" / f"genconsum_z2_{y}.json"
        if not out2.exists() or y == end.year:
            url = (f"{BR}/CommonInfo/GenConsum_Z2?priceZone[]=8&oesTerritory[]={OES_EAST}"
                   f"&startDate={d0}&endDate={d1}")
            save(out2, http(url))
        print("genconsum", y, flush=True)


def per_day(root, name, start, end, make_url):
    """One request per day; one json per year: {"YYYY-MM-DD": <answer>}."""
    for y in years(max(start, 2017), end):
        out = root / "br" / f"{name}_{y}.json"
        have = json.loads(out.read_text()) if out.exists() else {}
        day = dt.date(y, 1, 1)
        last = min(dt.date(y, 12, 31), end + dt.timedelta(days=1))
        while day <= last:
            key = day.isoformat()
            if key not in have:
                have[key] = json.loads(http(make_url(day), tries=8, timeout=15))
            day += dt.timedelta(days=1)
        save(out, json.dumps(have, ensure_ascii=False).encode("utf8"))
        print(name, y, len(have), flush=True)


def do_ats(root, start, end):
    """Monthly xls (until 2025) or xlsx per subject. The file list is read from the site (POST /js-data)."""
    for y in years(start, end):
        for m in range(1, 13):
            if dt.date(y, m, 1) > end:
                break
            period = f"{m:02d}{y}"
            lst = root / "ats" / "lists" / f"{period}.html"
            recent = dt.date(y, m, 1) >= end.replace(day=1) - dt.timedelta(days=62)
            if not lst.exists() or recent:
                body = http(f"{ATS}/js-data", urllib.parse.urlencode(
                    {"data": "results/market/fact_region", "id": period}).encode())
                save(lst, body)
            html = lst.read_text(encoding="utf8")
            links = re.findall(r'href="([^"]+fact_region\.xlsx?)">([^<]+)<', html)
            n = 0
            for url, _name in links:
                f = root / "ats" / "xlsx" / f"{y}{m:02d}" / url.rsplit("/", 1)[1]
                if f.exists() and f.stat().st_size > 1000 and not recent:
                    continue
                save(f, http(url))
                n += 1
            print("ats", period, "links", len(links), "new", n, flush=True)


def main():
    global PAUSE
    ap = argparse.ArgumentParser()
    ap.add_argument("what", nargs="+", choices=["dict", "genconsum", "subrf", "subrf_z2",
                                                "forecast_ips", "ats"])
    ap.add_argument("--root", default=os.environ.get(
        "SO_UPS_ROOT", str(Path(__file__).resolve().parents[1] / "data/raw/so_ups")))
    ap.add_argument("--start", type=int, default=2015)
    ap.add_argument("--end", default=dt.date.today().isoformat())
    ap.add_argument("--pause", type=float, default=PAUSE)
    a = ap.parse_args()
    PAUSE = a.pause
    root, end = Path(a.root), dt.date.fromisoformat(a.end)
    ids = "&".join(f"oesTerritory[]={i}" for i in OES.values())
    for w in a.what:
        if w == "dict":
            do_dict(root)
        elif w == "genconsum":
            do_genconsum(root, a.start, end)
        elif w == "subrf":
            per_day(root, "forecast_subrf", a.start, end,
                    lambda d: f"{BR}/CommonInfo/ForecastConsumSubRf?startDate={d:%Y.%m.%d}")
        elif w == "subrf_z2":
            per_day(root, "forecast_subrf_z2", a.start, end,
                    lambda d: f"{BR}/CommonInfo/ForecastConsumSubRf_Z2?startDate={d:%Y.%m.%d}")
        elif w == "forecast_ips":
            per_day(root, "forecast_ips", a.start, end,
                    lambda d: f"{BR}/CommonInfo/ForecastConsum?priceZone=&{ids}&startDate={d:%Y.%m.%d}")
        elif w == "ats":
            do_ats(root, a.start, end)
    return 0


if __name__ == "__main__":
    sys.exit(main())
