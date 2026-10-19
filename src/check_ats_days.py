#!/usr/bin/env python3
"""Count, per month and weekday, the days present in ATS fact_region files (stdlib reader).

    python3 -u src/check_ats_days.py ROOT [YYYYMM ...]
"""
import collections
import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ats_reader as ar  # noqa: E402

root = Path(sys.argv[1])
months = sys.argv[2:] or sorted(p.name for p in (root / "ats" / "xlsx").iterdir())
for m in months:
    files = sorted((root / "ats" / "xlsx" / m).glob("*_46_fact_region.*")) or \
        sorted((root / "ats" / "xlsx" / m).glob("*_01_fact_region.*"))
    if not files:
        print(m, "no file")
        continue
    reg, d = ar.parse(files[0])
    days = sorted({dt.datetime.strptime(x[0], "%d.%m.%Y").date() for x in d})
    wd = collections.Counter(x.weekday() for x in days)
    print(m, files[0].name.split("_")[1], "days", len(days), "weekday counts Mon..Sun",
          [wd.get(k, 0) for k in range(7)], "hours/day", sorted(collections.Counter(x[0] for x in d).values())[:1],
          "first", days[0] if days else None, flush=True)
