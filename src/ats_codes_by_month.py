#!/usr/bin/env python3
"""Which ATS subject codes are listed in which months (from the saved list pages). Stdlib only.

    python3 -I src/ats_codes_by_month.py ROOT
"""
import re
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "ats" / "lists"
seen = {}
names = {}
for f in sorted(root.glob("*.html"), key=lambda p: (p.stem[2:], p.stem[:2])):
    mm, yy = f.stem[:2], f.stem[2:]
    per = f"{yy}-{mm}"
    for url, nm in re.findall(r'href="([^"]+fact_region\.xlsx?)">([^<]+)<', f.read_text(encoding="utf8")):
        code = url.rsplit("/", 1)[1].split("_")[1]
        seen.setdefault(code, []).append(per)
        names[code] = nm
for code, per in sorted(seen.items(), key=lambda kv: kv[1][0]):
    print(f"{code:>6} {names[code]:<45} {per[0]} .. {per[-1]}  months {len(per)}")
print("months with lists:", len(list(root.glob('*.html'))))
