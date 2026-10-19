"""Raw downloads -> data/processed/<name>/ (see data.py for the layout).

The builders depend on the exact raw formats found by the initial research (docs/research);
they are filled in once the downloads are settled. Until then `make data` stops with the list
of what is missing instead of producing half a dataset.
"""

import sys
from pathlib import Path

from .config import ROOT

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
BUILDERS = {}


def builder(name):
    def wrap(fn):
        BUILDERS[name] = fn
        return fn
    return wrap


def main(argv):
    names = list(BUILDERS) if not argv or argv[0] == "all" else argv
    missing = [n for n in names if n not in BUILDERS]
    if missing or not BUILDERS:
        sys.exit(f"prepare: no builder yet for {missing or ['ru', 'europe']}; "
                 f"raw data in {RAW} must be settled first (docs/research)")
    for n in names:
        BUILDERS[n](RAW, PROCESSED / n)


if __name__ == "__main__":
    main(sys.argv[1:])
