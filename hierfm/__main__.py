"""hierfm command line.

    python -m hierfm synth    --config CFG                   build a synthetic dataset
    python -m hierfm forecast --config CFG [--models a,b]    stage 1, under torchrun one rank per GPU
    python -m hierfm evaluate --config CFG [--jobs N]        stage 2, CPU pool
    python -m hierfm report   --config CFG                   stage 3, tables
    python -m hierfm suite    SUITE.yaml                     write the configs of a suite, print their paths
    python -m hierfm collect  SUITE.yaml                     merge the scores of a suite
"""

import argparse
import os
import time
from pathlib import Path

from . import config


def ensure_synthetic(cfg):
    """A synthetic run builds its dataset on first use; other ranks wait for rank 0."""
    if not cfg.synthetic or (Path(cfg.processed) / "hierarchy.json").exists():
        return
    if int(os.environ.get("RANK", 0)) == 0:
        from . import synthetic
        synthetic.make(cfg.processed, **cfg.synthetic)
    while not (Path(cfg.processed) / "hierarchy.json").exists():
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser(prog="hierfm")
    ap.add_argument("stage", choices=["synth", "forecast", "evaluate", "report", "suite", "collect"])
    ap.add_argument("suite_file", nargs="?")
    ap.add_argument("--config")
    ap.add_argument("--models", default="", help="comma separated subset of the config models")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=8, help="origins per model call")
    ap.add_argument("--force", action="store_true", help="evaluate: re-score files that already exist")
    a = ap.parse_args()

    if a.stage in ("suite", "collect"):
        from . import suite
        if a.stage == "suite":
            for p in suite.expand(a.suite_file):
                print(p)
        else:
            suite.collect(a.suite_file)
        return

    cfg = config.load(a.config)
    only = [m for m in a.models.split(",") if m] or None
    if a.stage == "synth":
        from . import synthetic
        synthetic.make(cfg.processed, **cfg.synthetic)
        return
    ensure_synthetic(cfg)
    if a.stage == "forecast":
        from . import forecast
        forecast.run(cfg, only, a.chunk)
    elif a.stage == "evaluate":
        from . import evaluate
        evaluate.run(cfg, a.jobs or None, only, a.force)
    else:
        from . import report
        report.run(cfg)


if __name__ == "__main__":
    main()
