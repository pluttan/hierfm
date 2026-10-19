"""hierfm command line.

    python -m hierfm synth    --config configs/smoke.yaml   build the synthetic dataset
    python -m hierfm forecast --config CFG [--models a,b]    stage 1, under torchrun one rank per GPU
    python -m hierfm evaluate --config CFG [--jobs N]        stage 2, CPU pool
    python -m hierfm report   --config CFG                   stage 3, tables
"""

import argparse

from . import config


def main():
    ap = argparse.ArgumentParser(prog="hierfm")
    ap.add_argument("stage", choices=["synth", "forecast", "evaluate", "report"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", default="", help="comma separated subset of the config models")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=8, help="origins per model call")
    a = ap.parse_args()
    cfg = config.load(a.config)
    only = [m for m in a.models.split(",") if m] or None

    if a.stage == "synth":
        from . import synthetic
        synthetic.make(cfg.processed)
    elif a.stage == "forecast":
        from . import forecast
        forecast.run(cfg, only, a.chunk)
    elif a.stage == "evaluate":
        from . import evaluate
        evaluate.run(cfg, a.jobs or None, only)
    else:
        from . import report
        report.run(cfg)


if __name__ == "__main__":
    main()
