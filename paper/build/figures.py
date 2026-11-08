"""Figures of the paper from the CSVs of numbers.py.
usage: figures.py DATA_DIR OUT_DIR [en|ru]
Print style for IEEE proceedings: white page, Times, one column wide, markers + direct labels.
"""
import csv
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

COLUMN_IN = 3.35
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DEPS = [("schaake_hist", "#2a78d6", "o"), ("gauss_copula", "#1baf7a", "^"), ("schaake_temp", "#eda100", "D"),
        ("schaake_err", "#e87ba4", "v"), ("independent", "#eb6834", "s")]
WINDOWS = [7, 14, 28, 91, 365]
TEXT = {
    "en": {"dep": {"schaake_hist": "Schaake, load history", "gauss_copula": "Gaussian copula",
                   "schaake_temp": "Schaake, temperature analogues", "schaake_err": "Schaake, past errors",
                   "independent": "independent draws"},
           "suite": {"ru": "UES of Russia", "europe": "Europe"},
           "skill": "CRPS skill over base (%)", "window": "Covariance window (days)",
           "lambda": "Shrinkage weight λ", "standard": "hourly rows", "neff": "days ($n_{eff}$)",
           "mint": "MinT-shrink", "mint_neff": "MinT-shrink, $n_{eff}$", "wls": "WLS (variances)"},
    "ru": {"dep": {"schaake_hist": "Шааке, история нагрузки", "gauss_copula": "гауссова копула",
                   "schaake_temp": "Шааке, аналоги по температуре", "schaake_err": "Шааке, прошлые ошибки",
                   "independent": "независимые выборки"},
           "suite": {"ru": "ЕЭС России", "europe": "Европа"},
           "skill": "Выигрыш CRPS над базой, %", "window": "Окно ковариации, сутки",
           "lambda": "Вес сжатия λ", "standard": "по часам", "neff": "по суткам ($n_{eff}$)",
           "mint": "MinT-shrink", "mint_neff": "MinT-shrink, $n_{eff}$", "wls": "WLS (дисперсии)"},
}
T = TEXT["en"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "custom", "mathtext.rm": "Times New Roman", "mathtext.it": "Times New Roman:italic",
    "font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 6.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 7.5,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.1, "lines.markersize": 3.2, "savefig.dpi": 300,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def num(v):
    return float(v) if v not in ("", None) else float("nan")


def fmt(v, _=None):
    s = f"{v:g}".replace("-", "\u2212")                # a true minus sign, not a hyphen
    return s.replace(".", ",") if T is TEXT["ru"] else s


def window_axis(ax):
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(WINDOWS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt))
    ax.set_xlim(6, 420)


def fig_dependence(data, out):
    """Skill of MinT-shrink with the n_eff weight under the five ways of drawing joint paths."""
    fig, axes = plt.subplots(2, 2, figsize=(COLUMN_IN, 2.9), sharex=True, sharey="row")
    for col, suite in enumerate(("ru", "europe")):
        rows = read(data / f"dep_skill_{suite}.csv")
        for r_i, h in enumerate((24, 168)):
            ax = axes[r_i, col]
            for dep, color, marker in DEPS:
                pts = sorted((int(num(r["window"])), num(r["mint_shrink_neff"])) for r in rows
                             if r["dependence"] == dep and int(num(r["h"])) == h)
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, marker=marker,
                        label=T["dep"][dep])
            ax.axhline(0, color=INK, linewidth=0.6)
            window_axis(ax)
            ax.set_title(f"{T['suite'][suite]}, h = {h}", pad=3)
            if r_i == 1:
                ax.set_xlabel(T["window"])
    fig.supylabel(T["skill"], fontsize=7.5, x=0.01)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.005),
               handlelength=1.6, columnspacing=1.0)
    fig.tight_layout(rect=(0.02, 0.16, 1, 1), h_pad=0.6, w_pad=0.8)
    fig.savefig(out / "fig1-dependence.png")
    plt.close(fig)


def fig_weight(data, out):
    """(a) shrinkage weight with rows and with days as samples; (b) what it does to the 168-h skill."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(COLUMN_IN, 1.9))
    for suite, ls in (("ru", "-"), ("europe", "--")):
        rows = read(data / f"lambda_{suite}.csv")
        w = [int(num(r["window"])) for r in rows]
        a.plot(w, [num(r["mint_shrink"]) for r in rows], color="#eb6834", marker="s", linestyle=ls)
        a.plot(w, [num(r["mint_shrink_neff"]) for r in rows], color="#2a78d6", marker="o", linestyle=ls)
    window_axis(a)
    a.set_ylim(0, 1.05)
    a.set_ylabel(T["lambda"])
    a.set_xlabel(T["window"])
    a.text(8, 0.14, T["standard"], color="#eb6834", fontsize=6.5)
    a.text(16, 0.97, T["neff"], color="#2a78d6", fontsize=6.5)
    a.set_title("(a)", pad=3)

    rows = read(data / "dep_skill_ru.csv")
    sel = [r for r in rows if r["dependence"] == "schaake_hist" and int(num(r["h"])) == 168]
    sel.sort(key=lambda r: num(r["window"]))
    w = [int(num(r["window"])) for r in sel]
    for key, color, marker, label in (("mint_shrink", "#eb6834", "s", T["mint"]),
                                      ("wls_var", "#52514e", "^", T["wls"]),
                                      ("mint_shrink_neff", "#2a78d6", "o", T["mint_neff"])):
        b.plot(w, [num(r[key]) for r in sel], color=color, marker=marker, label=label)
    b.axhline(0, color=INK, linewidth=0.6)
    window_axis(b)
    b.set_xlabel(T["window"])
    b.set_ylabel(T["skill"])
    b.set_title("(b)", pad=3)
    h, lab = b.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01),
               handlelength=1.4, columnspacing=0.9)
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.6)
    fig.savefig(out / "fig2-weight.png")
    plt.close(fig)


def main(data, out, lang="en"):
    global T
    T = TEXT[lang]
    data, out = pathlib.Path(data), pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fig_dependence(data, out)
    fig_weight(data, out)
    print(f"figures: {out}")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    main(*sys.argv[1:])
