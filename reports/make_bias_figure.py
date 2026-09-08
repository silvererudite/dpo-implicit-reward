"""
Does either scorer's length preference drift as it trains?

Hypothesis going in: the explicit reward model's length bias GROWS with training, which would
explain its in-distribution gains, its collapse on the length-inverted Chat-Hard subset, and the
shape of its overfitting curve with a single mechanism.

The data refutes that for the explicit model and supports a different drift for DPO. Both panels
share an x-axis; the left shows deviation from the human rate on the same pairs, the right shows
accuracy, so the two can be read against each other without a dual axis.

    python reports/make_bias_figure.py
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "reports", "figures")
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8a8a85"; GRID = "#e4e4e0"
C = {"implicit": "#2a78d6", "explicit": "#eb6834"}
LAB = {"implicit": "DPO implicit reward", "explicit": "Explicit reward model"}


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(color=GRID, linewidth=1, alpha=0.9); ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11, color=INK, loc="left", pad=8, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=9, color=INK2)
    ax.set_xlabel("training epochs", fontsize=9, color=INK2)


def main():
    data = {}
    for k in ("explicit", "implicit"):
        p = os.path.join(ROOT, "results", f"bias_over_training_{k}_s0.json")
        if os.path.exists(p):
            data[k] = json.load(open(p))["checkpoints"]
    if not data:
        raise SystemExit("no bias results found")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6), facecolor=SURFACE)

    for k, rows in data.items():
        ep = [r["epoch"] for r in rows]
        bias = [r["rates"]["length"]["scorer"] - r["rates"]["length"]["human"] for r in rows]
        a1.plot(ep, bias, color=C[k], linewidth=2.4, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.4, label=LAB[k], zorder=4)
        a2.plot(ep, [r["accuracy"] for r in rows], color=C[k], linewidth=2.4, marker="o",
                markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.4, label=LAB[k], zorder=4)

    a1.axhline(0, color=INK, linewidth=1.8, zorder=2)
    a1.annotate("matches the human rate (humans pick longer 55.7% of the time)",
                xy=(0.98, 0.004), xycoords=("axes fraction", "data"), ha="right",
                fontsize=8.5, color=MUTED, va="bottom")
    style(a1, "Length preference, relative to humans", "scorer rate − human rate")
    a1.annotate("prefers LONGER than humans", xy=(0.02, 0.93), xycoords="axes fraction",
                fontsize=8.5, color=MUTED)
    a1.annotate("prefers SHORTER than humans", xy=(0.02, 0.04), xycoords="axes fraction",
                fontsize=8.5, color=MUTED)
    style(a2, "Held-out accuracy on the same pairs", "pairwise accuracy")
    a2.axhline(0.5, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)

    for ax in (a1, a2):
        ax.legend(loc="best", frameon=False, fontsize=9, labelcolor=INK2)

    fig.suptitle("Does either scorer's length preference drift during training?",
                 fontsize=13.5, color=INK, x=0.006, ha="left", fontweight="bold", y=0.995)
    fig.text(0.006, 0.900,
             "Measured on 8 checkpoints per model, 500 held-out pairs each. The explicit model's bias "
             "peaks early and then FADES — the opposite of what we expected.\nDPO's drifts steadily "
             "further from humans, toward preferring shorter answers, and its accuracy falls with it.",
             fontsize=9.5, color=INK2, linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.868))
    p = os.path.join(FIG, "fig12_bias_drift.png")
    fig.savefig(p, dpi=200, facecolor=SURFACE); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
