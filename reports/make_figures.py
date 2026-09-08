"""
Result figures for the instructor report / paper.

Three figures, each answering one question a reviewer will ask:
  fig1_headline      -- who wins, where, and is the difference bigger than the noise?
  fig2_length_shift  -- how much of each score was just "prefer the longer answer"?
  fig3_paired_delta  -- which explicit-vs-implicit gaps survive seed variance?

Palette: dataviz categorical slots 1-3 (blue/orange/aqua), validated all-pairs in light
mode (worst CVD dE 9.2, normal-vision 24.0). The aqua slot sits under 3:1 contrast on the
light surface, so the relief rule applies -- every mark carries a visible value label and
the numbers are also published as a table in the report.

    python reports/make_figures.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "reports", "figures")
os.makedirs(FIG, exist_ok=True)

SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8a8a85"; GRID = "#e4e4e0"
SERIES = {"implicit": "#2a78d6", "explicit": "#eb6834", "base_logprob": "#1baf7a"}
LABEL = {"implicit": "DPO implicit reward", "explicit": "Explicit BT reward model",
         "base_logprob": "Base log-prob (no reward training)"}
ORDER = ["implicit", "explicit", "base_logprob"]
SETS = ["UltraFeedback (ID)", "RewardBench:Chat (OOD)", "RewardBench:Chat-Hard (OOD)",
        "RewardBench:Safety (OOD)", "RewardBench:Reasoning (OOD)", "HH-harmless (OOD)"]
SHORT = {"UltraFeedback (ID)": "UltraFeedback\n(in-distribution)",
         "RewardBench:Chat (OOD)": "RewardBench\nChat",
         "RewardBench:Chat-Hard (OOD)": "RewardBench\nChat-Hard",
         "RewardBench:Safety (OOD)": "RewardBench\nSafety",
         "RewardBench:Reasoning (OOD)": "RewardBench\nReasoning",
         "HH-harmless (OOD)": "HH-RLHF\nharmless"}
T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=1, alpha=0.9)
    ax.set_axisbelow(True)


def load():
    agg = json.load(open(os.path.join(ROOT, "results", "aggregate_beta0.1_8k.json")))
    seeds = {t: json.load(open(os.path.join(ROOT, "results", f"eval_{t}.json")))
             for t in agg["tags"]}
    return agg["results"], seeds, agg["tags"]


def ci(vals):
    n = len(vals)
    return T95.get(n, 2.0) * (np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0


# ---------------------------------------------------------------- fig 1: headline
def fig1(agg, seeds, tags):
    fig, ax = plt.subplots(figsize=(10, 7.2), facecolor=SURFACE)
    style(ax)
    band, off = 1.0, 0.26
    yt, yl = [], []
    for i, s in enumerate(SETS):
        base_y = -i * band
        yt.append(base_y); yl.append(SHORT[s])
        for j, sc in enumerate(ORDER):
            d = agg[s][sc]["len_controlled_acc"]
            if not d or d.get("mean") is None:
                continue
            y = base_y + (1 - j) * off
            m, e = d["mean"], (d.get("ci95") or 0.0)
            per = [seeds[t][s][sc]["len_controlled_acc"] for t in tags
                   if seeds[t][s][sc].get("len_controlled_acc") is not None]
            # individual seeds behind the summary: n=3 is visible, not implied
            ax.scatter(per, [y] * len(per), s=26, facecolors="none",
                       edgecolors=SERIES[sc], linewidths=1.2, alpha=0.75, zorder=3)
            ax.plot([m - e, m + e], [y, y], color=SERIES[sc], linewidth=2,
                    solid_capstyle="round", zorder=4)
            ax.scatter([m], [y], s=64, color=SERIES[sc], zorder=5,
                       edgecolors=SURFACE, linewidths=1.5)
            ax.annotate(f"{m:.3f}", (m, y), textcoords="offset points", xytext=(0, 9),
                        ha="center", fontsize=8, color=INK2, zorder=6)
        if i:
            ax.axhline(base_y + band / 2, color=GRID, linewidth=1, zorder=1)
    ax.axvline(0.5, color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate("chance (0.50)", (0.5, 0.72), fontsize=8.5, color=MUTED,
                ha="center", va="bottom")
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=9, color=INK)
    ax.set_ylim(-len(SETS) * band + band / 2 - 0.15, band / 2 + 0.25)
    ax.set_xlim(0.30, 1.0)
    ax.set_xlabel("Length-controlled pairwise accuracy   (mean of 3 seeds, bars = 95% CI)",
                  fontsize=9.5, color=INK2, labelpad=9)
    ax.set_title("Which reward generalizes? Length-controlled accuracy by test set",
                 fontsize=13, color=INK, pad=46, loc="left", fontweight="bold")
    ax.annotate("Hollow rings are individual seeds. Bars that cross the chance line, or that "
                "overlap each other,\nare not resolvable at 3 seeds.",
                xy=(0, 1.012), xycoords="axes fraction", fontsize=9, color=INK2, va="bottom")
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="none", markersize=8,
                              color=SERIES[k], label=LABEL[k]) for k in ORDER],
              loc="lower right", frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    p = os.path.join(FIG, "fig1_headline.png")
    fig.savefig(p, dpi=200, facecolor=SURFACE); plt.close(fig)
    return p


# ------------------------------------------------- fig 2: how much was just length?
def fig2(agg):
    fig, ax = plt.subplots(figsize=(10, 7.2), facecolor=SURFACE)
    style(ax)
    band, off = 1.0, 0.26
    yt, yl = [], []
    for i, s in enumerate(SETS):
        base_y = -i * band
        yt.append(base_y); yl.append(SHORT[s])
        for j, sc in enumerate(ORDER):
            raw = agg[s][sc]["accuracy"]; lc = agg[s][sc]["len_controlled_acc"]
            if not raw or not lc or lc.get("mean") is None:
                continue
            y = base_y + (1 - j) * off
            a, b = raw["mean"], lc["mean"]
            ax.annotate("", xy=(b, y), xytext=(a, y),
                        arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.5",
                                        color=SERIES[sc], linewidth=2, shrinkA=0, shrinkB=0))
            ax.scatter([a], [y], s=30, facecolors=SURFACE, edgecolors=SERIES[sc],
                       linewidths=1.6, zorder=5)
            ax.scatter([b], [y], s=64, color=SERIES[sc], zorder=5,
                       edgecolors=SURFACE, linewidths=1.5)
            d = b - a
            if abs(d) >= 0.03:  # label only the shifts big enough to matter
                ax.annotate(f"{d:+.2f}", (b, y), textcoords="offset points",
                            xytext=(14 if d > 0 else -14, 0), ha="left" if d > 0 else "right",
                            va="center", fontsize=8, color=INK2, zorder=6)
        if i:
            ax.axhline(base_y + band / 2, color=GRID, linewidth=1, zorder=1)
    ax.axvline(0.5, color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate("chance (0.50)", (0.5, 0.72), fontsize=8.5, color=MUTED, ha="center", va="bottom")
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=9, color=INK)
    ax.set_ylim(-len(SETS) * band + band / 2 - 0.15, band / 2 + 0.25)
    ax.set_xlim(0.30, 1.0)
    ax.set_xlabel("Pairwise accuracy   (hollow = raw  →  filled = length-controlled)",
                  fontsize=9.5, color=INK2, labelpad=9)
    ax.set_title("How much of each score was just “prefer the longer answer”?",
                 fontsize=13, color=INK, pad=46, loc="left", fontweight="bold")
    ax.annotate("Each arrow runs from raw accuracy to accuracy on length-matched pairs. A long "
                "arrow means the raw\nscore was largely a length signal, not preference quality.",
                xy=(0, 1.012), xycoords="axes fraction", fontsize=9, color=INK2, va="bottom")
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="none", markersize=8,
                              color=SERIES[k], label=LABEL[k]) for k in ORDER],
              loc="lower right", frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    p = os.path.join(FIG, "fig2_length_shift.png")
    fig.savefig(p, dpi=200, facecolor=SURFACE); plt.close(fig)
    return p


# ------------------------------------- fig 3: which gaps survive the seed noise?
def fig3(seeds, tags):
    fig, ax = plt.subplots(figsize=(10, 5.6), facecolor=SURFACE)
    style(ax)
    ys, labels, means, errs = [], [], [], []
    for i, s in enumerate(SETS):
        per = []
        for t in tags:
            e = seeds[t][s]["explicit"].get("len_controlled_acc")
            m = seeds[t][s]["implicit"].get("len_controlled_acc")
            if e is not None and m is not None:
                per.append(e - m)
        if not per:
            continue
        ys.append(-i); labels.append(SHORT[s].replace("\n", " "))
        means.append(float(np.mean(per))); errs.append(ci(per))
        ax.scatter(per, [-i] * len(per), s=26, facecolors="none",
                   edgecolors=MUTED, linewidths=1.2, alpha=0.8, zorder=3)
    for y, m, e in zip(ys, means, errs):
        # resolvable == the interval excludes zero
        col = SERIES["explicit"] if (m - e) > 0 else (SERIES["implicit"] if (m + e) < 0 else MUTED)
        ax.plot([m - e, m + e], [y, y], color=col, linewidth=2, solid_capstyle="round", zorder=4)
        ax.scatter([m], [y], s=64, color=col, zorder=5, edgecolors=SURFACE, linewidths=1.5)
        ax.annotate(f"{m:+.3f}", (m, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color=INK2, zorder=6)
    ax.axvline(0, color=INK, linewidth=1.5, zorder=2)
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.75)
    ax.set_xlabel("Explicit − implicit, length-controlled accuracy   (bars = 95% CI over 3 seeds)",
                  fontsize=9.5, color=INK2, labelpad=9)
    ax.set_title("Which differences survive seed noise?", fontsize=13, color=INK,
                 pad=34, loc="left", fontweight="bold")
    ax.annotate("Paired per seed, so run-to-run variation cancels. A bar crossing zero means the "
                "two rewards are\nnot distinguishable on that set at 3 seeds — grey marks exactly "
                "those.", xy=(0, 1.012), xycoords="axes fraction", fontsize=9, color=INK2, va="bottom")
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markersize=8, color=SERIES["explicit"],
               label="explicit better (CI excludes 0)"),
        Line2D([], [], marker="o", linestyle="none", markersize=8, color=SERIES["implicit"],
               label="implicit better (CI excludes 0)"),
        Line2D([], [], marker="o", linestyle="none", markersize=8, color=MUTED,
               label="not resolvable at 3 seeds")],
        loc="lower right", frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    p = os.path.join(FIG, "fig3_paired_delta.png")
    fig.savefig(p, dpi=200, facecolor=SURFACE); plt.close(fig)
    return p


if __name__ == "__main__":
    agg, seeds, tags = load()
    for p in (fig1(agg, seeds, tags), fig2(agg), fig3(seeds, tags)):
        print("wrote", p)
