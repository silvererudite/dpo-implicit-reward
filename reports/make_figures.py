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
         "base_logprob": "SFT baseline (no reward training)"}
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




# ============================================================ alternative encodings
DIV_POS = "#2a78d6"   # diverging warm/cool poles: blue = training helped
DIV_NEG = "#e34948"   # red = training actively hurt
DIV_MID = "#b8b7b2"   # neutral: difference not distinguishable from zero


def _lc_per_seed(seeds, tags, s, scorer):
    return np.array([seeds[t][s][scorer]["len_controlled_acc"] for t in tags
                     if seeds[t][s][scorer].get("len_controlled_acc") is not None])


def fig6_value_added(agg, seeds, tags):
    """Did reward training help at all, versus not training one?

    Reframes every number against the SFT log-probability baseline. Zero is not
    'chance' here -- it is 'what you get without reward-specific training', which is the
    comparison that actually matters if the question is whether to train a reward model
    at all.
    """
    fig, ax = plt.subplots(figsize=(10.5, 7.4), facecolor=SURFACE)
    style(ax)
    t = T95.get(len(tags), 2.0)
    yt, yl, heads = [], [], []
    row = 0.0
    for s in SETS:
        base = agg[s]["base_logprob"]["len_controlled_acc"]["mean"]
        heads.append((row + 0.78, s, base))   # label sits above each pair of bars
        for scorer, name in (("implicit", "DPO implicit"), ("explicit", "Explicit RM")):
            v = _lc_per_seed(seeds, tags, s, scorer) - base
            m = v.mean(); e = t * v.std(ddof=1) / np.sqrt(len(v))
            col = DIV_POS if (m - e) > 0 else (DIV_NEG if (m + e) < 0 else DIV_MID)
            ax.barh(row, m, height=0.62, color=col, edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.plot([m - e, m + e], [row, row], color=INK, linewidth=1.4, alpha=0.55, zorder=4)
            anchor = (m + e) if m >= 0 else (m - e)   # clear the whisker, not just the bar
            ax.annotate(f"{m:+.3f}", (anchor, row), textcoords="offset points",
                        xytext=(8 if m >= 0 else -8, 0), ha="left" if m >= 0 else "right",
                        va="center", fontsize=8.5, color=INK2, zorder=5)
            yt.append(row); yl.append(name)
            row -= 1.0
        row -= 0.55
    ax.axvline(0, color=INK, linewidth=2, zorder=2)
    for y, s, base in heads:
        ax.annotate(f"{SHORT[s].replace(chr(10), ' ')}   ·   SFT baseline = {base:.3f}",
                    xy=(0.012, y), xycoords=("axes fraction", "data"),
                    va="center", ha="left", fontsize=9.5, color=INK,
                    fontweight="bold", annotation_clip=False)
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=9, color=INK2)
    ax.set_ylim(row + 0.6, 1.25)
    ax.set_xlim(-0.33, 0.40)
    ax.set_xlabel("Accuracy minus the SFT baseline   (bars = 95% CI over "
                  f"{len(tags)} seeds)", fontsize=9.5, color=INK2, labelpad=9)
    ax.set_title("Did training a reward model help — or hurt?", fontsize=13.5, color=INK,
                 pad=46, loc="left", fontweight="bold")
    ax.annotate("Zero is the SFT baseline: the plain model's log-probability, no reward-specific "
                "training. Bars left of zero mean\ntraining made the scorer WORSE than not "
                "training one. Grey = not distinguishable from the baseline.",
                xy=(0, 1.012), xycoords="axes fraction", fontsize=9, color=INK2, va="bottom")
    ax.legend(handles=[Line2D([], [], marker="s", linestyle="none", markersize=9, color=DIV_POS,
                              label="training helped"),
                       Line2D([], [], marker="s", linestyle="none", markersize=9, color=DIV_NEG,
                              label="training hurt"),
                       Line2D([], [], marker="s", linestyle="none", markersize=9, color=DIV_MID,
                              label="no detectable difference")],
              loc="lower left", frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    p = os.path.join(FIG, "fig6_value_added.png")
    fig.savefig(p, dpi=200, facecolor=SURFACE); plt.close(fig)
    return p


def fig7_small_multiples(agg, seeds, tags):
    """Same data as fig1, one panel per test set -- less crowded, and it gives each set room
    to carry its own baseline reference and its length-matched sample size."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.4), facecolor=SURFACE)
    t = T95.get(len(tags), 2.0)
    for ax, s in zip(axes.ravel(), SETS):
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID); ax.spines["bottom"].set_linewidth(1)
        ax.tick_params(colors=INK2, labelsize=8.5, length=0)
        ax.grid(axis="x", color=GRID, linewidth=1, alpha=0.9); ax.set_axisbelow(True)
        base = agg[s]["base_logprob"]["len_controlled_acc"]["mean"]
        ax.axvspan(0.30, 0.5, color=GRID, alpha=0.45, zorder=0)      # worse than chance
        ax.axvline(0.5, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
        ax.axvline(base, color=SERIES["base_logprob"], linewidth=2, zorder=2)
        for j, sc in enumerate(("implicit", "explicit")):
            v = _lc_per_seed(seeds, tags, s, sc)
            m = v.mean(); e = t * v.std(ddof=1) / np.sqrt(len(v))
            y = 1 - j
            ax.scatter(v, [y] * len(v), s=22, facecolors="none", edgecolors=SERIES[sc],
                       linewidths=1.1, alpha=0.75, zorder=3)
            ax.plot([m - e, m + e], [y, y], color=SERIES[sc], linewidth=2.4,
                    solid_capstyle="round", zorder=4)
            ax.scatter([m], [y], s=70, color=SERIES[sc], zorder=5,
                       edgecolors=SURFACE, linewidths=1.5)
            ax.annotate(f"{m:.3f}", (m, y), textcoords="offset points", xytext=(0, 11),
                        ha="center", fontsize=8.5, color=INK2, zorder=6)
        ax.set_yticks([1, 0]); ax.set_yticklabels(["implicit", "explicit"], fontsize=9, color=INK2)
        ax.set_ylim(-0.75, 1.85); ax.set_xlim(0.30, 1.0)
        n = agg[s]["implicit"]["len_controlled_n"]
        warn = "  ⚠ too few to conclude" if n < 100 else ""
        ax.set_title(f"{SHORT[s].replace(chr(10), ' ')}\n{n} length-matched pairs{warn}",
                     fontsize=10, color=INK, loc="left", pad=6, fontweight="bold")
    fig.suptitle("Length-controlled accuracy, one panel per test set",
                 fontsize=13.5, color=INK, x=0.006, ha="left", fontweight="bold", y=0.995)
    fig.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markersize=8, color=SERIES["implicit"],
               label="DPO implicit"),
        Line2D([], [], marker="o", linestyle="none", markersize=8, color=SERIES["explicit"],
               label="Explicit RM"),
        Line2D([], [], color=SERIES["base_logprob"], linewidth=2, label="SFT baseline"),
        Line2D([], [], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)),
               label="chance (shaded = worse than chance)")],
        loc="upper right", frameon=False, fontsize=9, ncol=4,
        bbox_to_anchor=(0.998, 1.008), labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    p = os.path.join(FIG, "fig7_small_multiples.png")
    fig.savefig(p, dpi=200, facecolor=SURFACE); plt.close(fig)
    return p


def fig8_head_to_head(agg, seeds, tags):
    """One point per test set: implicit vs explicit, with the diagonal as 'no difference'.
    Marker area encodes the length-matched sample size, so the sets that cannot support a
    conclusion are visibly small rather than silently equal-weighted."""
    fig, ax = plt.subplots(figsize=(8.4, 7.6), facecolor=SURFACE)
    style(ax); ax.grid(axis="y", color=GRID, linewidth=1, alpha=0.9)
    lo, hi = 0.38, 0.92
    ax.plot([lo, hi], [lo, hi], color=INK, linewidth=1.6, zorder=2)
    ax.annotate("equal performance", xy=(0.66, 0.66), rotation=45, fontsize=8.5,
                color=INK2, ha="center", va="bottom")
    ax.axvline(0.5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.axhline(0.5, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate("explicit better ↑", xy=(0.40, 0.895), fontsize=9, color=INK2)
    ax.annotate("implicit better →", xy=(0.70, 0.405), fontsize=9, color=INK2)
    for s in SETS:
        im = _lc_per_seed(seeds, tags, s, "implicit").mean()
        ex = _lc_per_seed(seeds, tags, s, "explicit").mean()
        n = agg[s]["implicit"]["len_controlled_n"] or 1
        # both must clear the SFT baseline for reward training to have earned its keep
        beats_base = min(im, ex) > agg[s]["base_logprob"]["len_controlled_acc"]["mean"]
        ax.scatter([im], [ex], s=40 + 160 * np.log10(n) / np.log10(1000),
                   color=(SERIES["explicit"] if beats_base else DIV_MID),
                   edgecolors=SURFACE, linewidths=1.8, zorder=5, alpha=0.95)
        ax.annotate(f"{SHORT[s].replace(chr(10), ' ')}  (n={n})", (im, ex),
                    textcoords="offset points", xytext=(0, 15), ha="center",
                    fontsize=8.5, color=INK2, zorder=6)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("DPO implicit reward — length-controlled accuracy", fontsize=9.5, color=INK2)
    ax.set_ylabel("Explicit reward model — length-controlled accuracy", fontsize=9.5, color=INK2)
    ax.set_title("Head to head, by test set", fontsize=13.5, color=INK, pad=46,
                 loc="left", fontweight="bold")
    ax.annotate("Marker area grows with the number of length-matched pairs, so small, unreliable "
                "sets look small.\nPoints above the line: the explicit RM wins. Below: DPO's implicit "
                "reward wins.",
                xy=(0, 1.012), xycoords="axes fraction", fontsize=9, color=INK2, va="bottom")
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markersize=9, color=SERIES["explicit"],
               label="both scorers beat the SFT baseline"),
        Line2D([], [], marker="o", linestyle="none", markersize=9, color=DIV_MID,
               label="at least one loses to the SFT baseline")],
        loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.tight_layout()
    p = os.path.join(FIG, "fig8_head_to_head.png")
    fig.savefig(p, dpi=200, facecolor=SURFACE); plt.close(fig)
    return p

if __name__ == "__main__":
    agg, seeds, tags = load()
    print(f"aggregating {len(tags)} seeds: {', '.join(tags)}")
    for p in (fig1(agg, seeds, tags), fig2(agg), fig3(seeds, tags),
              fig6_value_added(agg, seeds, tags), fig7_small_multiples(agg, seeds, tags),
              fig8_head_to_head(agg, seeds, tags)):
        print("wrote", p)
