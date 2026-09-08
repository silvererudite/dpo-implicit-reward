"""
Convergence figure: held-out accuracy across 4 epochs for both scorers.

Answers the strongest objection to the one-epoch comparison -- that it compared two models
which were both still improving, and which need not converge at the same rate, so the ranking
might invert with more training.

Both scorers are evaluated on the SAME held-out slice at the same cadence, so the curves are
directly comparable. Read from the training logs rather than W&B so the figure rebuilds offline.

    python reports/make_convergence_figure.py
"""
import glob, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "reports", "figures")
LOGS = glob.glob("/mnt/custom-file-systems/efs/fs-*/dpo-project/logs")
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8a8a85"; GRID = "#e4e4e0"
C_IMP, C_EXP = "#2a78d6", "#eb6834"


def series(path, key):
    if not os.path.exists(path):
        return []
    txt = open(path, errors="ignore").read().replace("\r", "\n")
    out = []
    for m in re.finditer(r"\{'eval_[^}]*\}", txt):
        try:
            d = eval(m.group(0))
        except Exception:
            continue
        if key in d and d.get("epoch") is not None:
            out.append((float(d["epoch"]), float(d[key])))
    return sorted(out)


def band(ax, runs, color, label):
    """Mean across seeds with a min-max band, on the epochs every seed reached."""
    if not runs:
        return None
    common = sorted(set.intersection(*[{round(e, 2) for e, _ in r} for r in runs]))
    if not common:
        return None
    M = np.array([[dict((round(e, 2), a) for e, a in r)[c] for c in common] for r in runs])
    mean = M.mean(axis=0)
    ax.fill_between(common, M.min(axis=0), M.max(axis=0), color=color, alpha=0.16, linewidth=0)
    ax.plot(common, mean, color=color, linewidth=2.4, label=label, zorder=4)
    i = int(np.argmax(mean))
    ax.scatter([common[i]], [mean[i]], s=95, color=color, edgecolors=SURFACE,
               linewidths=1.8, zorder=6)
    ax.annotate(f"peak {mean[i]:.3f}\nepoch {common[i]:.1f}", (common[i], mean[i]),
                textcoords="offset points", xytext=(0, 16), ha="center",
                fontsize=8.5, color=color, fontweight="bold", zorder=7)
    return common, mean


def main():
    if not LOGS:
        sys.exit("no log directory found")
    L = LOGS[0]
    dpo = [series(f"{L}/dpo_long_s{s}.log", "eval_rewards/accuracies") for s in (0, 1)]
    rm = [series(f"{L}/rm_long_s{s}.log", "eval_accuracy") for s in (0, 1)]
    dpo = [d for d in dpo if d]; rm = [r for r in rm if r]

    fig, ax = plt.subplots(figsize=(11, 6.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(color=GRID, linewidth=1, alpha=0.9); ax.set_axisbelow(True)

    ax.axvline(1.0, color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)), zorder=2)
    # anchor in axes fraction on y so the label cannot fall outside the data range
    ax.annotate("where the headline\nexperiment stopped", xy=(1.0, 0.10),
                xycoords=("data", "axes fraction"), xytext=(10, 0),
                textcoords="offset points", fontsize=8.5, color=MUTED, va="center")
    band(ax, rm, C_EXP, "Explicit reward model")
    band(ax, dpo, C_IMP, "DPO implicit reward")

    ax.set_xlabel("training epochs", fontsize=10, color=INK2, labelpad=8)
    ax.set_ylabel("held-out pairwise accuracy (400 shared pairs)", fontsize=10, color=INK2)
    ax.set_title("Does the ranking survive training to convergence?", fontsize=13.5, color=INK,
                 pad=46, loc="left", fontweight="bold")
    ax.annotate("Shaded band spans the two seeds. The explicit model climbs to a peak near epoch 2 and "
                "then overfits;\nDPO's implicit reward is flat from the start. The explicit model leads "
                "at every epoch — the ranking does not invert.",
                xy=(0, 1.012), xycoords="axes fraction", fontsize=9.5, color=INK2, va="bottom")
    ax.legend(loc="lower left", frameon=False, fontsize=9.5, labelcolor=INK2)
    fig.tight_layout()
    p = os.path.join(FIG, "fig10_convergence.png")
    fig.savefig(p, dpi=200, facecolor=SURFACE); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
