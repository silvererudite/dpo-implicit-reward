"""
Extract every training curve to CSV + PNG so they outlive W&B and work offline.

Parses the trainer logs directly rather than querying W&B, because (a) it works without an
account or network, and (b) W&B run names do not encode the epoch budget, so the 1-epoch and
4-epoch runs collide under the same name and "latest run wins" would silently mix them.

Writes:
  results/curves/<run>.csv          one row per logged step, all metrics
  reports/figures/fig11_long_curves.png

    python reports/extract_curves.py
"""
import csv, glob, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "reports", "figures")
CSV = os.path.join(ROOT, "results", "curves")
os.makedirs(FIG, exist_ok=True); os.makedirs(CSV, exist_ok=True)
LOGS = glob.glob("/mnt/custom-file-systems/efs/fs-*/dpo-project/logs")[0]

SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8a8a85"; GRID = "#e4e4e0"
SEED_C = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a"}


def parse(path):
    """Every dict the trainer printed: train rows and eval rows, kept separate."""
    if not os.path.exists(path):
        return [], []
    txt = open(path, errors="ignore").read().replace("\r", "\n")
    train, ev = [], []
    for m in re.finditer(r"\{'(?:loss|eval_)[^}]*\}", txt):
        try:
            d = eval(m.group(0))
        except Exception:
            continue
        (ev if any(k.startswith("eval_") for k in d) else train).append(d)
    return train, ev


def write_csv(name, rows):
    if not rows:
        return None
    keys = sorted({k for r in rows for k in r})
    p = os.path.join(CSV, f"{name}.csv")
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    return p


def style(ax, title, ylabel, xlabel="training epochs"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=INK2, labelsize=8.5, length=0)
    ax.grid(color=GRID, linewidth=1, alpha=0.9); ax.set_axisbelow(True)
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=6, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=8.5, color=INK2)
    ax.set_xlabel(xlabel, fontsize=8.5, color=INK2)


def line(ax, data, xkey, ykey):
    for seed, rows in sorted(data.items()):
        pts = [(r[xkey], r[ykey]) for r in rows if xkey in r and ykey in r]
        if pts:
            x, y = zip(*sorted(pts))
            ax.plot(x, y, color=SEED_C[seed], linewidth=2, label=f"seed {seed}")


def main():
    runs = {}
    for tag, pat in (("dpo_long", "dpo_long_s{}.log"), ("rm_long", "rm_long_s{}.log"),
                     ("dpo_1ep", "dpo_s{}.log"), ("rm_1ep", "rm_s{}.log")):
        for s in range(5):
            p = os.path.join(LOGS, pat.format(s))
            tr, ev = parse(p)
            if tr or ev:
                runs[(tag, s)] = (tr, ev)
                for kind, rows in (("train", tr), ("eval", ev)):
                    out = write_csv(f"{tag}_s{s}_{kind}", rows)
                    if out:
                        print("wrote", os.path.relpath(out, ROOT))

    dpo_tr = {s: r[0] for (t, s), r in runs.items() if t == "dpo_long"}
    dpo_ev = {s: r[1] for (t, s), r in runs.items() if t == "dpo_long"}
    rm_tr = {s: r[0] for (t, s), r in runs.items() if t == "rm_long"}
    rm_ev = {s: r[1] for (t, s), r in runs.items() if t == "rm_long"}
    if not dpo_tr:
        print("no long-run logs found; skipping figure"); return

    fig, ax = plt.subplots(2, 3, figsize=(15, 8), facecolor=SURFACE)
    line(ax[0][0], dpo_tr, "epoch", "loss");            style(ax[0][0], "DPO training loss", "loss")
    ax[0][0].axhline(0.6931, color=MUTED, lw=1.4, ls=(0, (4, 3)))
    line(ax[0][1], dpo_ev, "epoch", "eval_loss");       style(ax[0][1], "DPO held-out loss", "loss")
    line(ax[0][2], dpo_ev, "epoch", "eval_rewards/accuracies")
    style(ax[0][2], "DPO held-out accuracy", "accuracy")
    line(ax[1][0], rm_tr, "epoch", "loss");             style(ax[1][0], "Reward-model training loss", "loss")
    ax[1][0].axhline(0.6931, color=MUTED, lw=1.4, ls=(0, (4, 3)))
    line(ax[1][1], rm_ev, "epoch", "eval_loss");        style(ax[1][1], "Reward-model held-out loss", "loss")
    line(ax[1][2], rm_ev, "epoch", "eval_accuracy")
    style(ax[1][2], "Reward-model held-out accuracy", "accuracy")
    for a in (ax[0][2], ax[1][2]):
        a.axhline(0.5, color=MUTED, lw=1.2, ls=(0, (4, 3)))

    fig.suptitle("Convergence run — 4 epochs, 2 seeds, β=0.1, 8k pairs",
                 fontsize=14, color=INK, x=0.007, ha="left", fontweight="bold", y=0.995)
    fig.text(0.007, 0.945, "Held-out columns are the generalization signal; training loss keeps "
             "falling after held-out accuracy has peaked, which is why the training curve alone "
             "cannot tell you when to stop.", fontsize=9, color=INK2)
    fig.legend(handles=[Line2D([], [], color=SEED_C[s], lw=2, label=f"seed {s}") for s in sorted(dpo_tr)],
               loc="upper right", frameon=False, fontsize=9, ncol=2,
               bbox_to_anchor=(0.995, 1.005), labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    p = os.path.join(FIG, "fig11_long_curves.png")
    fig.savefig(p, dpi=200, facecolor=SURFACE); plt.close(fig)
    print("wrote", os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
