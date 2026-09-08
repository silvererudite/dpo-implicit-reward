"""
Pull training curves from Weights & Biases and render them as static figures.

Why static copies: the W&B project is the live record, but a report and a git history need
figures that survive without network access or an account. This fetches the run histories,
writes them to CSV (so the curves are reproducible without W&B at all), and renders PNGs.

Note on what exists: DPOTrainer logs a rich set (loss, reward accuracy, margins, chosen/rejected
rewards). TRL 0.15's RewardTrainer logs only loss / grad-norm / LR -- no accuracy -- so there is
no RM accuracy curve to plot. There are no "eval curves" either: evaluation is a single scoring
pass at the end, not a training loop, so its results live in the figures built by make_figures.py.

    python reports/make_wandb_curves.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "reports", "figures")
CSV = os.path.join(ROOT, "results", "curves")
os.makedirs(FIG, exist_ok=True); os.makedirs(CSV, exist_ok=True)
PROJECT = "shamima2hossain/dpo-implicit-reward"

SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8a8a85"; GRID = "#e4e4e0"
SEED_COLOR = {0: "#2a78d6", 1: "#eb6834", 2: "#1baf7a"}   # validated categorical slots 1-3


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    ax.grid(color=GRID, linewidth=1, alpha=0.9); ax.set_axisbelow(True)
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=6, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=8.5, color=INK2)
    ax.set_xlabel("optimizer step", fontsize=8.5, color=INK2)


def fetch():
    """Newest run per name -- run names repeat across the August and September experiments."""
    import wandb
    api = wandb.Api()
    latest = {}
    for r in api.runs(PROJECT):
        if not r.name or not (r.name.startswith("dpo-") or r.name.startswith("rm-")):
            continue
        prev = latest.get(r.name)
        if prev is None or str(r.created_at) > str(prev.created_at):
            latest[r.name] = r
    return latest


def history(run, keys):
    df = run.history(keys=keys, pandas=True)
    return df.dropna(how="all", subset=[k for k in keys if k in df.columns])


def panel(ax, runs, key, title, ylabel, second=None, second_label=None):
    plotted = False
    for seed, run in sorted(runs.items()):
        keys = [key, "train/global_step"] + ([second] if second else [])
        df = history(run, keys)
        if key not in df.columns or df.empty:
            continue
        # global_step, not _step: _step counts LOGGING events (every 20 optimizer steps),
        # so plotting it would make a 500-step run look like a 25-step one.
        x = df["train/global_step"] if "train/global_step" in df.columns else df.get("_step", range(len(df)))
        ax.plot(x, df[key], color=SEED_COLOR[seed], linewidth=2, label=f"seed {seed}")
        if second and second in df.columns:
            ax.plot(x, df[second], color=SEED_COLOR[seed], linewidth=2,
                    linestyle=(0, (3, 2)), alpha=0.85)
        plotted = True
    style(ax, title, ylabel)
    return plotted


def main():
    latest = fetch()
    dpo = {int(n[-1]): r for n, r in latest.items() if n.startswith("dpo-beta0.1-8k-seed")}
    rm = {int(n[-1]): r for n, r in latest.items() if n.startswith("rm-8k-seed")}
    print(f"DPO runs: {sorted(dpo)}   RM runs: {sorted(rm)}")

    # ---- CSV so the curves survive without W&B ----
    for tag, runs, keys in (("dpo", dpo, ["train/loss", "train/rewards/accuracies",
                                          "train/rewards/margins", "train/rewards/chosen",
                                          "train/rewards/rejected", "train/grad_norm",
                                          "train/learning_rate"]),
                            ("rm", rm, ["train/loss", "train/grad_norm", "train/learning_rate"])):
        for seed, run in sorted(runs.items()):
            df = history(run, keys)
            p = os.path.join(CSV, f"{tag}_seed{seed}.csv")
            df.to_csv(p, index=False)
            print("wrote", p)

    # ---- DPO curves ----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), facecolor=SURFACE)
    panel(axes[0][0], dpo, "train/loss", "DPO loss", "loss")
    axes[0][0].axhline(0.6931, color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)))
    axes[0][0].annotate("ln 2 — value at initialization,\nwhen the implicit reward is exactly 0",
                        xy=(0.97, 0.6931), xycoords=("axes fraction", "data"),
                        fontsize=7.5, color=MUTED, ha="right", va="bottom")
    panel(axes[0][1], dpo, "train/rewards/accuracies", "Implicit reward: train accuracy", "accuracy")
    axes[0][1].axhline(0.5, color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)))
    axes[0][1].annotate("chance", xy=(0.97, 0.5), xycoords=("axes fraction", "data"),
                        fontsize=7.5, color=MUTED, ha="right", va="bottom")
    panel(axes[0][2], dpo, "train/rewards/margins", "Reward margin (chosen − rejected)", "margin")
    axes[0][2].axhline(0, color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)))
    panel(axes[1][0], dpo, "train/rewards/chosen", "Implicit reward level\n(solid = chosen, dashed = rejected)",
          "reward", second="train/rewards/rejected")
    axes[1][0].axhline(0, color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)))
    panel(axes[1][1], dpo, "train/grad_norm", "Gradient norm", "‖g‖")
    panel(axes[1][2], dpo, "train/learning_rate", "Learning rate (cosine)", "lr")
    fig.suptitle("DPO training curves — 3 seeds, β=0.1, 8k pairs",
                 fontsize=14, color=INK, x=0.008, ha="left", fontweight="bold", y=0.995)
    fig.legend(handles=[Line2D([], [], color=SEED_COLOR[s], linewidth=2, label=f"seed {s}")
                        for s in sorted(dpo)],
               loc="upper right", frameon=False, fontsize=9, ncol=3,
               bbox_to_anchor=(0.995, 1.005), labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    p = os.path.join(FIG, "fig4_dpo_curves.png")
    fig.savefig(p, dpi=200, facecolor=SURFACE); plt.close(fig); print("wrote", p)

    # ---- RM curves ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=SURFACE)
    panel(axes[0], rm, "train/loss", "Bradley–Terry loss", "loss")
    axes[0].axhline(0.6931, color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)))
    axes[0].annotate("ln 2 — chance separation", xy=(0.97, 0.6931),
                     xycoords=("axes fraction", "data"), fontsize=7.5, color=MUTED,
                     ha="right", va="bottom")
    panel(axes[1], rm, "train/grad_norm", "Gradient norm", "‖g‖")
    panel(axes[2], rm, "train/learning_rate", "Learning rate (cosine)", "lr")
    fig.suptitle("Explicit reward-model training curves — 3 seeds, 8k pairs "
                 "(TRL logs no accuracy for RewardTrainer)",
                 fontsize=14, color=INK, x=0.008, ha="left", fontweight="bold", y=0.995)
    fig.legend(handles=[Line2D([], [], color=SEED_COLOR[s], linewidth=2, label=f"seed {s}")
                        for s in sorted(rm)],
               loc="upper right", frameon=False, fontsize=9, ncol=3,
               bbox_to_anchor=(0.995, 1.01), labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    p = os.path.join(FIG, "fig5_rm_curves.png")
    fig.savefig(p, dpi=200, facecolor=SURFACE); plt.close(fig); print("wrote", p)


if __name__ == "__main__":
    main()
