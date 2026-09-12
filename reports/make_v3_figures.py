"""
Report figures, v3 — rebuilt for the final report.

Differences from the reports/ figure scripts this supersedes:
  * seed counts read from the data instead of a stale hard-coded "3 seeds";
  * the explanatory subtitle baked into each PNG is dropped — it duplicated the LaTeX
    caption and the body paragraph, and removing it lets each figure be set smaller;
  * fig12 keeps only the length-preference panel (accuracy over the same checkpoints is
    already fig10);
  * fig9's bottom panel carries one worked example rather than three.

Reads only committed artifacts (results/*.json, results/curves/*.csv), so it rebuilds off
the GPU box. Writes straight into final_report/figures/.

    python reports/make_v3_figures.py
"""
import csv, json, os, re, sys, textwrap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "final_report", "figures")
CURVES = os.path.join(ROOT, "results", "curves")
os.makedirs(FIG, exist_ok=True)

SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8a8a85"; GRID = "#e4e4e0"
SERIES = {"implicit": "#2a78d6", "explicit": "#eb6834", "base_logprob": "#1baf7a"}
LABEL = {"implicit": "DPO implicit", "explicit": "Explicit RM", "base_logprob": "SFT baseline"}
ORDER = ["implicit", "explicit", "base_logprob"]
DIV_POS = "#2a78d6"; DIV_NEG = "#e34948"; DIV_MID = "#b8b7b2"
SETS = ["UltraFeedback (ID)", "RewardBench:Chat (OOD)", "RewardBench:Chat-Hard (OOD)",
        "RewardBench:Safety (OOD)", "RewardBench:Reasoning (OOD)", "HH-harmless (OOD)"]
SHORT = {"UltraFeedback (ID)": "UltraFeedback (in-distribution)",
         "RewardBench:Chat (OOD)": "RewardBench Chat",
         "RewardBench:Chat-Hard (OOD)": "RewardBench Chat-Hard",
         "RewardBench:Safety (OOD)": "RewardBench Safety",
         "RewardBench:Reasoning (OOD)": "RewardBench Reasoning",
         "HH-harmless (OOD)": "HH-RLHF harmless"}
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


def _lc_per_seed(seeds, tags, s, scorer):
    return np.array([seeds[t][s][scorer]["len_controlled_acc"] for t in tags
                     if seeds[t][s][scorer].get("len_controlled_acc") is not None])


# --------------------------------------------- fig 3: which gaps survive seed noise?
def fig3(seeds, tags):
    n = len(tags)
    fig, ax = plt.subplots(figsize=(10, 4.6), facecolor=SURFACE)
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
        ys.append(-i); labels.append(SHORT[s])
        means.append(float(np.mean(per))); errs.append(ci(per))
        ax.scatter(per, [-i] * len(per), s=26, facecolors="none",
                   edgecolors=MUTED, linewidths=1.2, alpha=0.8, zorder=3)
    for y, m, e in zip(ys, means, errs):
        col = SERIES["explicit"] if (m - e) > 0 else (SERIES["implicit"] if (m + e) < 0 else MUTED)
        ax.plot([m - e, m + e], [y, y], color=col, linewidth=2, solid_capstyle="round", zorder=4)
        ax.scatter([m], [y], s=64, color=col, zorder=5, edgecolors=SURFACE, linewidths=1.5)
        ax.annotate(f"{m:+.3f}", (m, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color=INK2, zorder=6)
    ax.axvline(0, color=INK, linewidth=1.5, zorder=2)
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=9, color=INK)
    ax.set_ylim(min(ys) - 0.6, max(ys) + 0.75)
    ax.set_xlabel("Explicit − implicit, length-controlled accuracy   "
                  f"(bars = 95% CI over {n} seeds)", fontsize=9.5, color=INK2, labelpad=9)
    ax.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markersize=8, color=SERIES["explicit"],
               label="explicit better (CI excludes 0)"),
        Line2D([], [], marker="o", linestyle="none", markersize=8, color=SERIES["implicit"],
               label="implicit better (CI excludes 0)"),
        Line2D([], [], marker="o", linestyle="none", markersize=8, color=MUTED,
               label=f"not resolvable at {n} seeds")],
        loc="lower right", frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    p = os.path.join(FIG, "fig3_paired_delta.png")
    fig.savefig(p, dpi=200, facecolor=SURFACE); plt.close(fig)
    return p


# ------------------------------------------------ fig 6: value added over the baseline
def fig6_value_added(agg, seeds, tags):
    fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor=SURFACE)
    style(ax)
    t = T95.get(len(tags), 2.0)
    yt, yl, heads = [], [], []
    row = 0.0
    for s in SETS:
        base = agg[s]["base_logprob"]["len_controlled_acc"]["mean"]
        heads.append((row + 0.78, s, base))
        for scorer, name in (("implicit", "DPO implicit"), ("explicit", "Explicit RM")):
            v = _lc_per_seed(seeds, tags, s, scorer) - base
            m = v.mean(); e = t * v.std(ddof=1) / np.sqrt(len(v))
            col = DIV_POS if (m - e) > 0 else (DIV_NEG if (m + e) < 0 else DIV_MID)
            ax.barh(row, m, height=0.62, color=col, edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.plot([m - e, m + e], [row, row], color=INK, linewidth=1.4, alpha=0.55, zorder=4)
            anchor = (m + e) if m >= 0 else (m - e)
            ax.annotate(f"{m:+.3f}", (anchor, row), textcoords="offset points",
                        xytext=(8 if m >= 0 else -8, 0), ha="left" if m >= 0 else "right",
                        va="center", fontsize=8.5, color=INK2, zorder=5)
            yt.append(row); yl.append(name)
            row -= 1.0
        row -= 0.55
    ax.axvline(0, color=INK, linewidth=2, zorder=2)
    for y, s, base in heads:
        ax.annotate(f"{SHORT[s]}   ·   SFT baseline = {base:.3f}",
                    xy=(0.012, y), xycoords=("axes fraction", "data"),
                    va="center", ha="left", fontsize=9.5, color=INK,
                    fontweight="bold", annotation_clip=False)
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=9, color=INK2)
    ax.set_ylim(row + 0.6, 1.25)
    ax.set_xlim(-0.33, 0.40)
    ax.set_xlabel("Accuracy minus the SFT baseline   (bars = 95% CI over "
                  f"{len(tags)} seeds)", fontsize=9.5, color=INK2, labelpad=9)
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


# ------------------------------------------------------------ curve helpers (from CSV)
def curve(name):
    p = os.path.join(CURVES, f"{name}.csv")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [{k: (float(v) if v not in ("", None) else None) for k, v in r.items()}
                for r in csv.DictReader(f)]


def series(name, key):
    return sorted((r["epoch"], r[key]) for r in curve(name)
                  if r.get("epoch") is not None and r.get(key) is not None)


# ------------------------------------------------- fig 10: convergence over four epochs
def band(ax, runs, color, label):
    if not runs:
        return
    common = sorted(set.intersection(*[{round(e, 2) for e, _ in r} for r in runs]))
    if not common:
        return
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


def fig10_convergence():
    dpo = [s for s in (series(f"dpo_long_s{i}_eval", "eval_rewards/accuracies") for i in (0, 1)) if s]
    rm = [s for s in (series(f"rm_long_s{i}_eval", "eval_accuracy") for i in (0, 1)) if s]
    if not dpo or not rm:
        return None
    fig, ax = plt.subplots(figsize=(11, 5.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(1)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(color=GRID, linewidth=1, alpha=0.9); ax.set_axisbelow(True)
    ax.axvline(1.0, color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)), zorder=2)
    ax.annotate("where the headline\nexperiment stopped", xy=(1.0, 0.10),
                xycoords=("data", "axes fraction"), xytext=(10, 0),
                textcoords="offset points", fontsize=8.5, color=MUTED, va="center")
    band(ax, rm, SERIES["explicit"], "Explicit reward model")
    band(ax, dpo, SERIES["implicit"], "DPO implicit reward")
    ax.set_xlabel("training epochs", fontsize=10, color=INK2, labelpad=8)
    ax.set_ylabel("held-out pairwise accuracy (400 shared pairs)", fontsize=10, color=INK2)
    ax.legend(loc="lower left", frameon=False, fontsize=9.5, labelcolor=INK2)
    fig.tight_layout()
    p = os.path.join(FIG, "fig10_convergence.png")
    fig.savefig(p, dpi=200, facecolor=SURFACE); plt.close(fig)
    return p


# -------------------------------------------------- fig 11: train vs held-out loss
def fig11_long_curves():
    SEED_C = {0: "#2a78d6", 1: "#eb6834"}

    def line(ax, name, ykey):
        for seed in (0, 1):
            pts = series(name.format(seed), ykey)
            if pts:
                x, y = zip(*pts)
                ax.plot(x, y, color=SEED_C[seed], linewidth=2, label=f"seed {seed}")

    def st(ax, title, ylabel):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID); ax.spines[s].set_linewidth(1)
        ax.tick_params(colors=INK2, labelsize=8.5, length=0)
        ax.grid(color=GRID, linewidth=1, alpha=0.9); ax.set_axisbelow(True)
        ax.set_title(title, fontsize=10, color=INK, loc="left", pad=6, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=8.5, color=INK2)
        ax.set_xlabel("training epochs", fontsize=8.5, color=INK2)

    fig, ax = plt.subplots(2, 2, figsize=(10, 7), facecolor=SURFACE)
    line(ax[0][0], "dpo_long_s{}_train", "loss");     st(ax[0][0], "DPO training loss", "loss")
    ax[0][0].axhline(0.6931, color=MUTED, lw=1.4, ls=(0, (4, 3)))
    line(ax[0][1], "dpo_long_s{}_eval", "eval_loss"); st(ax[0][1], "DPO held-out loss", "loss")
    line(ax[1][0], "rm_long_s{}_train", "loss");      st(ax[1][0], "Reward-model training loss", "loss")
    ax[1][0].axhline(0.6931, color=MUTED, lw=1.4, ls=(0, (4, 3)))
    line(ax[1][1], "rm_long_s{}_eval", "eval_loss");  st(ax[1][1], "Reward-model held-out loss", "loss")
    fig.legend(handles=[Line2D([], [], color=SEED_C[s], lw=2, label=f"seed {s}") for s in (0, 1)],
               loc="upper right", frameon=False, fontsize=9, ncol=2,
               bbox_to_anchor=(0.995, 1.0), labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    p = os.path.join(FIG, "fig11_long_curves.png")
    fig.savefig(p, dpi=200, facecolor=SURFACE); plt.close(fig)
    return p


# ------------------------------------------------------- fig 12: length-bias drift
def fig12_bias_drift():
    data = {}
    for k in ("explicit", "implicit"):
        p = os.path.join(ROOT, "results", f"bias_over_training_{k}_s0.json")
        if os.path.exists(p):
            data[k] = json.load(open(p))["checkpoints"]
    if not data:
        return None
    LAB = {"implicit": "DPO implicit reward", "explicit": "Explicit reward model"}
    fig, a1 = plt.subplots(figsize=(9, 5.0), facecolor=SURFACE)
    a1.set_facecolor(SURFACE)
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a1.spines[s].set_color(GRID); a1.spines[s].set_linewidth(1)
    a1.tick_params(colors=INK2, labelsize=9, length=0)
    a1.grid(color=GRID, linewidth=1, alpha=0.9); a1.set_axisbelow(True)
    for k, rows in data.items():
        ep = [r["epoch"] for r in rows]
        bias = [r["rates"]["length"]["scorer"] - r["rates"]["length"]["human"] for r in rows]
        a1.plot(ep, bias, color=SERIES[k], linewidth=2.4, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.4, label=LAB[k], zorder=4)
    a1.axhline(0, color=INK, linewidth=1.8, zorder=2)
    a1.annotate("matches the human rate (humans pick longer 55.7% of the time)",
                xy=(0.98, 0.0), xycoords=("axes fraction", "data"), ha="right",
                xytext=(0, 6), textcoords="offset points",
                fontsize=8.5, color=MUTED, va="bottom")
    a1.annotate("prefers LONGER than humans", xy=(0.02, 0.93), xycoords="axes fraction",
                fontsize=8.5, color=MUTED)
    a1.annotate("prefers SHORTER than humans", xy=(0.02, 0.04), xycoords="axes fraction",
                fontsize=8.5, color=MUTED)
    a1.set_ylabel("scorer rate − human rate", fontsize=9, color=INK2)
    a1.set_xlabel("training epochs", fontsize=9, color=INK2)
    a1.legend(loc="center right", frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    p = os.path.join(FIG, "fig12_bias_drift.png")
    fig.savefig(p, dpi=200, facecolor=SURFACE); plt.close(fig)
    return p


# ------------------------------------------------ fig 9: what each scorer rewards
FEATURES = [("length", "longer response"), ("markdown", "bullet points / bold"),
            ("code", "code blocks"), ("digits", "numbers"),
            ("hedge", "hedging & disclaimers"), ("exclaim", "exclamation marks")]


def feats(t):
    return {
        "length":   len(t),
        "markdown": len(re.findall(r"(?m)^\s*(?:[-*•]|\d+\.)\s", t)) + t.count("**") + t.count("##"),
        "code":     t.count("```") + len(re.findall(r"(?m)^\s{4}\S", t)),
        "digits":   sum(c.isdigit() for c in t),
        "hedge":    len(re.findall(r"(?i)\b(as an ai|i cannot|i'm sorry|i am sorry|it depends|however|note that)\b", t)),
        "exclaim":  t.count("!"),
    }


def profile(sets, raws):
    out = {f: {"human": 0.0, "n": 0, **{k: [] for k in ORDER}} for f, _ in FEATURES}
    for sname, exs in sets.items():
        F = [(feats(e["chosen"]), feats(e["rejected"])) for e in exs]
        for f, _ in FEATURES:
            fc = np.array([a[f] for a, _ in F]); fr = np.array([b[f] for _, b in F])
            m = fc != fr
            if not m.sum():
                continue
            more_is_chosen = (fc > fr)[m]
            out[f]["n"] += int(m.sum())
            out[f]["human"] += float(more_is_chosen.mean()) * m.sum()
            for k in ORDER:
                per = []
                for raw in raws:
                    s_c = np.array(raw[sname][k]["chosen"]); s_r = np.array(raw[sname][k]["rejected"])
                    per.append(((s_c > s_r)[m] == more_is_chosen).mean())
                out[f][k].append(float(np.mean(per)) * m.sum())
    for f, _ in FEATURES:
        n = out[f]["n"]
        out[f]["human"] /= n
        for k in ORDER:
            out[f][k] = sum(out[f][k]) / n
    return out


def pick_examples(sets, raw, n=1):
    S = "UltraFeedback (ID)"; exs = sets[S]
    got = {k: (np.array(raw[S][k]["chosen"]) > np.array(raw[S][k]["rejected"])) for k in ORDER}
    lc = np.array([len(e["chosen"]) for e in exs]); lr = np.array([len(e["rejected"]) for e in exs])
    cand = np.where((lc < lr * 0.75) & (~got["explicit"]) & got["implicit"])[0]
    cand = sorted(cand, key=lambda i: len(exs[i]["prompt"]) + len(exs[i]["chosen"]))
    return [(exs[i], {k: bool(got[k][i]) for k in ORDER}, int(lc[i]), int(lr[i])) for i in cand[:n]]


def fig9_what_they_reward(tags):
    # datasets' legacy-cache probe pickles a module dict, which dill cannot do on
    # Python 3.14; it only decides where to look for an old cache, so skip it.
    import datasets.fingerprint as _fp
    _fp.Hasher.hash = classmethod(lambda cls, value: "v3")
    from src.eval import eval_sets
    sets = eval_sets(None)
    raws = [json.load(open(os.path.join(ROOT, "results", f"raw_{t}.json"))) for t in tags]
    prof = profile(sets, raws)
    examples = pick_examples(sets, raws[0], n=1)

    fig = plt.figure(figsize=(12.5, 7.2), facecolor=SURFACE)
    # the top panel needs a wide left gutter for its feature labels; the example box below
    # does not, so the two panels get their own extents rather than one shared gridspec.
    ax = fig.add_axes([0.205, 0.46, 0.775, 0.52]); ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    ax.grid(axis="x", color=GRID, linewidth=1, alpha=0.9); ax.set_axisbelow(True)
    yt, yl = [], []
    for i, (f, nice) in enumerate(FEATURES):
        base_y = -i * 1.0
        yt.append(base_y); yl.append(f"{nice}\n({prof[f]['n']:,} pairs)")
        for j, k in enumerate(ORDER):
            d = prof[f][k] - prof[f]["human"]
            y = base_y + (1 - j) * 0.26
            ax.plot([0, d], [y, y], color=SERIES[k], linewidth=2, solid_capstyle="round", zorder=3)
            ax.scatter([d], [y], s=62, color=SERIES[k], zorder=4,
                       edgecolors=SURFACE, linewidths=1.4)
            ax.annotate(f"{d:+.1%}", (d, y), textcoords="offset points",
                        xytext=(9 if d >= 0 else -9, 0), ha="left" if d >= 0 else "right",
                        va="center", fontsize=8, color=INK2, zorder=5)
        if i:
            ax.axhline(base_y + 0.5, color=GRID, linewidth=1, zorder=1)
    ax.axvline(0, color=INK, linewidth=2, zorder=2)
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=9, color=INK)
    ax.set_ylim(-len(FEATURES) + 0.5 - 0.15, 0.55)
    ax.set_xlim(-0.16, 0.24)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.0%}")
    ax.set_xlabel("How much more often than humans the scorer picks the side with this feature",
                  fontsize=9.5, color=INK2, labelpad=8)
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="none", markersize=8,
                              color=SERIES[k], label=LABEL[k]) for k in ORDER],
              loc="lower right", frameon=False, fontsize=9, labelcolor=INK2)

    ax2 = fig.add_axes([0.012, 0.02, 0.976, 0.36]); ax2.axis("off")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)

    def who(picks, side):
        names = [LABEL[k] for k in ORDER if picks[k] == side]
        return ", ".join(names) if names else "—"

    y = 0.98
    for ex, picks, nc, nr in examples:
        ax2.add_patch(FancyBboxPatch((0.002, y - 0.95), 0.996, 0.94,
                                     boxstyle="round,pad=0.004,rounding_size=0.012",
                                     facecolor="#f3f6f9", edgecolor=GRID, linewidth=1,
                                     transform=ax2.transAxes, zorder=0))
        ax2.text(0.014, y - 0.05, "PROMPT   " + textwrap.shorten(
                 ex["prompt"].replace("\n", " "), 130, placeholder=" …"),
                 fontsize=11, color=INK, fontweight="bold", va="top",
                 transform=ax2.transAxes)
        yy = y - 0.26
        for tag, text, n, side, col in [("✓ human preferred", ex["chosen"], nc, True, "#1baf7a"),
                                        ("✗ human rejected", ex["rejected"], nr, False, "#e34948")]:
            ax2.text(0.014, yy, f"{tag}   ({n} chars)   ·   picked by: {who(picks, side)}",
                     fontsize=10, color=col, fontweight="bold", va="top",
                     transform=ax2.transAxes)
            body = textwrap.fill(textwrap.shorten(text.replace("\n", " "), 230,
                                 placeholder=" …"), 118)
            ax2.text(0.028, yy - 0.13, body, fontsize=9.5, color=INK2, va="top",
                     family="monospace", linespacing=1.5, transform=ax2.transAxes)
            yy -= 0.36

    p = os.path.join(FIG, "fig9_what_they_reward.png")
    fig.savefig(p, dpi=200, facecolor=SURFACE); plt.close(fig)
    return p


if __name__ == "__main__":
    agg, seeds, tags = load()
    print(f"aggregating {len(tags)} seeds: {', '.join(tags)}")
    outs = [fig3(seeds, tags), fig6_value_added(agg, seeds, tags),
            fig10_convergence(), fig11_long_curves(), fig12_bias_drift()]
    if "--no-qualitative" not in sys.argv:
        outs.append(fig9_what_they_reward(tags))
    for p in outs:
        print("wrote", os.path.relpath(p, ROOT) if p else "(skipped)")
