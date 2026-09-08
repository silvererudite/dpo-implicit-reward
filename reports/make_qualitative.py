"""
What does each scorer actually reward?

Accuracy says which scorer wins; it does not say what any of them is rewarding. This figure
characterises each scorer by SURFACE FORM: for every pair where a textual feature differs
between the two responses, how often does the scorer pick the side with more of it?

The reference is the human rate, not 50%. A scorer that prefers longer answers is not
biased if humans prefer longer answers too -- bias is the DEVIATION from what humans did on
the same pairs, which is what the top panel plots. The bottom panel grounds this in real
examples, because a 15-point rate difference is abstract until you see the responses.

    python reports/make_qualitative.py
"""
import json, os, re, sys, textwrap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # import src/
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from src.eval import eval_sets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "reports", "figures")
SURFACE = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8a8a85"; GRID = "#e4e4e0"
SERIES = {"implicit": "#2a78d6", "explicit": "#eb6834", "base_logprob": "#1baf7a"}
LABEL = {"implicit": "DPO implicit", "explicit": "Explicit RM", "base_logprob": "Untrained baseline"}
ORDER = ["implicit", "explicit", "base_logprob"]
TAGS = [f"s{i}" for i in range(5)]

FEATURES = [
    ("length",   "longer response"),
    ("markdown", "bullet points / bold"),
    ("code",     "code blocks"),
    ("digits",   "numbers"),
    ("hedge",    "hedging & disclaimers"),
    ("exclaim",  "exclamation marks"),
]


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
    """Rate at which each scorer (and the human labels) pick the higher-feature side."""
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
                for raw in raws:                       # average the rate over seeds
                    s_c = np.array(raw[sname][k]["chosen"]); s_r = np.array(raw[sname][k]["rejected"])
                    per.append(((s_c > s_r)[m] == more_is_chosen).mean())
                out[f][k].append(float(np.mean(per)) * m.sum())
    for f, _ in FEATURES:
        n = out[f]["n"]
        out[f]["human"] /= n
        for k in ORDER:
            out[f][k] = sum(out[f][k]) / n
    return out


def pick_examples(sets, raw, n=3):
    """Cases where humans preferred the SHORTER answer and the scorers split on it."""
    S = "UltraFeedback (ID)"; exs = sets[S]
    got = {k: (np.array(raw[S][k]["chosen"]) > np.array(raw[S][k]["rejected"])) for k in ORDER}
    lc = np.array([len(e["chosen"]) for e in exs]); lr = np.array([len(e["rejected"]) for e in exs])
    cand = np.where((lc < lr * 0.75) & (~got["explicit"]) & got["implicit"])[0]
    cand = sorted(cand, key=lambda i: len(exs[i]["prompt"]) + len(exs[i]["chosen"]))
    return [(exs[i], {k: bool(got[k][i]) for k in ORDER}, int(lc[i]), int(lr[i])) for i in cand[:n]]


def main():
    sets = eval_sets(None)
    raws = [json.load(open(os.path.join(ROOT, "results", f"raw_{t}.json"))) for t in TAGS]
    prof = profile(sets, raws)
    examples = pick_examples(sets, raws[0])

    fig = plt.figure(figsize=(14, 11.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.25], hspace=0.30,
                          left=0.19, right=0.975, top=0.895, bottom=0.035)

    # ---------------- top: bias relative to the human rate ----------------
    ax = fig.add_subplot(gs[0]); ax.set_facecolor(SURFACE)
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
    ax.set_title("What is each scorer actually rewarding?", fontsize=14, color=INK,
                 pad=62, loc="left", fontweight="bold")
    ax.annotate("Zero = agrees with the human labels on the same pairs. Right of zero = over-prefers "
                "that feature.\nThe explicit reward model and the untrained baseline both chase surface "
                "form; the implicit reward does not.",
                xy=(0, 1.015), xycoords="axes fraction", fontsize=9.5, color=INK2, va="bottom")
    ax.legend(handles=[Line2D([], [], marker="o", linestyle="none", markersize=8,
                              color=SERIES[k], label=LABEL[k]) for k in ORDER],
              loc="lower right", frameon=False, fontsize=9, labelcolor=INK2)

    # ---------------- bottom: worked examples ----------------
    ax2 = fig.add_subplot(gs[1]); ax2.axis("off")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.set_title("What that looks like in practice", fontsize=12.5, color=INK,
                  pad=34, loc="left", fontweight="bold")
    ax2.annotate("Real pairs from the held-out set. In each, humans preferred the SHORT direct answer "
                 "and rejected the padded one.",
                 xy=(0, 1.005), xycoords="axes fraction", fontsize=9.5, color=INK2, va="bottom")

    def who(picks, side):
        names = [LABEL[k] for k in ORDER if picks[k] == side]
        return ", ".join(names) if names else "—"

    y = 0.97
    for ex, picks, nc, nr in examples:
        ax2.add_patch(FancyBboxPatch((0.002, y - 0.305), 0.996, 0.296,
                                     boxstyle="round,pad=0.004,rounding_size=0.012",
                                     facecolor="#f3f6f9", edgecolor=GRID, linewidth=1,
                                     transform=ax2.transAxes, zorder=0))
        ax2.text(0.018, y - 0.022, "PROMPT   " + textwrap.shorten(
                 ex["prompt"].replace("\n", " "), 115, placeholder=" …"),
                 fontsize=9.5, color=INK, fontweight="bold", va="top",
                 transform=ax2.transAxes)
        yy = y - 0.082
        for tag, text, n, side, col in [("✓ human preferred", ex["chosen"], nc, True, "#1baf7a"),
                                        ("✗ human rejected", ex["rejected"], nr, False, "#e34948")]:
            ax2.text(0.018, yy, f"{tag}   ({n} chars)   ·   picked by: {who(picks, side)}",
                     fontsize=8.5, color=col, fontweight="bold", va="top",
                     transform=ax2.transAxes)
            body = textwrap.fill(textwrap.shorten(text.replace("\n", " "), 205,
                                 placeholder=" …"), 112)
            ax2.text(0.034, yy - 0.036, body, fontsize=8, color=INK2, va="top",
                     family="monospace", linespacing=1.5, transform=ax2.transAxes)
            yy -= 0.115
        y -= 0.335

    p = os.path.join(FIG, "fig9_what_they_reward.png")
    fig.savefig(p, dpi=200, facecolor=SURFACE); plt.close(fig)
    print("wrote", p)


if __name__ == "__main__":
    main()
