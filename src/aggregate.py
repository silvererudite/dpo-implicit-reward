"""
Aggregate per-seed evaluation runs into mean +/- 95% CI.

A single seed cannot separate a real implicit-vs-explicit difference from training noise,
so TASKS.md requires >=3 seeds for any headline claim. This reads results/eval_<tag>.json for
each seed and reports, per (test set, scorer), the mean and a 95% confidence interval.

The PRIMARY metric here is length-controlled accuracy, not raw accuracy. Raw accuracy on these
benchmarks is substantially a length signal: the explicit RM scored 0.407 on RewardBench
Chat-Hard (worse than chance) purely by preferring longer responses on a set built to invert
that cue, and recovered to 0.589 once pairs were length-matched. Raw accuracy is still reported
alongside, and so is the length-controlled sample size -- some subsets retain very few matched
pairs, and a mean over 32 pairs deserves to be read with that in view.

    python -m src.aggregate --tags s0,s1,s2 --out results/aggregate_8k.json
"""
from __future__ import annotations
import argparse, json, os
from statistics import mean, stdev

SCORERS = ("implicit", "explicit", "base_logprob")
METRICS = ("len_controlled_acc", "accuracy", "ece")


def ci95(vals):
    """95% CI half-width using the t multiplier for small n (2.776 at n=3, 3.182 at n=4)."""
    n = len(vals)
    if n < 2:
        return None
    t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 2.0)
    return t * (stdev(vals) / (n ** 0.5))


def main(tags, out_path):
    runs = []
    for t in tags:
        p = f"results/eval_{t}.json"
        if not os.path.exists(p):
            print(f"[warn] missing {p} -- skipping")
            continue
        runs.append(json.load(open(p)))
    if not runs:
        raise SystemExit("no per-seed result files found")
    print(f"aggregating {len(runs)} seed(s): {', '.join(tags)}\n")

    sets = list(runs[0].keys())
    agg = {}
    for s in sets:
        agg[s] = {}
        for sc in SCORERS:
            vals = {m: [r[s][sc][m] for r in runs
                        if sc in r.get(s, {}) and r[s][sc].get(m) is not None]
                    for m in METRICS}
            if not vals["accuracy"]:
                continue
            agg[s][sc] = {}
            for m in METRICS:
                v = vals[m]
                if not v:
                    agg[s][sc][m] = None
                    continue
                agg[s][sc][m] = {"mean": round(mean(v), 4),
                                 "ci95": (round(ci95(v), 4) if ci95(v) is not None else None),
                                 "n_seeds": len(v)}
            lcn = [r[s][sc].get("len_controlled_n") for r in runs if sc in r.get(s, {})]
            agg[s][sc]["len_controlled_n"] = lcn[0] if lcn else None
        ps = [r[s]["_mcnemar_p_impl_vs_expl"] for r in runs if "_mcnemar_p_impl_vs_expl" in r.get(s, {})]
        if ps:
            agg[s]["_mcnemar_p_impl_vs_expl"] = {"mean": round(mean(ps), 5), "per_seed": ps}

    # Console table: primary metric first, with CIs, so overlap is visible at a glance.
    def fmt(d):
        if not d or d.get("mean") is None:
            return "     -    "
        c = d.get("ci95")
        return f"{d['mean']:.3f}±{c:.3f}" if c is not None else f"{d['mean']:.3f}      "

    print(f"{'test set':30s} {'scorer':13s} {'LC-acc (primary)':>18s} {'raw acc':>16s} {'LC-n':>6s}")
    for s in sets:
        for sc in SCORERS:
            if sc not in agg[s]:
                continue
            a = agg[s][sc]
            print(f"{s[:29]:30s} {sc:13s} {fmt(a['len_controlled_acc']):>18s} "
                  f"{fmt(a['accuracy']):>16s} {str(a['len_controlled_n']):>6s}")
        print()

    os.makedirs("results", exist_ok=True)
    json.dump({"tags": tags, "n_seeds": len(runs), "results": agg}, open(out_path, "w"), indent=2)
    print("Saved ->", out_path)

    if os.environ.get("WANDB_DISABLED", "").lower() not in ("1", "true"):
        _log_wandb(tags, agg, out_path)


def _log_wandb(tags, agg, out_path):
    """Log the cross-seed summary as its own W&B run.

    The per-seed training runs already stream loss curves, but the headline number is the
    seed-aggregated accuracy with its CI -- that lives nowhere until it is logged here.
    """
    try:
        import wandb
    except ImportError:
        return
    try:
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "dpo-implicit-reward"),
                         name=f"aggregate-{'-'.join(tags)}", job_type="aggregate",
                         config={"seeds": tags, "n_seeds": len(tags)}, reinit=True)
        cols = ["test_set", "scorer", "lc_acc_mean", "lc_acc_ci95", "acc_mean", "acc_ci95",
                "ece_mean", "lc_n", "n_seeds"]
        tbl = wandb.Table(columns=cols)
        for s, per in agg.items():
            for sc in SCORERS:
                a = per.get(sc)
                if not a:
                    continue
                g = lambda m, k: (a[m] or {}).get(k) if a.get(m) else None
                tbl.add_data(s, sc, g("len_controlled_acc", "mean"), g("len_controlled_acc", "ci95"),
                             g("accuracy", "mean"), g("accuracy", "ci95"), g("ece", "mean"),
                             a.get("len_controlled_n"), g("accuracy", "n_seeds"))
                # scalar summaries so runs are comparable in the W&B UI
                for m, short in (("len_controlled_acc", "lc_acc"), ("accuracy", "acc")):
                    v = g(m, "mean")
                    if v is not None:
                        run.summary[f"{s}/{sc}/{short}"] = v
        run.log({"results": tbl})
        run.finish()
        print(f"[wandb] logged aggregate -> {run.url}")
    except Exception as exc:
        print(f"[wandb] skipped ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="comma-separated eval tags, one per seed")
    ap.add_argument("--out", default="results/aggregate.json")
    a = ap.parse_args()
    main([t.strip() for t in a.tags.split(",")], a.out)
