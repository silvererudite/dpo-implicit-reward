"""
Does the reward model's length bias GROW as it trains?

The surface-form analysis (fig9) showed the explicit reward model over-prefers longer, more
formatted answers by 10-16 points relative to the human labels, while DPO's implicit reward does
not. The convergence run showed the same model peaking at ~epoch 2 and then overfitting.

If the bias grows monotonically with training, one mechanism explains three separate
observations: the in-distribution gains, the collapse on RewardBench Chat-Hard (which inverts
the length cue), and the shape of the overfitting curve. This scores the checkpoints saved every
250 steps to find out.

    python -m src.bias_over_training --scorer explicit --seed 0 --n 500
"""
from __future__ import annotations
import argparse, json, os, re
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

from src.data import load_ultrafeedback
from src.runtime import model_dtype
from src import scoring

FEATURES = ("length", "markdown", "digits")


def feats(t):
    return {"length": len(t),
            "markdown": len(re.findall(r"(?m)^\s*(?:[-*•]|\d+\.)\s", t)) + t.count("**") + t.count("##"),
            "digits": sum(c.isdigit() for c in t)}


def rates(picked_chosen, F):
    """For each feature: how often the scorer picked the side with MORE of it (pairs where it differs)."""
    out = {}
    for f in FEATURES:
        fc = np.array([a[f] for a, _ in F]); fr = np.array([b[f] for _, b in F])
        m = fc != fr
        if not m.sum():
            out[f] = None; continue
        out[f] = {"scorer": float((picked_chosen[m] == (fc > fr)[m]).mean()),
                  "human": float((fc > fr)[m].mean()), "n": int(m.sum())}
    return out


def main(a):
    dev = "cuda"
    cfg = {"bf16": True}
    dtype = model_dtype(cfg)
    base = "Qwen/Qwen2.5-0.5B"; ref_dir = "outputs/sft_merged"
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ex = load_ultrafeedback("test_prefs", limit=a.n)
    F = [(feats(e["chosen"]), feats(e["rejected"])) for e in ex]

    root = (f"outputs/long/rm_8k_s{a.seed}" if a.scorer == "explicit"
            else f"outputs/long/dpo_beta0.1_8k_s{a.seed}")
    ckpts = sorted(
        [d for d in os.listdir(root) if d.startswith("checkpoint-")],
        key=lambda d: int(d.split("-")[1]))

    ref = None
    if a.scorer == "implicit":
        ref = AutoModelForCausalLM.from_pretrained(ref_dir, torch_dtype=dtype).to(dev).eval()

    results = []
    for c in ckpts:
        step = int(c.split("-")[1])
        path = os.path.join(root, c)
        if a.scorer == "explicit":
            m = AutoModelForSequenceClassification.from_pretrained(
                ref_dir, num_labels=1, torch_dtype=dtype)
            m.config.pad_token_id = tok.pad_token_id
            model = PeftModel.from_pretrained(m, path).to(dev).eval()
            fn = lambda p, r: scoring.explicit_reward(model, tok, p, r, dev)
        else:
            bb = AutoModelForCausalLM.from_pretrained(ref_dir, torch_dtype=dtype).to(dev)
            model = PeftModel.from_pretrained(bb, path).to(dev).eval()
            fn = lambda p, r: scoring.implicit_reward(model, ref, tok, p, r, 0.1, dev)

        sc = np.array([fn(e["prompt"], e["chosen"]) for e in ex])
        sr = np.array([fn(e["prompt"], e["rejected"]) for e in ex])
        picked = sc > sr
        rec = {"step": step, "epoch": round(step / 500, 2), "accuracy": float(picked.mean()),
               "rates": rates(picked, F)}
        results.append(rec)
        ln = rec["rates"]["length"]
        print(f"  step {step:5d} (ep {rec['epoch']:.1f})  acc {rec['accuracy']:.3f}  "
              f"picks-longer {ln['scorer']:.3f} (human {ln['human']:.3f}, "
              f"bias {ln['scorer']-ln['human']:+.3f})", flush=True)
        del model
        torch.cuda.empty_cache()

    os.makedirs("results", exist_ok=True)
    out = f"results/bias_over_training_{a.scorer}_s{a.seed}.json"
    json.dump({"scorer": a.scorer, "seed": a.seed, "n_pairs": len(ex),
               "checkpoints": results}, open(out, "w"), indent=2)
    print("Saved ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scorer", choices=["explicit", "implicit"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    main(a)
