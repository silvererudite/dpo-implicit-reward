"""
Stage 4 -- Evaluate scorers as preference classifiers on the benchmark test sets.

Loads the trained policy (+ frozen reference) and/or the explicit RM, scores every pair in
each evaluation set, and reports the metric suite (accuracy, ECE, length-controlled accuracy,
Spearman vs gold, and McNemar between implicit and explicit). Raw per-pair scores are dumped so
every metric is recomputable without re-running the models.

Evaluation sets (all benchmarks):
  - UltraFeedback test_prefs          -> in-distribution (ID)
  - RewardBench (per subset-category) -> distribution shift (OOD)
  - HH-RLHF harmless                  -> cross-dataset OOD

    python -m src.eval --config configs/eval.yaml
"""
from __future__ import annotations
import argparse, json, os, yaml
from collections import defaultdict
import numpy as np
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

from src.data import load_ultrafeedback, load_rewardbench, load_hh, rewardbench_category
from src.runtime import model_dtype
from src import scoring, metrics


def eval_sets(test_limit=None):
    sets = {"UltraFeedback (ID)": load_ultrafeedback("test_prefs", limit=test_limit)}
    rb = load_rewardbench(limit=None)
    by_cat = defaultdict(list)
    for e in rb:
        by_cat[rewardbench_category(e["subset"])].append(e)
    for cat, exs in by_cat.items():
        # test_limit caps every set, RewardBench included -- otherwise a "quick" eval still
        # scores all ~3k RewardBench pairs and there is no way to smoke-test the harness.
        sets[f"RewardBench:{cat} (OOD)"] = exs[:test_limit] if test_limit else exs
    sets["HH-harmless (OOD)"] = load_hh(subset="harmless-base", split="test", limit=test_limit)
    return sets


def score_set(examples, score_fn, device):
    sc, sr, lc, lr = [], [], [], []
    for e in examples:
        sc.append(score_fn(e["prompt"], e["chosen"]))
        sr.append(score_fn(e["prompt"], e["rejected"]))
        lc.append(len(e["chosen"])); lr.append(len(e["rejected"]))
    return map(np.asarray, (sc, sr, lc, lr))


def summarize(sc, sr, lc, lr):
    lca, n_lc = metrics.length_controlled_accuracy(sc, sr, lc, lr)
    return {
        "accuracy": round(metrics.pairwise_accuracy(sc, sr), 4),
        "ece": round(metrics.expected_calibration_error(sc, sr), 4),
        "len_controlled_acc": None if np.isnan(lca) else round(lca, 4),
        "len_controlled_n": n_lc,
    }


def main(cfg):
    device = cfg.get("device", "cuda")
    dtype = model_dtype(cfg)
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # pi_ref is the MERGED SFT checkpoint -- the same reference DPOTrainer used during training
    # (it derives pi_ref by disabling the DPO adapter on this backbone). Scoring against
    # base+SFT-adapter, or against raw base, would measure a different quantity than was trained.
    ref_dir = cfg["sft_merged_dir"]
    scorers = {}
    if cfg.get("dpo_dir"):
        ref = AutoModelForCausalLM.from_pretrained(
            ref_dir, torch_dtype=dtype).to(device).eval()
        policy_backbone = AutoModelForCausalLM.from_pretrained(
            ref_dir, torch_dtype=dtype).to(device)
        policy = PeftModel.from_pretrained(policy_backbone, cfg["dpo_dir"]).to(device).eval()
        beta = cfg.get("beta", 0.1)
        scorers["implicit"] = lambda p, r: scoring.implicit_reward(policy, ref, tok, p, r, beta, device)
        scorers["base_logprob"] = lambda p, r: scoring.base_logprob(ref, tok, p, r, device)
    if cfg.get("rm_dir"):
        rmb = AutoModelForSequenceClassification.from_pretrained(
            ref_dir, num_labels=1, torch_dtype=dtype)
        rmb.config.pad_token_id = tok.pad_token_id
        rm = PeftModel.from_pretrained(rmb, cfg["rm_dir"]).to(device).eval()
        scorers["explicit"] = lambda p, r: scoring.explicit_reward(rm, tok, p, r, device)

    sets = eval_sets(cfg.get("test_limit"))
    report, raw = {}, {}
    for set_name, exs in sets.items():
        report[set_name], raw[set_name] = {}, {}
        for scorer_name, fn in scorers.items():
            sc, sr, lc, lr = score_set(exs, fn, device)
            report[set_name][scorer_name] = summarize(sc, sr, lc, lr)
            raw[set_name][scorer_name] = {"chosen": sc.tolist(), "rejected": sr.tolist()}
        # paired significance implicit vs explicit
        if {"implicit", "explicit"} <= set(scorers):
            ci = np.asarray(raw[set_name]["implicit"]["chosen"]) > np.asarray(raw[set_name]["implicit"]["rejected"])
            ce = np.asarray(raw[set_name]["explicit"]["chosen"]) > np.asarray(raw[set_name]["explicit"]["rejected"])
            report[set_name]["_mcnemar_p_impl_vs_expl"] = round(metrics.mcnemar_pvalue(ci, ce), 5)

    os.makedirs("results", exist_ok=True)
    tag = cfg.get("tag", "run")
    json.dump(report, open(f"results/eval_{tag}.json", "w"), indent=2)
    json.dump(raw, open(f"results/raw_{tag}.json", "w"))
    print(json.dumps(report, indent=2))
    print(f"\nSaved -> results/eval_{tag}.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    main(yaml.safe_load(open(a.config)))
