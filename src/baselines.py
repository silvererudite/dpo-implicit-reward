"""
Non-neural reference baselines for preference-pair classification.

These require NO GPU and no neural forward pass -- they are the *reference floors*
that the neural conditions (DPO implicit reward, explicit Bradley-Terry RM,
base-model log-prob) must beat. Reported in the preliminary update as real numbers.

Baselines:
  - random        : coin flip (chance = 0.5)
  - pick_longer   : predict the longer response is preferred (length confound probe)
  - pick_shorter  : predict the shorter response is preferred
  - tfidf_lr      : TF-IDF features + logistic regression on (prompt, response) pairs,
                    trained on UltraFeedback train, evaluated on each test set.

Usage:
    python -m src.baselines            # runs the full suite, writes results/baselines.json
"""
from __future__ import annotations
import json
import os
from collections import defaultdict
from typing import List, Dict, Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data import (
    load_ultrafeedback, load_rewardbench, load_hh, rewardbench_category,
)

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
SEED = 0


# ----------------------------- deterministic baselines ----------------------------- #
def acc_random(examples: List[Dict]) -> float:
    """Expected accuracy of a coin flip is 0.5 regardless of data."""
    return 0.5


def acc_length(examples: List[Dict], prefer: str = "longer") -> float:
    correct = 0
    for e in examples:
        lc, lr = len(e["chosen"]), len(e["rejected"])
        if lc == lr:
            correct += 0.5
        elif (lc > lr) == (prefer == "longer"):
            correct += 1
    return correct / max(1, len(examples))


# ----------------------------- learned TF-IDF baseline ----------------------------- #
def _pair_texts(e: Dict) -> (str, str):
    """Represent each side as prompt + response so the classifier sees context."""
    return (e["prompt"] + " [SEP] " + e["chosen"], e["prompt"] + " [SEP] " + e["rejected"])


def train_tfidf_lr(train: List[Dict]):
    """
    Build a *pointwise* preference scorer: fit TF-IDF on all responses, then train
    logistic regression to score a (prompt,response) as chosen(1)/rejected(0).
    At eval time we score both sides and pick the higher-scoring one.
    """
    corpus, labels = [], []
    for e in train:
        c, r = _pair_texts(e)
        corpus += [c, r]
        labels += [1, 0]
    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X = vec.fit_transform(corpus)
    # liblinear is numerically robust for high-dim sparse features (avoids lbfgs overflow
    # seen on older numpy/BLAS stacks); L2-regularized logistic regression.
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="liblinear", random_state=SEED)
    clf.fit(X, labels)
    return vec, clf


def acc_tfidf_lr(examples: List[Dict], vec, clf) -> float:
    c_texts = [e["prompt"] + " [SEP] " + e["chosen"] for e in examples]
    r_texts = [e["prompt"] + " [SEP] " + e["rejected"] for e in examples]
    sc = clf.decision_function(vec.transform(c_texts))
    sr = clf.decision_function(vec.transform(r_texts))
    return float(np.mean(sc > sr))


# ----------------------------- eval driver ----------------------------- #
def evaluate_suite(test_sets: Dict[str, List[Dict]], vec, clf) -> Dict[str, Dict[str, float]]:
    rows = {}
    for name, exs in test_sets.items():
        rows[name] = {
            "n": len(exs),
            "random": round(acc_random(exs), 4),
            "pick_longer": round(acc_length(exs, "longer"), 4),
            "pick_shorter": round(acc_length(exs, "shorter"), 4),
            "tfidf_lr": round(acc_tfidf_lr(exs, vec, clf), 4),
        }
    return rows


def rewardbench_by_category(rb: List[Dict]) -> Dict[str, List[Dict]]:
    buckets = defaultdict(list)
    for e in rb:
        buckets[rewardbench_category(e["subset"])].append(e)
    return dict(buckets)


def main(train_limit: int = 20000, test_limit: int = 3000):
    print("Loading benchmark datasets ...")
    uf_train = load_ultrafeedback("train_prefs", limit=train_limit)
    uf_test = load_ultrafeedback("test_prefs", limit=test_limit)
    rb = load_rewardbench(limit=None)
    hh = load_hh(subset="harmless-base", split="test", limit=test_limit)

    print(f"  UF train={len(uf_train)}  UF test={len(uf_test)}  RewardBench={len(rb)}  HH-harmless={len(hh)}")

    print("Training TF-IDF + logistic regression on UltraFeedback train ...")
    vec, clf = train_tfidf_lr(uf_train)

    test_sets = {"UltraFeedback (ID)": uf_test, "HH-RLHF harmless (OOD)": hh}
    for cat, exs in rewardbench_by_category(rb).items():
        test_sets[f"RewardBench: {cat} (OOD)"] = exs
    test_sets["RewardBench: ALL (OOD)"] = rb

    rows = evaluate_suite(test_sets, vec, clf)

    # pretty print
    print(f"\n{'test set':32s} {'n':>5s} {'rand':>6s} {'long':>6s} {'short':>6s} {'tfidf':>6s}")
    print("-" * 70)
    for name, r in rows.items():
        print(f"{name:32s} {r['n']:5d} {r['random']:6.3f} {r['pick_longer']:6.3f} "
              f"{r['pick_shorter']:6.3f} {r['tfidf_lr']:6.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {"seed": SEED, "train_size": len(uf_train), "results": rows}
    with open(os.path.join(RESULTS_DIR, "baselines.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {os.path.join(RESULTS_DIR, 'baselines.json')}")
    return out


if __name__ == "__main__":
    main()
