"""
Evaluation metrics for scoring preference pairs (scorer-agnostic).

Given, for a set of pairs, the scorer's value on the chosen and rejected responses:
  - pairwise_accuracy      : fraction where score(chosen) > score(rejected)   [PRIMARY]
  - expected_calibration_error : ECE of implied choice probs sigma(delta)
  - length_controlled_accuracy : accuracy restricted to length-matched pairs  (length confound)
  - spearman_vs_gold       : rank correlation of scorer values against gold ratings
  - mcnemar_pvalue         : paired significance between two scorers on the same pairs
"""
from __future__ import annotations
from typing import List, Sequence
import numpy as np


def pairwise_accuracy(s_chosen: Sequence[float], s_rejected: Sequence[float]) -> float:
    sc, sr = np.asarray(s_chosen), np.asarray(s_rejected)
    wins = (sc > sr).astype(float) + 0.5 * (sc == sr).astype(float)
    return float(wins.mean())


def implied_choice_prob(s_chosen, s_rejected) -> np.ndarray:
    """P(chosen preferred) = sigma(score_chosen - score_rejected)."""
    delta = np.asarray(s_chosen) - np.asarray(s_rejected)
    return 1.0 / (1.0 + np.exp(-delta))


def expected_calibration_error(s_chosen, s_rejected, n_bins: int = 10) -> float:
    """
    ECE over the implied prob that the chosen side wins. Ground truth label is always 1
    (chosen is by construction the preferred response), so accuracy in a bin is the
    fraction of pairs the scorer actually got right among those with that confidence.
    """
    p = implied_choice_prob(s_chosen, s_rejected)          # confidence chosen wins
    correct = (np.asarray(s_chosen) > np.asarray(s_rejected)).astype(float)
    # fold: confidence in the *predicted* winner is max(p, 1-p)
    conf = np.maximum(p, 1 - p)
    pred_correct = np.where(p >= 0.5, correct, 1 - correct)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.mean()) * abs(pred_correct[m].mean() - conf[m].mean())
    return float(ece)


def length_controlled_accuracy(s_chosen, s_rejected, len_chosen, len_rejected,
                               max_ratio: float = 1.2):
    """Accuracy on pairs whose two responses are within `max_ratio` in length (both directions)."""
    lc, lr = np.asarray(len_chosen), np.asarray(len_rejected)
    ratio = np.maximum(lc, lr) / np.maximum(1, np.minimum(lc, lr))
    keep = ratio <= max_ratio
    if keep.sum() == 0:
        return float("nan"), 0
    acc = pairwise_accuracy(np.asarray(s_chosen)[keep], np.asarray(s_rejected)[keep])
    return acc, int(keep.sum())


def spearman_vs_gold(scores: Sequence[float], gold: Sequence[float]) -> float:
    from scipy.stats import spearmanr
    rho, _ = spearmanr(scores, gold)
    return float(rho)


def mcnemar_pvalue(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> float:
    """Paired test that two scorers have equal accuracy on the same pairs (exact binomial)."""
    from scipy.stats import binomtest
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    b01 = int(((a) & (~b)).sum())     # A right, B wrong
    b10 = int(((~a) & (b)).sum())     # A wrong, B right
    n = b01 + b10
    if n == 0:
        return 1.0
    return float(binomtest(min(b01, b10), n, 0.5, alternative="two-sided").pvalue)
