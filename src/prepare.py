"""
Convert the unified benchmark examples (src/data.py) into HuggingFace Datasets in the
formats the TRL trainers expect, and build the matched budget subsets {2k, 8k, 32k}.

  - SFT  : {"messages"} chat format on the *chosen* response
  - DPO  : {"prompt", "chosen", "rejected"} (conversational)
  - RM   : {"chosen", "rejected"} (conversational; RewardTrainer tokenizes both sides)
"""
from __future__ import annotations
import hashlib, json, os
from typing import List, Dict, Optional
from datasets import Dataset

from src.data import load_ultrafeedback

BUDGETS = {"2k": 2000, "8k": 8000, "32k": 32000}
CACHE_DIR = "outputs/data_cache"


def _conv(prompt, response):
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]


def to_dpo(examples: List[Dict]) -> Dataset:
    return Dataset.from_list([
        {"prompt": [{"role": "user", "content": e["prompt"]}],
         "chosen": [{"role": "assistant", "content": e["chosen"]}],
         "rejected": [{"role": "assistant", "content": e["rejected"]}]}
        for e in examples
    ])


def to_rm(examples: List[Dict]) -> Dataset:
    return Dataset.from_list([
        {"chosen": _conv(e["prompt"], e["chosen"]),
         "rejected": _conv(e["prompt"], e["rejected"])}
        for e in examples
    ])


def to_sft(examples: List[Dict]) -> Dataset:
    return Dataset.from_list([{"messages": _conv(e["prompt"], e["chosen"])} for e in examples])


def _fits(tok, e, max_length: int, max_prompt_length: int) -> bool:
    """True if the prompt and BOTH responses fit the trainers' length limits under the chat template."""
    prompt_ids = tok.apply_chat_template(
        [{"role": "user", "content": e["prompt"]}], add_generation_prompt=True, tokenize=True)
    if len(prompt_ids) > max_prompt_length:
        return False
    for side in ("chosen", "rejected"):
        full = tok.apply_chat_template(
            [{"role": "user", "content": e["prompt"]},
             {"role": "assistant", "content": e[side]}], tokenize=True)
        if len(full) > max_length:
            return False
    return True


def length_eligible(split: str, tok, max_length: int, max_prompt_length: int,
                    pool: Optional[int] = None) -> List[Dict]:
    """Pairs short enough that NEITHER trainer alters them. Cached to disk (regenerable).

    Why this exists: TRL's RewardTrainer *drops* examples over max_length while DPOTrainer
    *truncates* them, so at the same nominal budget the two conditions silently saw different
    data (7,520 vs 8,000 pairs at 8k). Filtering to a common length-eligible pool first means
    both see byte-identical pairs, restoring the "same pairs" half of the DESIGN.md compute match.
    """
    key = hashlib.md5(f"{split}|{max_length}|{max_prompt_length}|{pool}|"
                      f"{getattr(tok, 'name_or_path', '')}".encode()).hexdigest()[:12]
    path = os.path.join(CACHE_DIR, f"eligible_{split}_{key}.json")
    if os.path.exists(path):
        return json.load(open(path))
    full = load_ultrafeedback(split, limit=pool)
    keep = [e for e in full if _fits(tok, e, max_length, max_prompt_length)]
    os.makedirs(CACHE_DIR, exist_ok=True)
    json.dump(keep, open(path, "w"))
    print(f"[data] {split}: {len(keep)}/{len(full)} pairs length-eligible "
          f"(max_length={max_length}, max_prompt_length={max_prompt_length}) -> {path}")
    return keep


def heldout_pairs(tok, max_length: int, max_prompt_length: int, n: int = 400) -> List[Dict]:
    """A held-out slice for measuring generalization *during* training.

    Drawn from UltraFeedback test_prefs and passed to BOTH trainers, so the DPO and BT
    convergence curves are measured on identical pairs. Without this there is no way to
    see where each method peaks -- training loss keeps falling long after held-out
    accuracy stops improving, so "train longer" would otherwise be flying blind.
    """
    return length_eligible("test_prefs", tok, max_length, max_prompt_length, pool=None)[:n]


def build_budget_subsets(split: str = "train_prefs", tok=None, max_length: Optional[int] = None,
                         max_prompt_length: Optional[int] = None) -> Dict[str, List[Dict]]:
    """Nested subsets (32k superset of 8k superset of 2k) so scale is the only variable.

    Pass tok + the length limits to draw from the length-eligible pool instead of raw order;
    required for DPO and the RM to train on identical pairs.
    """
    if tok is not None and max_length:
        full = length_eligible(split, tok, max_length, max_prompt_length or max_length, pool=None)
    else:
        full = load_ultrafeedback(split, limit=BUDGETS["32k"])
    out = {}
    for name, n in BUDGETS.items():
        if len(full) < n:
            print(f"[data] WARNING: only {len(full)} eligible pairs, short of the {name} budget ({n})")
        out[name] = full[:n]
    return out


if __name__ == "__main__":
    subs = build_budget_subsets()
    for k, v in subs.items():
        print(k, len(v))
    print("DPO example:", to_dpo(subs["2k"][:1])[0])
    print("RM  example keys:", list(to_rm(subs["2k"][:1])[0].keys()))
