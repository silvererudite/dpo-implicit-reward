"""
Convert the unified benchmark examples (src/data.py) into HuggingFace Datasets in the
formats the TRL trainers expect, and build the matched budget subsets {2k, 8k, 32k}.

  - SFT  : {"messages"} chat format on the *chosen* response
  - DPO  : {"prompt", "chosen", "rejected"} (conversational)
  - RM   : {"chosen", "rejected"} (conversational; RewardTrainer tokenizes both sides)
"""
from __future__ import annotations
from typing import List, Dict, Optional
from datasets import Dataset

from src.data import load_ultrafeedback

BUDGETS = {"2k": 2000, "8k": 8000, "32k": 32000}


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


def build_budget_subsets(split: str = "train_prefs") -> Dict[str, List[Dict]]:
    """Nested subsets (32k superset of 8k superset of 2k) so scale is the only variable."""
    full = load_ultrafeedback(split, limit=BUDGETS["32k"])
    return {name: full[:n] for name, n in BUDGETS.items()}


if __name__ == "__main__":
    subs = build_budget_subsets()
    for k, v in subs.items():
        print(k, len(v))
    print("DPO example:", to_dpo(subs["2k"][:1])[0])
    print("RM  example keys:", list(to_rm(subs["2k"][:1])[0].keys()))
