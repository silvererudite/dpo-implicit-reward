"""
Unified loaders for the benchmark preference datasets used in this project.

All loaders return a list of dicts with a common schema:
    {"prompt": str, "chosen": str, "rejected": str, "subset": str}
where `chosen`/`rejected` are the *response* strings (prompt stripped) and
`subset` tags the source split/category (used for per-category accuracy).

Datasets (all standard benchmarks, per instructor feedback to use benchmark data):
  - UltraFeedback-binarized  (HuggingFaceH4/ultrafeedback_binarized) : training + in-distribution test
  - RewardBench              (allenai/reward-bench)                  : reward-model eval benchmark (OOD via subsets)
  - Anthropic HH-RLHF        (Anthropic/hh-rlhf)                     : classic preference benchmark (cross-dataset OOD)
"""
from __future__ import annotations
import re
from typing import List, Dict, Optional
from datasets import load_dataset

Example = Dict[str, str]


def _last_assistant_turn(text: str) -> str:
    """Extract the final assistant response from an HH-RLHF transcript string."""
    parts = re.split(r"\n\nAssistant:", text)
    return parts[-1].strip() if len(parts) > 1 else text.strip()


def _hh_prompt(text: str) -> str:
    """Everything up to (and including) the last Human turn in an HH transcript."""
    idx = text.rfind("\n\nAssistant:")
    return text[:idx].strip() if idx != -1 else text.strip()


def load_ultrafeedback(split: str = "train_prefs", limit: Optional[int] = None) -> List[Example]:
    """UltraFeedback-binarized. `chosen`/`rejected` are message lists; take the last assistant content."""
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    out = []
    for ex in ds:
        chosen_resp = ex["chosen"][-1]["content"].strip()
        rejected_resp = ex["rejected"][-1]["content"].strip()
        if not chosen_resp or not rejected_resp:
            continue
        out.append({
            "prompt": ex["prompt"].strip(),
            "chosen": chosen_resp,
            "rejected": rejected_resp,
            "subset": "ultrafeedback",
        })
    return out


def load_rewardbench(split: str = "filtered", limit: Optional[int] = None) -> List[Example]:
    """RewardBench: flat prompt/chosen/rejected strings + `subset` category (natural distribution shift)."""
    ds = load_dataset("allenai/reward-bench", split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    out = []
    for ex in ds:
        out.append({
            "prompt": ex["prompt"].strip(),
            "chosen": ex["chosen"].strip(),
            "rejected": ex["rejected"].strip(),
            "subset": ex.get("subset", "rewardbench"),
        })
    return out


def load_hh(split: str = "test", subset: str = "harmless-base", limit: Optional[int] = None) -> List[Example]:
    """Anthropic HH-RLHF transcripts; parse out the shared prompt and final assistant responses."""
    ds = load_dataset("Anthropic/hh-rlhf", data_dir=subset, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    out = []
    for ex in ds:
        out.append({
            "prompt": _hh_prompt(ex["chosen"]),
            "chosen": _last_assistant_turn(ex["chosen"]),
            "rejected": _last_assistant_turn(ex["rejected"]),
            "subset": f"hh-{subset}",
        })
    return out


# Category groupings for RewardBench -> coarse buckets for reporting.
REWARDBENCH_CATEGORIES = {
    "Chat": ["alpacaeval-easy", "alpacaeval-length", "alpacaeval-hard", "mt-bench-easy", "mt-bench-med"],
    "Chat-Hard": ["mt-bench-hard", "llmbar-natural", "llmbar-adver-neighbor",
                   "llmbar-adver-GPTInst", "llmbar-adver-GPTOut", "llmbar-adver-manual"],
    "Safety": ["refusals-dangerous", "refusals-offensive", "xstest-should-refuse",
                "xstest-should-respond", "donotanswer"],
    "Reasoning": ["math-prm", "hep-cpp", "hep-go", "hep-java", "hep-js", "hep-python", "hep-rust"],
}


def rewardbench_category(subset: str) -> str:
    for cat, members in REWARDBENCH_CATEGORIES.items():
        if subset in members:
            return cat
    return "Other"


if __name__ == "__main__":
    # Smoke check: print a couple examples per dataset.
    for name, fn in [
        ("UltraFeedback[test_prefs]", lambda: load_ultrafeedback("test_prefs", limit=2)),
        ("RewardBench[filtered]", lambda: load_rewardbench(limit=2)),
        ("HH-RLHF[harmless test]", lambda: load_hh(limit=2)),
    ]:
        exs = fn()
        print(f"\n### {name}: {len(exs)} shown")
        for e in exs:
            print({k: (v[:80] + "...") if len(v) > 80 else v for k, v in e.items()})
