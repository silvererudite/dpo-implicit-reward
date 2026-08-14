"""
Materialize the SFT reference policy pi_ref as a standalone checkpoint.

Why this step exists (DESIGN.md items 1 and 6). SFT produces a LoRA *adapter*, but an adapter
cannot serve as the shared initialization the design calls for:

  - DPO: `DPOTrainer(model=<base+SFT adapter>, ref_model=None)` builds its reference by
    *disabling the adapter*, which yields the RAW BASE model -- not the SFT checkpoint. Training
    would then optimize beta*(log pi_theta - log pi_base) while src/eval.py scores against
    base+SFT. Two different quantities, and neither is the pi_ref DESIGN.md specifies.

  - RM: loading a CAUSAL_LM adapter onto AutoModelForSequenceClassification leaves the freshly
    initialized `score` head frozen (the adapter carries no modules_to_save), so the Bradley-Terry
    head never trains -- verified empirically: score head trainable = False.

Merging the adapter into the weights fixes both at once. Afterwards:
  pi_ref            = this merged checkpoint
  DPO policy        = merged + a fresh (zero-init) LoRA  -> adapter-disabled == pi_ref exactly
  explicit RM       = merged backbone + fresh SEQ_CLS LoRA (modules_to_save=["score"])

    python -m src.merge_sft --config configs/sft.yaml
"""
from __future__ import annotations
import argparse, yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from src.runtime import model_dtype


def main(cfg, out_dir: str):
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=model_dtype(cfg))
    merged = PeftModel.from_pretrained(base, cfg["output_dir"]).merge_and_unload()
    merged.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"Merged SFT adapter {cfg['output_dir']} into base -> {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft.yaml")
    ap.add_argument("--out", default="outputs/sft_merged")
    a = ap.parse_args()
    main(yaml.safe_load(open(a.config)), a.out)
