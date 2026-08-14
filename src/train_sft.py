"""
Stage 1 -- SFT reference policy pi_ref.

Fine-tune the base model (Qwen2.5-0.5B) with LoRA on the *chosen* responses to obtain the
shared reference policy. DPO trains from this checkpoint; the explicit RM is ALSO initialized
from it (see DESIGN.md) so the comparison is budget-matched from identical starting weights.

Run on Colab / a single consumer GPU:
    python -m src.train_sft --config configs/sft.yaml
"""
from __future__ import annotations
import argparse, yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

from src.prepare import build_budget_subsets, to_sft


def main(cfg):
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"])

    train = to_sft(build_budget_subsets("train_prefs")[cfg["budget"]])

    peft_cfg = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"], task_type="CAUSAL_LM",
    )
    args = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"], lr_scheduler_type="cosine", warmup_ratio=0.03,
        max_length=cfg["max_length"], logging_steps=20, save_strategy="epoch",
        max_steps=cfg.get("max_steps", -1),          # >0 for a quick smoke test
        bf16=cfg.get("bf16", True), fp16=cfg.get("fp16", False),  # T4: bf16:false, fp16:true
        seed=cfg["seed"], report_to=cfg.get("report_to", "none"),
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=train,
                         processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    print(f"Saved SFT reference policy -> {cfg['output_dir']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    main(yaml.safe_load(open(a.config)))
