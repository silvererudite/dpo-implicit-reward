"""
Stage 1 -- SFT reference policy pi_ref.

Fine-tune the base model (Qwen2.5-0.5B) with LoRA on the *chosen* responses to obtain the
shared reference policy. DPO trains from this checkpoint; the explicit RM is ALSO initialized
from it (see DESIGN.md) so the comparison is budget-matched from identical starting weights.

Run on Colab / a single consumer GPU:
    python -m src.train_sft --config configs/sft.yaml
"""
from __future__ import annotations
import argparse, inspect, yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

from src.prepare import build_budget_subsets, to_sft
from src.runtime import model_dtype, wandb_setup, log_budget

# TRL API drift: SFTConfig takes `max_seq_length` in trl <0.16 and `max_length` in >=0.16.
# (DPOConfig/RewardConfig use `max_length` in both, so only SFT needs this.)
_SFT_LEN_KEY = "max_length" if "max_length" in getattr(
    SFTConfig, "__dataclass_fields__", {}) else "max_seq_length"


def main(cfg):
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], torch_dtype=model_dtype(cfg))

    train = to_sft(build_budget_subsets("train_prefs")[cfg["budget"]])

    peft_cfg = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"], task_type="CAUSAL_LM",
    )
    _RUN = f"sft-{cfg['budget']}-seed{cfg['seed']}"
    args = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"], lr_scheduler_type="cosine", warmup_ratio=0.03,
        logging_steps=20, save_strategy="epoch",
        max_steps=cfg.get("max_steps", -1),          # >0 for a quick smoke test
        bf16=cfg.get("bf16", True), fp16=cfg.get("fp16", False),  # T4: bf16:false, fp16:true
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=cfg["seed"], run_name=_RUN,
        report_to=wandb_setup(cfg, _RUN),
        **{_SFT_LEN_KEY: cfg["max_length"]},
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=train,
                         processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(cfg["output_dir"])
    log_budget(cfg["output_dir"], trainer, "sft", {"budget": cfg["budget"]})
    print(f"Saved SFT reference policy -> {cfg['output_dir']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    main(yaml.safe_load(open(a.config)))
