"""
Stage 3 -- Explicit Bradley-Terry reward model r_phi.

Trained from the SAME SFT checkpoint as DPO (fairness: identical starting weights), with a scalar
value head and the BT pairwise loss
    L = -E[ log sigma( r_phi(x, y_chosen) - r_phi(x, y_rejected) ) ].
Compute is matched to DPO: same pairs, epochs, LoRA config, effective batch size, LR
(see DESIGN.md). Log tokens-seen / wall-clock so "budget-matched" is auditable.

    python -m src.train_rm --config configs/rm.yaml --budget 8k
"""
from __future__ import annotations
import argparse, yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import RewardTrainer, RewardConfig

from src.prepare import build_budget_subsets, to_rm


def main(cfg, budget):
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # scalar head (num_labels=1) on top of the base transformer
    base = AutoModelForSequenceClassification.from_pretrained(
        cfg["base_model"], num_labels=1)
    base.config.pad_token_id = tok.pad_token_id
    # initialize the backbone from the shared SFT adapter for a fair, matched start
    model = PeftModel.from_pretrained(base, cfg["sft_dir"], is_trainable=True) \
        if cfg.get("sft_dir") else base

    peft_cfg = None if cfg.get("sft_dir") else LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"], task_type="SEQ_CLS",
    )

    train = to_rm(build_budget_subsets("train_prefs")[budget])

    args = RewardConfig(
        output_dir=f"{cfg['output_root']}/rm_{budget}",
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"], lr_scheduler_type="cosine", warmup_ratio=0.03,
        max_length=cfg["max_length"], logging_steps=20, save_strategy="epoch",
        bf16=True, seed=cfg["seed"], report_to=cfg.get("report_to", "none"),
    )
    trainer = RewardTrainer(model=model, args=args, train_dataset=train,
                            processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Saved explicit reward model (budget={budget}) -> {args.output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--budget", default="8k", choices=["2k", "8k", "32k"])
    a = ap.parse_args()
    main(yaml.safe_load(open(a.config)), a.budget)
