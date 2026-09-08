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
from src.runtime import model_dtype, wandb_setup, log_budget


def main(cfg, budget):
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Scalar head (num_labels=1) on the MERGED SFT backbone -- the same starting weights DPO
    # uses (DESIGN.md item 1). Loading the SFT *adapter* onto a SEQ_CLS model instead would
    # leave the freshly initialized `score` head frozen (a CAUSAL_LM adapter carries no
    # modules_to_save), so the Bradley-Terry head would never train. See src/merge_sft.py.
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["sft_merged_dir"], num_labels=1, torch_dtype=model_dtype(cfg))
    model.config.pad_token_id = tok.pad_token_id

    # task_type="SEQ_CLS" makes PEFT add modules_to_save=["score"], so the head trains and is saved.
    peft_cfg = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"], task_type="SEQ_CLS",
    )

    # Identical pairs to DPO: same length-eligible pool, so nothing is dropped here that
    # DPO merely truncated (that mismatch made the 8k budgets differ, 7520 vs 8000).
    train = to_rm(build_budget_subsets("train_prefs", tok, cfg["max_length"],
                                       cfg.get("max_prompt_length", cfg["max_length"]))[budget])

    run = f"rm-{budget}-seed{cfg['seed']}"
    args = RewardConfig(
        output_dir=f"{cfg['output_root']}/rm_{budget}_s{cfg['seed']}",
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"], lr_scheduler_type="cosine", warmup_ratio=0.03,
        max_length=cfg["max_length"], logging_steps=20, save_strategy="epoch",
        max_steps=cfg.get("max_steps", -1),          # >0 for a quick smoke test
        bf16=cfg.get("bf16", True), fp16=cfg.get("fp16", False),  # T4: bf16:false, fp16:true
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=cfg["seed"], run_name=run, report_to=wandb_setup(cfg, run),
    )
    trainer = RewardTrainer(model=model, args=args, train_dataset=train,
                            processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(args.output_dir)
    log_budget(args.output_dir, trainer, "rm", {"budget": budget})
    print(f"Saved explicit reward model (budget={budget}) -> {args.output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--budget", default="8k", choices=["2k", "8k", "32k"])
    ap.add_argument("--seed", type=int, default=None, help="override cfg seed (multi-seed runs)")
    a = ap.parse_args()
    _cfg = yaml.safe_load(open(a.config))
    if a.seed is not None:
        _cfg["seed"] = a.seed
    main(_cfg, a.budget)
