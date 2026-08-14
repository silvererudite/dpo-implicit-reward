"""
Stage 2 -- DPO policy pi_theta, from which the IMPLICIT reward is read off.

Trains pi_theta from the SFT reference on UltraFeedback preference pairs, saving intermediate
checkpoints for the training-progress ablation. The implicit reward is
    r_hat(x,y) = beta * [ log pi_theta(y|x) - log pi_ref(y|x) ]
extracted later by src/scoring.py. Sanity-check the extraction against the trainer's own logged
`rewards/chosen` and `rewards/rejected` (same quantity) before trusting eval numbers.

    python -m src.train_dpo --config configs/dpo.yaml --beta 0.1 --budget 8k
"""
from __future__ import annotations
import argparse, yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, PeftModel
from trl import DPOTrainer, DPOConfig

from src.prepare import build_budget_subsets, to_dpo


def load_sft_policy(cfg):
    """Load base + SFT LoRA adapter as the starting policy (and, frozen, as the reference)."""
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(cfg["base_model"])
    policy = PeftModel.from_pretrained(base, cfg["sft_dir"], is_trainable=True)
    return policy, tok


def main(cfg, beta, budget):
    policy, tok = load_sft_policy(cfg)
    train = to_dpo(build_budget_subsets("train_prefs")[budget])

    args = DPOConfig(
        output_dir=f"{cfg['output_root']}/dpo_beta{beta}_{budget}",
        beta=beta,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"], lr_scheduler_type="cosine", warmup_ratio=0.03,
        max_length=cfg["max_length"], max_prompt_length=cfg["max_prompt_length"],
        logging_steps=20,
        # checkpoints across training -> training-progress / overoptimization ablation
        save_strategy="steps", save_steps=cfg["save_steps"],
        max_steps=cfg.get("max_steps", -1),          # >0 for a quick smoke test
        bf16=cfg.get("bf16", True), fp16=cfg.get("fp16", False),  # T4: bf16:false, fp16:true
        seed=cfg["seed"], report_to=cfg.get("report_to", "none"),
    )
    # ref_model=None => TRL uses the frozen base of the PEFT model as pi_ref (LoRA disabled).
    trainer = DPOTrainer(model=policy, ref_model=None, args=args,
                         train_dataset=train, processing_class=tok)
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Saved DPO policy (beta={beta}, budget={budget}) -> {args.output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--budget", default="8k", choices=["2k", "8k", "32k"])
    a = ap.parse_args()
    main(yaml.safe_load(open(a.config)), a.beta, a.budget)
