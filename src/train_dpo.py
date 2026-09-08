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

from src.prepare import build_budget_subsets, to_dpo, heldout_pairs
from src.runtime import model_dtype, wandb_setup, log_budget


def load_sft_policy(cfg):
    """Load the MERGED SFT checkpoint (pi_ref) as the policy backbone.

    Must be the merged checkpoint, not base + SFT adapter: with a PEFT model and
    ref_model=None, TRL derives pi_ref by disabling the adapter. On base+SFT-adapter that
    yields the raw base model; on the merged checkpoint it yields the SFT policy, which is
    the pi_ref DESIGN.md specifies and the one src/eval.py scores against. A fresh
    zero-init DPO LoRA is attached by DPOTrainer via peft_config, so pi_theta == pi_ref at
    step 0 and the implicit reward starts at exactly 0. See src/merge_sft.py.
    """
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    policy = AutoModelForCausalLM.from_pretrained(
        cfg["sft_merged_dir"], torch_dtype=model_dtype(cfg))
    return policy, tok


def main(cfg, beta, budget):
    policy, tok = load_sft_policy(cfg)
    # Same length-eligible pool the RM uses, so both conditions train on identical pairs.
    train = to_dpo(build_budget_subsets("train_prefs", tok, cfg["max_length"],
                                        cfg["max_prompt_length"])[budget])

    # Held-out eval is opt-in (config sets eval_steps). It does not alter training, so runs
    # with and without it remain comparable -- only wall-clock differs.
    ev = None
    if cfg.get("eval_steps"):
        ev = to_dpo(heldout_pairs(tok, cfg["max_length"], cfg["max_prompt_length"],
                                  cfg.get("heldout_n", 400)))

    run = f"dpo-beta{beta}-{budget}-seed{cfg['seed']}"
    args = DPOConfig(
        output_dir=f"{cfg['output_root']}/dpo_beta{beta}_{budget}_s{cfg['seed']}",
        beta=beta,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"],
        learning_rate=cfg["lr"], lr_scheduler_type="cosine", warmup_ratio=0.03,
        max_length=cfg["max_length"], max_prompt_length=cfg["max_prompt_length"],
        logging_steps=20,
        # checkpoints across training -> training-progress / overoptimization ablation
        save_strategy="steps", save_steps=cfg["save_steps"],
        eval_strategy=("steps" if ev is not None else "no"),
        eval_steps=cfg.get("eval_steps"),
        per_device_eval_batch_size=cfg.get("eval_batch_size", 8),
        max_steps=cfg.get("max_steps", -1),          # >0 for a quick smoke test
        bf16=cfg.get("bf16", True), fp16=cfg.get("fp16", False),  # T4: bf16:false, fp16:true
        gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=cfg["seed"], run_name=run,
        report_to=wandb_setup(cfg, run),
    )
    # Fresh DPO adapter on top of the merged SFT policy. ref_model=None => TRL disables this
    # adapter to get pi_ref, which is now exactly the merged SFT checkpoint.
    peft_cfg = LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"], task_type="CAUSAL_LM",
    )
    trainer = DPOTrainer(model=policy, ref_model=None, args=args,
                         train_dataset=train, eval_dataset=ev,
                         processing_class=tok, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(args.output_dir)
    log_budget(args.output_dir, trainer, "dpo", {"beta": beta, "budget": budget})
    print(f"Saved DPO policy (beta={beta}, budget={budget}) -> {args.output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--budget", default="8k", choices=["2k", "8k", "32k"])
    ap.add_argument("--seed", type=int, default=None, help="override cfg seed (multi-seed runs)")
    a = ap.parse_args()
    _cfg = yaml.safe_load(open(a.config))
    if a.seed is not None:
        _cfg["seed"] = a.seed
    main(_cfg, a.beta, a.budget)
