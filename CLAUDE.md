# CLAUDE.md

Guidance for running this repo on a GPU box (Amazon SageMaker). Read this before running or
changing anything. Human-readable too.

## What this project is

Empirical study: does DPO's **implicit reward** `r̂(x,y) = β·log[π_θ(y|x)/π_ref(y|x)]` generalize
as a reward model as well as an **explicitly-trained Bradley–Terry (BT) reward model**, under
matched data + compute, in-distribution and under distribution shift?
- **H1:** explicit RM > implicit on held-out in-distribution pairs.
- **H2:** the gap widens under distribution shift.

Base model **Qwen2.5-0.5B** + LoRA via HuggingFace TRL. Full plan in `TASKS.md`; fairness
decisions in `DESIGN.md`. Non-neural baseline numbers already computed in `results/baselines.json`.

## Environment — SageMaker GPU

**Recommended instance: `ml.g5.xlarge`** (A10G, 24 GB, **Ampere → bf16 works out of the box**).
`ml.g4dn.xlarge` (T4, 16 GB) also fits the 0.5B model but T4 has **no bf16** — see the bf16 gotcha below.

From a SageMaker Studio/Notebook **terminal**, in the repo root:

```bash
# fresh venv keeps us off the base image's pinned torch
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# sanity: GPU must be visible
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

- Datasets stream from the HuggingFace Hub → the instance needs **internet** (default unless the
  domain is VPC-only). Optional: `huggingface-cli login` (or `export HF_TOKEN=...`) to lift rate limits.
- Every command below is run **from the repo root with `.venv` active** and uses `python -m src.<module>`.

## Run the pipeline (in order)

```bash
# 0. SMOKE TEST FIRST — proves the TRL stack + LoRA + save/reload before spending compute.
#    20 steps on a tiny slice (configs/smoke.yaml). If this fails, fix it before anything else.
python -m src.train_sft --config configs/smoke.yaml

# 1. SFT reference policy π_ref  (shared start for BOTH DPO and the RM)
python -m src.train_sft --config configs/sft.yaml

# 2. DPO policy π_θ  → implicit reward. Start with the reference run:
python -m src.train_dpo --config configs/dpo.yaml --beta 0.1 --budget 8k

# 3. Explicit Bradley–Terry reward model (compute-matched to DPO)
python -m src.train_rm  --config configs/rm.yaml  --budget 8k

# 4. Evaluate implicit vs explicit (+ base_logprob) on ID + RewardBench + HH
python -m src.eval      --config configs/eval.yaml
```

MVP = the four steps above at β=0.1 / 8k → the first H1/H2 numbers. **Then** widen (see `TASKS.md` M8):
β ∈ {0.05,0.1,0.3,0.5}, budget ∈ {2k,8k,32k}, and the training-progress checkpoint curve.

## Things that will bite you (check these first when something breaks)

- **bf16 vs fp16.** Configs default to bf16 (works on Ampere: g5/A10G, A100). On a **T4 (g4dn),
  bf16 is unsupported** → set `bf16: false` and `fp16: true` in the config you're running (no code
  edit needed — the scripts read these keys). Simplest: just use a g5 instance.
- **TRL API drift.** `DPOTrainer`/`RewardTrainer`/`SFTTrainer` signatures change across TRL
  releases (`requirements.txt` pins `trl>=0.12,<0.16`). If you hit a `TypeError` on trainer init
  (e.g. `processing_class` vs `tokenizer`, or `max_length` location), that's the cause — adjust to
  the installed version, don't rewrite the logic.
- **Verify the implicit reward.** After DPO, confirm `src/scoring.implicit_reward` matches
  `DPOTrainer`'s own logged `rewards/chosen` − `rewards/rejected` on a few training pairs. A
  masking/scaling error here silently corrupts every downstream number. This is the #1 correctness check.
- **Budget-matching is the whole point** (`DESIGN.md`): DPO and the RM must share the SFT init and
  see identical pairs/epochs/LoRA/batch/LR. If you change one, change both, and re-log tokens-seen/wall-clock.
- **Benchmark datasets only** (instructor requirement): UltraFeedback (train/ID), RewardBench (OOD),
  HH-RLHF (cross-dataset OOD). Don't swap in ad-hoc data.
- **Small-model noise.** 0.5B accuracy can sit near chance on hard RewardBench subsets; headline
  claims need **≥3 seeds** with CIs before you trust them.

## Where things go

- `outputs/` — model checkpoints (gitignored). `results/eval_*.json` — metric summaries (keep).
  `results/raw_*.json` — per-pair scores for recomputing metrics.
- Regenerate the instructor PDF after new numbers: `python reports/make_report.py`.

## Repo map

```
src/data.py       load the 3 benchmark datasets → unified (prompt,chosen,rejected,subset)
src/prepare.py    → TRL formats + matched budget subsets {2k,8k,32k}
src/baselines.py  non-neural floors (CPU): random / pick-longer / TF-IDF-LR
src/train_sft.py  SFT reference policy (LoRA)
src/train_dpo.py  DPO policy + implicit-reward source (β / budget)
src/train_rm.py   Bradley–Terry reward model (matched compute)
src/scoring.py    implicit_reward / explicit_reward / base_logprob
src/metrics.py    accuracy, ECE, length-controlled, Spearman, McNemar
src/eval.py       score both models on ID + OOD, dump metrics + raw scores
configs/*.yaml    one per stage
```

## Git conventions (this repo)

- Remote: `https://github.com/silvererudite/dpo-implicit-reward.git` (branch `main`).
- On a fresh box, set identity first: `git config user.name "silvererudite" && git config user.email "shamima2hossain@gmail.com"`.
- **Commit as the user only — do NOT add any `Co-Authored-By` trailer.** Plain commit messages.
- Commit `results/*.json` and code; never commit `outputs/`, `.venv/`, or dataset caches (see `.gitignore`).
