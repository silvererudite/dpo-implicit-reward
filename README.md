# Exploring DPO's Implicit Reward

Does DPO's **implicit reward** `r̂(x,y) = β·log[π_θ(y|x) / π_ref(y|x)]` generalize as a reward
model as well as an **explicitly-trained Bradley–Terry reward model**, under matched data and
compute — in-distribution and under distribution shift?

- **H1:** the explicit RM beats the implicit reward on held-out in-distribution pairs.
- **H2:** the gap widens under distribution shift.
- A null result (implicit ≈ explicit) would empirically substantiate DPO's "secretly a reward model" claim at small scale.

See `dpo_implicit_reward_proposal.pdf` for the proposal and `TASKS.md` for the full plan.

## Benchmark datasets

Per instructor feedback, all data are established benchmarks:

| Dataset | Role |
|---|---|
| **UltraFeedback-binarized** (`HuggingFaceH4/ultrafeedback_binarized`) | training pairs + in-distribution (ID) test |
| **RewardBench** (`allenai/reward-bench`) | reward-model eval benchmark; subsets (Chat / Chat-Hard / Safety / Reasoning) give distribution shift |
| **Anthropic HH-RLHF** (`Anthropic/hh-rlhf`) | cross-dataset OOD |

## Pipeline

```
SFT (π_ref) ──┬──▶ DPO (π_θ)  ──▶ implicit reward  r̂ = β·log[π_θ/π_ref]
              └──▶ Bradley–Terry RM  ──▶ explicit reward  r_φ
                                    ──▶ eval as preference classifiers (ID + OOD)
```

## Layout

```
src/data.py       load the three benchmark datasets into a unified schema
src/baselines.py  CPU reference floors: random / pick-longer / TF-IDF-LR   (runs locally, no GPU)
src/prepare.py    convert to TRL formats + build matched budget subsets {2k,8k,32k}
src/train_sft.py  Stage 1: SFT reference policy (LoRA)
src/train_dpo.py  Stage 2: DPO policy + implicit-reward source (β / budget sweeps)
src/train_rm.py   Stage 3: explicit Bradley–Terry reward model (matched compute)
src/scoring.py    implicit_reward / explicit_reward / base_logprob scorers
src/metrics.py    accuracy, ECE, length-controlled accuracy, Spearman, McNemar
src/eval.py       Stage 4: score both models on ID + OOD, dump metrics + raw scores
configs/          YAML per stage
results/          metric JSON (baselines.json committed)
reports/          instructor project update (PDF)
```

## Running

**Locally (CPU) — benchmark data + reference baselines:**
```bash
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/python -m src.baselines        # writes results/baselines.json
```

**On Colab / a single GPU — the neural conditions:**
```bash
python -m src.train_sft --config configs/sft.yaml
python -m src.train_dpo --config configs/dpo.yaml --beta 0.1 --budget 8k
python -m src.train_rm  --config configs/rm.yaml  --budget 8k
python -m src.eval      --config configs/eval.yaml
```

## Status (preliminary update)

- ✅ Benchmark data pipeline + reference baselines (real numbers in `results/baselines.json`)
- ✅ Full training + eval code (SFT / DPO / RM / scoring / metrics) — implemented, ready for the Colab run
- ⏳ Neural conditions (implicit reward, explicit RM, base-logprob) — queued for GPU
