# DPO Implicit Reward — Task Plan

Execution plan for the project **"Exploring DPO's Implicit Reward"** (see `dpo_implicit_reward_proposal.pdf`).

**Goal:** Under matched data + compute, compare held-out preference accuracy of DPO's implicit reward
`r̂(x,y) = β·Σ_t [log π_θ(y_t|·) − log π_ref(y_t|·)]` against an explicitly-trained Bradley–Terry (BT) reward model,
in-distribution (ID) and under distribution shift (OOD).

**Hypotheses:** H1 — explicit RM > implicit on held-out ID pairs. H2 — the gap widens under distribution shift.
A null result (implicit ≈ explicit) is equally publishable.

---

## Status & updates (2026-08-14)

- **Instructor feedback: use benchmark datasets.** Data pivoted from ad-hoc HH splits to established
  benchmarks — **train on UltraFeedback-binarized, evaluate on RewardBench** (its Chat / Chat-Hard /
  Safety / Reasoning subsets give controlled distribution shift), **HH-RLHF as cross-dataset OOD**.
  This makes the gold-judge OOD set (old c2) optional stretch rather than core.
- **Implemented so far:** benchmark data pipeline (`src/data.py`, `src/prepare.py`), non-neural
  reference baselines run on all sets (`src/baselines.py` → `results/baselines.json`), full neural
  pipeline written for Colab (`train_sft` / `train_dpo` / `train_rm` / `scoring` / `metrics` / `eval`),
  and the instructor project-update PDF (`reports/project_update.pdf`).
- **Next:** run SFT → DPO → BT-RM at the 8k budget on Colab GPU for the first H1/H2 numbers.

---

## 0. Design decisions to lock before coding

These determine whether the "budget-matched" comparison is actually fair — the entire validity of the project rests here.

- [ ] **RM initialization.** Initialize the explicit RM from the **same SFT checkpoint** π_ref (not raw base), so DPO and RM share identical starting weights. Document this as the fairness choice.
- [ ] **Compute-match definition.** Match on **epochs over the identical preference pairs** with identical LoRA config, effective batch size, and optimizer. Both DPO and BT process 2 sequences/pair; DPO adds a frozen reference forward pass (~same FLOPs as one extra forward). Log tokens-seen and wall-clock for both and report them so "matched" is auditable.
- [ ] **Fixed hyperparameters across conditions.** LoRA rank/alpha/dropout, target modules, LR, scheduler, warmup, max sequence length, effective batch size, weight decay, seed set. Only the *loss* and *β* vary.
- [ ] **Seeds.** Run ≥3 seeds for every headline number so H1/H2 claims carry confidence intervals. Fix data-split seed separately from training seed.
- [ ] **β at eval for implicit reward.** The ranking of a pair is invariant to β (it's a positive scalar multiplier), so β only affects *calibration/ECE*, not accuracy. Note this explicitly — the β sweep tests calibration + training dynamics, not pairwise accuracy directly. (Different β *checkpoints* differ because training differs, not because of the eval scaling.)
- [ ] **Gold judge choice** (drives OOD-2 + Spearman). Decide: larger local instruct model (e.g. Qwen2.5-7B-Instruct, 4-bit) vs. an API judge. Freeze it; version-pin it.
- [ ] **Scope tiering.** MVP = full pipeline at one budget (8k) + cross-subset OOD. Stretch = full β/budget/checkpoint ablations + gold-judge OOD. Build MVP end-to-end first, then widen.

**Deliverable:** a short `DESIGN.md` recording each choice + rationale.

---

## Milestone 1 — Environment & scaffolding

- [ ] Confirm compute: single GPU, VRAM available (Colab Pro / cluster / local). Note VRAM ceiling — dictates gold-judge feasibility.
- [ ] Create env; pin versions (`torch`, `transformers`, `trl`, `peft`, `accelerate`, `datasets`, `bitsandbytes`, `wandb`, `scipy`, `scikit-learn`). **Pin `trl`** — `DPOTrainer`/`RewardTrainer` APIs shift across releases.
- [ ] Repo layout: `src/` (data, train_sft, train_dpo, train_rm, score, eval), `configs/` (YAML per condition), `scripts/`, `results/`, `notebooks/`.
- [ ] Config system (YAML or Hydra) so every run is a config file → reproducible + sweepable.
- [ ] `wandb` (or CSV) logging wired for loss/accuracy/eval metrics.
- [ ] **Smoke test:** load Qwen2.5-0.5B, run a 20-step LoRA fine-tune on 100 examples, save + reload adapter. Confirms the whole stack before committing compute.
- [ ] Fallback path: verify Pythia-410M loads with the same code (swap via config).

**Deliverable:** repo skeleton + passing smoke test.

---

## Milestone 2 — Data pipeline & splits  ✅ implemented (`src/data.py`, `src/prepare.py`)

Per instructor feedback, all data are **established benchmarks** (not ad-hoc splits):

- [x] **UltraFeedback-binarized** (`HuggingFaceH4/ultrafeedback_binarized`) — training pairs + **(b) in-distribution held-out test** (`test_prefs`).
- [x] **RewardBench** (`allenai/reward-bench`) — the standard reward-model eval benchmark; its **Chat / Chat-Hard / Safety / Reasoning** subsets are the **(c1) OOD** sets (controlled distribution shift), now the primary OOD in place of the ad-hoc gold-judge set.
- [x] **Anthropic HH-RLHF** (`Anthropic/hh-rlhf`, harmless-base) — **(c2) cross-dataset OOD**.
- [x] Unified schema `(prompt, chosen, rejected, subset)`; chat template applied consistently for **all** conditions (SFT/DPO/RM/eval format identically).
- [x] Budget subsets **2k / 8k / 32k** train pairs, nested (32k ⊃ 8k ⊃ 2k) so scale is the only variable (`build_budget_subsets`).
- [ ] (stretch) gold-judge-labeled third-party generations as an extra OOD-response set + scalar Spearman ratings.
- [ ] Length filtering / truncation policy (max prompt + response length) documented.
- [x] **Data card** proxied by per-set reference baselines (pick-longer, TF-IDF) in `results/baselines.json`.

**Deliverable:** benchmark loaders + matched budget subsets (done).

---

## Milestone 3 — SFT reference policy π_ref

- [ ] Fine-tune base model with LoRA on **chosen** responses (SFT objective) → π_ref.
- [ ] Sanity: loss decreases; sample generations are coherent; adapter saves/loads.
- [ ] Freeze + archive π_ref checkpoint — the shared starting point for DPO and (per §0) the RM.

**Deliverable:** π_ref checkpoint + training curves.

---

## Milestone 4 — DPO training + implicit reward extraction

- [ ] Train π_θ from π_ref on train pairs via TRL `DPOTrainer` (LoRA).
- [ ] **Checkpoint at multiple steps** (e.g. 10/25/50/75/100% of training) for the training-progress ablation.
- [ ] β sweep: {0.05, 0.1, 0.3, 0.5} — one run each (start with β=0.1 as the default reference run).
- [ ] Budget sweep: {2k, 8k, 32k} at the default β.
- [ ] Implement `implicit_reward(model, ref, x, y)` = β · Σ_t (log π_θ − log π_ref) over response tokens only (mask the prompt). **Verify** against `DPOTrainer`'s internal `chosen_rewards`/`rejected_rewards` logging to confirm the extraction matches TRL's convention.
- [ ] Unit test: on a train pair the model was optimized on, chosen implicit reward > rejected (margin should be positive on average).

**Deliverable:** DPO checkpoints (β × budget × step grid) + verified scoring fn.

---

## Milestone 5 — Explicit Bradley–Terry RM

- [ ] Train RM from the **same SFT checkpoint** (§0) with a scalar value head via TRL `RewardTrainer`; BT loss `−log σ(r(x,y_c) − r(x,y_r))`, LoRA + head.
- [ ] Match DPO's compute: same epochs/pairs/LoRA/batch/LR (§0). Log tokens-seen + wall-clock.
- [ ] Budget sweep: {2k, 8k, 32k}, matched to DPO.
- [ ] Implement `explicit_reward(rm, x, y)` = scalar head output on the full sequence.
- [ ] Unit test: train-pair margins positive; eval accuracy on a tiny held-out slice is above chance.

**Deliverable:** RM checkpoints (budget grid) + scoring fn.

---

## Milestone 6 — Evaluation harness, metrics & gold judge

- [ ] Unified scorer interface: any model → per-response scalar; pairwise decision = sign of score difference.
- [ ] **Primary metric — pairwise accuracy** on sets (b), (c1), (c2) for both scorers.
- [ ] **Calibration / ECE:** implied `P(y_c ≻ y_r) = σ(Δscore)`; bin + expected calibration error. (For implicit, this is where β matters.)
- [ ] **Length-controlled accuracy:** report accuracy on length-matched pairs (|Δlen| below threshold) and the length-only baseline, to isolate the length confound for *both* scorers.
- [ ] **Gold judge setup:**
  - [ ] Stand up the frozen judge (per §0). Prompt template for pairwise + scalar (1–10) rating; fix decoding (temp 0), version, and prompt.
  - [ ] Generate OOD-2 responses: sample from 2–3 third-party small instruct models on held-out prompts; judge labels the winner → OOD-2 preference pairs.
  - [ ] **Spearman correlation** of each scorer's scores vs gold scalar ratings on a shared response pool.
- [ ] Persist **all raw per-pair scores to disk** (parquet/CSV) so every metric is recomputable without re-running models.
- [ ] **Paired significance:** McNemar's test (or bootstrap over pairs) for the implicit-vs-explicit accuracy difference on each test set.

**Deliverable:** `eval.py` producing a tidy results table + saved raw scores.

---

## Milestone 7 — Main experiments (test H1 & H2)

- [ ] Headline run: default β, 8k budget, ≥3 seeds → explicit vs implicit accuracy on (b), (c1), (c2) with CIs.
- [ ] **H1:** compare on held-out ID (b). **H2:** compare the ID→OOD *gap* between the two scorers.
- [ ] Report calibration, length-controlled accuracy, and Spearman alongside.
- [ ] Summary table + forest/bar plot with confidence intervals; McNemar p-values.

**Deliverable:** main results table + figures + a paragraph stating whether H1/H2 hold.

---

## Milestone 8 — Ablations

- [ ] **β sweep** {0.05,0.1,0.3,0.5}: implicit accuracy (should be ~flat) + ECE (should move) → separates ranking from calibration.
- [ ] **Preference budget** {2k,8k,32k}: does the explicit–implicit gap shrink/grow with data scale? Plot both curves.
- [ ] **Training-progress curve:** implicit accuracy across DPO checkpoints vs a policy-quality proxy (e.g. win-rate or held-out reward) — does RM quality peak before/after policy quality? (overoptimization link.)

**Deliverable:** three ablation figures + interpretation.

---

## Milestone 9 — Analysis, writeup & presentation

- [ ] Aggregate all results; consolidated tables/figures; reproducibility appendix (configs, seeds, versions, tokens-seen/wall-clock proving budget-match).
- [ ] Write findings against H1/H2; discuss the practical implication (can DPO checkpoints be reused as free RMs?) and threats to validity (0.5B scale, single dataset, judge bias).
- [ ] Final report + slides/demo per course requirements.
- [ ] Clean repo + README so a grader can reproduce the headline number with one command.

**Deliverable:** report, slides, reproducible repo.

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| TRL API drift breaks `DPOTrainer`/`RewardTrainer` | Pin versions; verify implicit-reward extraction against TRL's own reward logging. |
| Gold judge (7B) won't fit alongside training on one GPU | 4-bit quantize, run judge as a **separate offline pass** (not concurrent), or use an API judge. |
| 0.5B reward accuracy near chance / noisy | ≥3 seeds + CIs; if too noisy, escalate to Qwen2.5-1.5B or reduce to the cleaner helpful subset. |
| Length confound dominates the signal | Length-controlled metric + length-only baseline reported everywhere. |
| Third-party generation (OOD-2) blows the time budget | It's stretch scope — MVP ships with cross-subset OOD (c1) only. |
| Compute-match disputed by grader | Log + report tokens-seen and wall-clock for both; state the epoch-matched definition up front. |

---

## Suggested sequencing / critical path

```
M1 → M2 → M3(SFT) → ┬→ M4(DPO) ─┐
                    └→ M5(RM)  ─┴→ M6(eval) → M7(main) → M8(ablations) → M9(writeup)
```

- **MVP (prove the pipeline end-to-end first):** M1 → M2 → M3 → M4(β=0.1, 8k) → M5(8k) → M6(accuracy + length-control, ID + c1) → M7. Get one honest H1 number before scaling.
- **Then widen:** budget sweep, β sweep, checkpoint curve, gold-judge OOD-2, Spearman/ECE, extra seeds.
- M4 and M5 are independent once π_ref exists → run them in parallel.

## Rough effort (adjust to your deadline)

| Milestone | Size |
|---|---|
| M1 setup | S |
| M2 data | M |
| M3 SFT | S–M |
| M4 DPO (+sweeps) | M–L |
| M5 explicit RM | M |
| M6 eval + gold judge | L (judge/OOD-2 is the long pole) |
| M7 main experiments | M |
| M8 ablations | M |
| M9 writeup | M |
