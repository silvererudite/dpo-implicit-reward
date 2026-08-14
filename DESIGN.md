# Design decisions (fairness of the budget-matched comparison)

The whole study hinges on the implicit-vs-explicit comparison being *fair*. Locked choices:

1. **Shared initialization.** SFT produces π_ref. **Both** DPO and the explicit RM start from this
   same SFT checkpoint (identical LoRA-adapted weights), so neither scorer gets a head start.

2. **Compute match = epochs over identical pairs.** Same UltraFeedback pairs, same number of
   epochs, same LoRA config (r=16, α=32, dropout=0.05, same target modules), same effective batch
   size (batch×grad_accum) and LR schedule. Only the *loss* differs (DPO vs Bradley–Terry).
   Both process 2 sequences/pair; DPO adds a frozen reference forward (~one extra forward).
   We log tokens-seen and wall-clock for both so "matched" is auditable.

3. **Fixed hyperparameters** across conditions; only β (DPO) and the training budget vary in sweeps.

4. **Seeds.** ≥3 training seeds for headline numbers → confidence intervals. Data-split seed fixed
   separately from the training seed.

5. **β and eval.** Pairwise ranking within a prompt is invariant to β (a positive scalar), so the
   β sweep tests **calibration/ECE and training dynamics**, not ranking accuracy directly.

6. **Implicit-reward extraction.** `r̂ = β·Σ_t[log π_θ − log π_ref]` over response tokens only
   (prompt masked). Verified against `DPOTrainer`'s logged `rewards/chosen` / `rewards/rejected`.

7. **Scope tiering.** MVP = one budget (8k) + cross-dataset OOD, end-to-end, before any sweep.
   Then widen to β / budget / checkpoint ablations and the RewardBench category breakdown.
