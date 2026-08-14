# Design decisions (fairness of the budget-matched comparison)

The whole study hinges on the implicit-vs-explicit comparison being *fair*. Locked choices:

1. **Shared initialization.** SFT produces π_ref. **Both** DPO and the explicit RM start from this
   same SFT checkpoint (identical LoRA-adapted weights), so neither scorer gets a head start.

   *Implementation note (why `src/merge_sft.py` exists).* The SFT stage emits a LoRA **adapter**,
   which cannot serve this role directly — two failures, both verified empirically:
   - `DPOTrainer(model=base+SFT_adapter, ref_model=None)` derives π_ref by *disabling the adapter*,
     which returns the **raw base model**, not the SFT checkpoint. Training would optimize
     `β(log π_θ − log π_base)` while `src/eval.py` scored `β(log π_θ − log π_SFT)` — different
     quantities, and neither the π_ref specified here.
   - Loading that `CAUSAL_LM` adapter onto `AutoModelForSequenceClassification` leaves the fresh
     `score` head **frozen at random init** (the adapter carries no `modules_to_save`), so the
     Bradley–Terry head never trains. Measured: `score head trainable = False`.

   So SFT is **merged into the weights once** (`outputs/sft_merged`) and that checkpoint is π_ref.
   DPO then trains a fresh zero-init LoRA on it (⇒ π_θ = π_ref at step 0, implicit reward exactly 0),
   and the RM adds a fresh `SEQ_CLS` LoRA (⇒ `modules_to_save=["score"]`, head trains and is saved).
   Eval loads the same merged checkpoint as π_ref, so training and evaluation agree by construction.

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
