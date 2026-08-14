"""
Scorers that turn a model into a preference classifier.

Three neural scorers (all run on GPU / Colab):
  - implicit_reward   : DPO's implicit reward  r_hat(x,y) = beta * [log pi_theta(y|x) - log pi_ref(y|x)]
  - explicit_reward   : Bradley-Terry reward model scalar head  r_phi(x,y)
  - base_logprob      : length-normalized log-prob under a single model (a free neural baseline)

Every scorer maps (prompt, response) -> real number; the pairwise decision is
`score(chosen) > score(rejected)`. The prompt tokens are always masked so only the
response's log-probability contributes to the implicit reward (matching TRL's convention;
verify against DPOTrainer's logged `rewards/chosen` before trusting downstream numbers).
"""
from __future__ import annotations
from typing import List
import torch


def _format(tokenizer, prompt: str, response: str):
    """Apply the chat template if present; return (full_ids, prompt_len)."""
    if getattr(tokenizer, "chat_template", None):
        prompt_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=True,
        )
        full_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt},
             {"role": "assistant", "content": response}],
            add_generation_prompt=False, tokenize=True,
        )
    else:  # plain LM (e.g. Pythia fallback)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(prompt + response, add_special_tokens=False)["input_ids"]
    return full_ids, len(prompt_ids)


@torch.no_grad()
def sequence_logprob(model, tokenizer, prompt: str, response: str, device="cuda") -> float:
    """Sum of log p(response tokens | prompt, earlier response tokens) under `model`."""
    full_ids, prompt_len = _format(tokenizer, prompt, response)
    ids = torch.tensor([full_ids], device=device)
    logits = model(ids).logits.float()                      # [1, T, V]
    logp = torch.log_softmax(logits[:, :-1], dim=-1)        # predict token t+1 from t
    targets = ids[:, 1:]                                    # [1, T-1]
    tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)[0]  # [T-1]
    # response tokens are those whose *target index* >= prompt_len
    resp_mask = torch.arange(1, ids.shape[1], device=device) >= prompt_len
    return float(tok_logp[resp_mask].sum().item())


@torch.no_grad()
def implicit_reward(policy, ref, tokenizer, prompt: str, response: str,
                    beta: float = 0.1, device="cuda") -> float:
    """r_hat = beta * (logprob under policy - logprob under reference), summed over response tokens.

    Note: ranking within a fixed prompt is invariant to beta (positive scalar); beta only
    rescales the implied choice probability, so it matters for calibration/ECE, not accuracy.
    """
    lp_policy = sequence_logprob(policy, tokenizer, prompt, response, device)
    lp_ref = sequence_logprob(ref, tokenizer, prompt, response, device)
    return beta * (lp_policy - lp_ref)


@torch.no_grad()
def base_logprob(model, tokenizer, prompt: str, response: str, device="cuda",
                 length_normalize: bool = True) -> float:
    """Free neural baseline: (length-normalized) log-prob of the response under one model."""
    full_ids, prompt_len = _format(tokenizer, prompt, response)
    n_resp = max(1, len(full_ids) - prompt_len)
    lp = sequence_logprob(model, tokenizer, prompt, response, device)
    return lp / n_resp if length_normalize else lp


@torch.no_grad()
def explicit_reward(rm, tokenizer, prompt: str, response: str, device="cuda") -> float:
    """Bradley-Terry reward-model scalar for (prompt, response)."""
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt},
             {"role": "assistant", "content": response}],
            tokenize=False,
        )
    else:
        text = prompt + response
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    return float(rm(**enc).logits[0].item())
