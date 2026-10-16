"""Sampling loop shared by the fake, packed, and BF16 backends.

The loop calls the model directly so a packed cache is not replaced by a
dynamic BF16 cache inside ``generate``.
"""
from __future__ import annotations

import torch


def sample_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    scores = logits.float()
    if temperature <= 0:
        return torch.argmax(scores, dim=-1, keepdim=True)
    scores = scores / temperature
    if top_p < 1:
        ordered, indices = torch.sort(scores, descending=True)
        cumulative = torch.cumsum(torch.softmax(ordered, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        ordered = ordered.masked_fill(remove, float("-inf"))
        scores = torch.full_like(scores, float("-inf")).scatter(-1, indices, ordered)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.multinomial(probabilities, num_samples=1)


def generate_text(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    past_key_values=None,
) -> tuple[str, int]:
    """Return the decoded completion and the number of new tokens."""
    eos = tokenizer.eos_token_id
    produced = []
    with torch.inference_mode():
        output = model(input_ids=prompt_ids, use_cache=True, past_key_values=past_key_values)
        logits = output.logits[:, -1]
        past = output.past_key_values
        for _ in range(max_new_tokens):
            token = sample_token(logits, temperature, top_p)
            produced.append(token)
            if eos is not None and int(token.item()) == int(eos):
                break
            output = model(input_ids=token, use_cache=True, past_key_values=past)
            logits = output.logits[:, -1]
            past = output.past_key_values
    if not produced:
        return "", 0
    tokens = torch.cat(produced, dim=1)
    text = tokenizer.decode(tokens[0], skip_special_tokens=True)
    return text, int(tokens.shape[-1])
