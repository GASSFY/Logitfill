"""Token-by-token NLL.

Prefill is one forward. Each response token is its own forward, so a method
that quantizes on decode steps sees the schedule it was written for.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _ce(logits: torch.Tensor, target: torch.Tensor) -> float:
    loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="sum")
    return float(loss.item())


def stepwise_nll(model, input_ids: torch.Tensor, prompt_len: int, past=None) -> float:
    if prompt_len <= 0 or prompt_len >= int(input_ids.shape[-1]):
        raise ValueError("prompt_len must leave at least one response token")
    total = 0.0
    count = 0
    with torch.inference_mode():
        prefill = model(input_ids=input_ids[:, :prompt_len], past_key_values=past, use_cache=True)
        total += _ce(prefill.logits[:, -1], input_ids[:, prompt_len])
        count += 1
        past = prefill.past_key_values
        for position in range(prompt_len, int(input_ids.shape[-1]) - 1):
            output = model(input_ids=input_ids[:, position : position + 1], past_key_values=past, use_cache=True)
            total += _ce(output.logits[:, -1], input_ids[:, position + 1])
            count += 1
            past = output.past_key_values
    return total / count
