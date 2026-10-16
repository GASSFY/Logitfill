"""Teacher-forced NLL under the page-delayed quantization schedule.

A single full-sequence forward cannot represent this schedule: prefill queries
read BF16 keys, while a later query reads only the pages that had already
filled. Scoring therefore prefill the prompt in one piece and then forwards
the response in chunks that do not cross a page flush.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def reference_nll(model, input_ids: torch.Tensor, prompt_len: int) -> float:
    """Full-precision next-token NLL of ``input_ids[prompt_len:]``."""
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    with torch.inference_mode():
        output = model(input_ids=input_ids, labels=labels, use_cache=False)
    return float(output.loss.item())


def _chunk_end(position: int, page_size: int, limit: int) -> int:
    boundary = (position // page_size + 1) * page_size
    return min(limit, boundary)


def quantized_nll(model, input_ids: torch.Tensor, prompt_len: int, page_size: int) -> float:
    """NLL of the response under BF16 prefill and post-attention page flushes."""
    if prompt_len <= 0 or prompt_len >= int(input_ids.shape[-1]):
        raise ValueError("prompt_len must leave at least one response token")
    total = 0.0
    count = 0
    length = int(input_ids.shape[-1])
    with torch.inference_mode():
        prefill = model(input_ids=input_ids[:, :prompt_len], use_cache=True)
        total += _sum_ce(prefill.logits[:, -1], input_ids[:, prompt_len])
        count += 1
        past = prefill.past_key_values
        position = prompt_len
        while position < length - 1:
            stop = _chunk_end(position, page_size, length - 1)
            if stop <= position:
                raise RuntimeError("quantized NLL chunk did not advance")
            output = model(
                input_ids=input_ids[:, position:stop],
                past_key_values=past,
                use_cache=True,
            )
            total += _sum_ce(output.logits, input_ids[:, position + 1 : stop + 1])
            count += stop - position
            past = output.past_key_values
            position = stop
    if count == 0:
        return float("nan")
    return total / count


def _sum_ce(logits: torch.Tensor, target: torch.Tensor) -> float:
    loss = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        reduction="sum",
    )
    return float(loss.item())


def token_nll(model, input_ids: torch.Tensor, prompt_len: int, quantized: bool, page_size: int) -> float:
    if quantized:
        return quantized_nll(model, input_ids, prompt_len, page_size)
    return reference_nll(model, input_ids, prompt_len)
