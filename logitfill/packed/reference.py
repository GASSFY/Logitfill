"""Dense reference attention for the packed cache, with a CUDA fast path."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from logitfill.packed.kernel_loader import get_kernel


def repeat_kv(states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return states
    batch, heads, tokens, dim = states.shape
    states = states[:, :, None].expand(batch, heads, repeats, tokens, dim)
    return states.reshape(batch, heads * repeats, tokens, dim)


def _causal_mask(query_len: int, key_len: int, device) -> torch.Tensor | None:
    """Float mask for queries aligned to the right edge of the key sequence."""
    if query_len == 1:
        return None
    mask = torch.zeros(query_len, key_len, dtype=torch.float32, device=device)
    for index in range(query_len):
        visible = key_len - query_len + index + 1
        if visible < key_len:
            mask[index, visible:] = float("-inf")
    return mask


def reference_attention(manager, layer_idx: int, query: torch.Tensor, key_new: torch.Tensor, value_new: torch.Tensor, scaling: float) -> torch.Tensor:
    """Dequantize history and run SDPA. ``query`` is ``[1, heads, tokens, dim]``."""
    if query.shape[0] != 1:
        raise NotImplementedError("packed reference attention supports batch size 1")
    layer = manager.layers[layer_idx]
    key_hist, value_hist = layer.dequantize()
    pieces_k = [key_hist]
    pieces_v = [value_hist]
    if layer.tail_k is not None:
        pieces_k.append(layer.tail_k.float())
        pieces_v.append(layer.tail_v.float())
    pieces_k.append(key_new[0].float())
    pieces_v.append(value_new[0].float())
    key = torch.cat(pieces_k, dim=1).unsqueeze(0)
    value = torch.cat(pieces_v, dim=1).unsqueeze(0)
    repeats = manager.n_q // manager.n_kv
    key = repeat_kv(key, repeats)
    value = repeat_kv(value, repeats)
    mask = _causal_mask(query.shape[-2], key.shape[-2], query.device)
    output = F.scaled_dot_product_attention(
        query.float(),
        key,
        value,
        attn_mask=mask,
        scale=scaling,
    )
    return output.to(dtype=query.dtype)


def _kernel_attention(manager, layer_idx: int, query: torch.Tensor, key_new: torch.Tensor, value_new: torch.Tensor, scaling: float):
    kernel = get_kernel()
    layer = manager.layers[layer_idx]
    if kernel is None or query.shape[-2] != 1 or manager.v_bits != 2 or not query.is_cuda:
        return None
    dim = int(query.shape[-1])
    if dim > 1024 or dim & (dim - 1):
        return None
    if layer.hist_k is None:
        return None
    recent_k = key_new[0]
    recent_v = value_new[0]
    if layer.tail_k is not None:
        recent_k = torch.cat((layer.tail_k, recent_k), dim=1)
        recent_v = torch.cat((layer.tail_v, recent_v), dim=1)
    recent_k = recent_k.to(torch.float16).contiguous()
    recent_v = recent_v.to(torch.float16).contiguous()
    tiers = layer.table
    hist = layer.hist_k
    value = layer.hist_v
    arguments = (
        query[0, :, 0].float().contiguous(),
        float(scaling),
        int(layer.k_group_size),
        int(layer.v_group_size),
        tiers["tier"].to(torch.int32).contiguous(),
        tiers[0]["local"].contiguous(),
        tiers[1]["local"].contiguous(),
        tiers[2]["local"].contiguous(),
        tiers[3]["local"].contiguous(),
        hist[0]["code"].contiguous(),
        hist[0]["scale"].contiguous(),
        hist[0]["min"].contiguous(),
        hist[1]["code"].contiguous(),
        hist[1]["scale"].contiguous(),
        hist[1]["min"].contiguous(),
        hist[2]["code"].contiguous(),
        hist[2]["scale"].contiguous(),
        hist[2]["min"].contiguous(),
        hist[3]["value"].contiguous(),
        value["code"].contiguous(),
        value["scale"].contiguous(),
        value["min"].contiguous(),
        recent_k,
        recent_v,
    )
    try:
        output = kernel.logitfill_decode(*arguments)
    except Exception:
        return None
    return output.to(dtype=query.dtype).view(1, query.shape[1], 1, dim)


def packed_attention(manager, layer_idx: int, query: torch.Tensor, key_new: torch.Tensor, value_new: torch.Tensor, scaling: float) -> torch.Tensor:
    fast = _kernel_attention(manager, layer_idx, query, key_new, value_new, scaling)
    if fast is not None:
        return fast
    return reference_attention(manager, layer_idx, query, key_new, value_new, scaling)
