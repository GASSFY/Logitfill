"""Qwen2 attention forward with per-channel Key groups.

Completed groups of ``group_size`` tokens are fake-quantized. The residual
tail stays in the original dtype. Value is not quantized. Channel levels are
0=int2, 1=int4, 2=int8, 3=fp16. Missing levels mean every channel is int2.
"""
from __future__ import annotations

import torch

from logitfill.quantization.quantizer import quantize_along_time


def quantize_channels(group: torch.Tensor, levels, quantizers: dict) -> torch.Tensor:
    """``group`` is ``[B, H, T, D]`` and T is a multiple of the group size."""
    if levels is None:
        return quantize_along_time(group, quantizers[2])
    level = torch.as_tensor(levels, device=group.device)
    out = group.clone()
    for code, bits in ((0, 2), (1, 4), (2, 8)):
        mask = level == code
        if not bool(mask.any()):
            continue
        quant = quantize_along_time(group, quantizers[bits])
        out = torch.where(mask.view(1, level.shape[0], 1, level.shape[1]), quant, out)
    return out


def quantize_new_groups(attn, key_states: torch.Tensor, past_len: int, step: int) -> None:
    group = int(attn.kv_group_size)
    quant_len = (past_len + step) // group * group
    done = int(getattr(attn, "k_quant_len", 0))
    if quant_len <= done:
        return
    key_states[..., done:quant_len, :] = quantize_channels(
        key_states[..., done:quant_len, :],
        attn.k_levels,
        attn.kv_quantizers,
    )
    attn.k_quant_len = quant_len


def make_qwen2_forward(model):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen2.modeling_qwen2 import (
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    def make_forward(layer_idx: int):
        def forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask=None,
            past_key_value=None,
            cache_position=None,
            **kwargs,
        ):
            adapter = getattr(model, "_kv_adapter", None)
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin
            )
            if adapter is not None and adapter.capture:
                adapter.captured["Q"][layer_idx] = query_states.detach()
                adapter.captured["K"][layer_idx] = key_states.detach()
                adapter.captured["V"][layer_idx] = value_states.detach()

            replaced = False
            overrides = None if adapter is None else adapter.k_overrides
            if overrides is not None and layer_idx in overrides:
                key_states = overrides[layer_idx].to(
                    device=key_states.device, dtype=key_states.dtype
                )
                replaced = True

            past_len = 0
            if past_key_value is not None and len(past_key_value.key_cache) > layer_idx:
                past_len = past_key_value.key_cache[layer_idx].shape[-2]
            if past_len == 0:
                self.k_quant_len = 0
            if past_key_value is not None:
                key_states, value_states = past_key_value.update(
                    key_states,
                    value_states,
                    self.layer_idx,
                    {"sin": sin, "cos": cos, "cache_position": cache_position},
                )
            if adapter is not None and adapter.skip_quant:
                replaced = True
            if not replaced:
                quantize_new_groups(self, key_states, past_len, input_shape[1])

            impl = getattr(self.config, "_attn_implementation", "sdpa")
            attention_interface = ALL_ATTENTION_FUNCTIONS.get(impl, eager_attention_forward)
            sliding_window = None
            if (
                getattr(self.config, "use_sliding_window", False)
                and getattr(self.config, "sliding_window", None) is not None
                and self.layer_idx >= self.config.max_window_layers
            ):
                sliding_window = self.config.sliding_window
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=sliding_window,
                **kwargs,
            )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, attn_weights

        return forward

    return make_forward
