"""Qwen2 attention with page-delayed Key/Value quantization.

Prefill attention runs in the original dtype. Completed pages are compressed
after that attention returns, so they affect only later queries. During
decode the current page stays in the original dtype until it reaches
``page_size`` tokens; the flush also happens after attention.

Key groups run along the token axis and may use a static per-channel level
map (0=int2, 1=int4, 2=int8, 3=fp16). Value groups run along the head
dimension at one fixed width, default int2. ``v_bits >= 16`` leaves Value
uncompressed, which the Key-only mechanism ablations use.
"""
from __future__ import annotations

import torch

from logitfill.quantization.quantizer import quantize_along_features, quantize_along_time


def cache_length(past_key_value, layer_idx: int) -> int:
    if past_key_value is None:
        return 0
    getter = getattr(past_key_value, "get_seq_length", None)
    if getter is not None:
        return int(getter(layer_idx))
    key_cache = getattr(past_key_value, "key_cache", None)
    if key_cache is not None and len(key_cache) > layer_idx and key_cache[layer_idx] is not None:
        return int(key_cache[layer_idx].shape[-2])
    return 0


def quantize_k_channels(group: torch.Tensor, levels, quantizers: dict) -> torch.Tensor:
    """``group`` is ``[B, H, T, D]`` and T is a multiple of the Key group size."""
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


def quantize_completed_pages(attn, key_states: torch.Tensor, value_states: torch.Tensor, past_len: int, step: int) -> None:
    """Fake-quantize every newly completed page in place.

    Tokens beyond the last full page stay in ``key_states`` / ``value_states``
    unchanged. The slice length is a multiple of the Key group because both
    the page size and the previously flushed length are.
    """
    page = int(attn.kv_page_size)
    quant_len = (past_len + step) // page * page
    done = int(getattr(attn, "k_quant_len", 0))
    if quant_len <= done:
        return
    key_states[..., done:quant_len, :] = quantize_k_channels(
        key_states[..., done:quant_len, :],
        attn.k_levels,
        attn.kv_quantizers,
    )
    v_bits = int(getattr(attn, "v_bits", 16))
    if v_bits < 16:
        value_states[..., done:quant_len, :] = quantize_along_features(
            value_states[..., done:quant_len, :],
            attn.v_quantizer,
        )
    attn.k_quant_len = quant_len
    attn.v_quant_len = quant_len


def make_qwen2_forward(model):
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

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

            backend = "fake" if adapter is None else adapter.backend
            if (
                backend == "packed"
                and adapter is not None
                and not adapter.skip_quant
                and not replaced
            ):
                attn_output = _packed_attention(
                    self,
                    adapter,
                    layer_idx,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    past_key_value,
                    kwargs,
                )
            else:
                attn_output = _fake_attention(
                    self,
                    adapter,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    past_key_value,
                    cache_position,
                    input_shape,
                    sin,
                    cos,
                    replaced,
                    kwargs,
                )
            attn_output = attn_output.reshape(*input_shape, -1).contiguous()
            attn_output = self.o_proj(attn_output)
            return attn_output, None

        return forward

    return make_forward


def _attention_interface(attn, kwargs):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.qwen2.modeling_qwen2 import eager_attention_forward

    impl = getattr(attn.config, "_attn_implementation", "sdpa")
    return ALL_ATTENTION_FUNCTIONS.get(impl, eager_attention_forward)


def _sliding_window(attn):
    if (
        getattr(attn.config, "use_sliding_window", False)
        and getattr(attn.config, "sliding_window", None) is not None
        and attn.layer_idx >= attn.config.max_window_layers
    ):
        return attn.config.sliding_window
    return None


def _run_dense_attention(attn, query_states, key_states, value_states, attention_mask, kwargs):
    attn_output, _attn_weights = _attention_interface(attn, kwargs)(
        attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not attn.training else attn.attention_dropout,
        scaling=attn.scaling,
        sliding_window=_sliding_window(attn),
        **kwargs,
    )
    return attn_output


def _fake_attention(
    attn,
    adapter,
    query_states,
    key_states,
    value_states,
    attention_mask,
    past_key_value,
    cache_position,
    input_shape,
    sin,
    cos,
    replaced,
    kwargs,
):
    past_len = cache_length(past_key_value, attn.layer_idx)
    if past_len == 0:
        attn.k_quant_len = 0
        attn.v_quant_len = 0
    if past_key_value is not None:
        key_states, value_states = past_key_value.update(
            key_states,
            value_states,
            attn.layer_idx,
            {"sin": sin, "cos": cos, "cache_position": cache_position},
        )
    attn_output = _run_dense_attention(
        attn, query_states, key_states, value_states, attention_mask, kwargs
    )
    # Compression follows attention. A prefill therefore reads BF16 keys, and a
    # decode token reads the previous history plus the still-BF16 current page.
    skip = adapter is not None and adapter.skip_quant
    if not replaced and not skip:
        quantize_completed_pages(attn, key_states, value_states, past_len, input_shape[1])
    return attn_output


def _packed_attention(
    attn,
    adapter,
    layer_idx,
    query_states,
    key_states,
    value_states,
    attention_mask,
    past_key_value,
    kwargs,
):
    from logitfill.packed.reference import packed_attention

    manager = adapter.packed_manager()
    past_len = manager.seq_len if layer_idx == 0 else manager.layer_length(layer_idx)
    if past_len == 0 and layer_idx == 0:
        manager.reset()
    if past_len == 0:
        attn_output = _run_dense_attention(
            attn, query_states, key_states, value_states, attention_mask, kwargs
        )
    else:
        attn_output = packed_attention(
            manager,
            layer_idx,
            query_states,
            key_states,
            value_states,
            attn.scaling,
        )
    manager.commit(layer_idx, key_states, value_states)
    if past_key_value is not None and hasattr(past_key_value, "note_layer"):
        past_key_value.note_layer(layer_idx, manager.seq_len)
    return attn_output
