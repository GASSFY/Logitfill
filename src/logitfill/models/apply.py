"""Install the Key/Value quantizer on a Qwen2 model.

Value uses one fixed per-token width. Key channels use the static level map
installed by ``DeepSeekQwenKVAdapter.set_k_levels``.
"""
from __future__ import annotations

from types import MethodType

from logitfill.models.qwen2_attention import make_qwen2_forward
from logitfill.quantizers.quantizer import make_quantizers, make_value_quantizer


def apply_kv(
    model,
    group_size: int = 128,
    v_group_size: int | None = None,
    page_size: int | None = None,
    bit_v: int = 2,
) -> None:
    if model.__class__.__name__ != "Qwen2ForCausalLM":
        raise TypeError(
            "apply_kv currently supports Qwen2ForCausalLM "
            f"(DeepSeek-R1-Distill-Qwen-7B), got {model.__class__.__name__}"
        )
    page = group_size if page_size is None else page_size
    value_group = group_size if v_group_size is None else v_group_size
    n_q = int(model.config.num_attention_heads)
    head_dim = int(getattr(model.config, "head_dim", None) or model.config.hidden_size // n_q)
    if page % group_size != 0:
        raise ValueError(f"page_size {page} must be a multiple of Key group_size {group_size}")
    if int(bit_v) < 16 and head_dim % value_group != 0:
        raise ValueError(f"head_dim {head_dim} is not divisible by v_group_size {value_group}")
    if int(bit_v) < 2 or int(bit_v) > 16:
        raise ValueError(f"bit_v must be in 2..16, got {bit_v}")
    quantizers = make_quantizers(group_size)
    value_quantizer = None if int(bit_v) >= 16 else make_value_quantizer(int(bit_v), value_group)
    make_forward = make_qwen2_forward(model)
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn.kv_quantizers = quantizers
        attn.v_quantizer = value_quantizer
        attn.k_levels = None
        attn.k_quant_len = 0
        attn.v_quant_len = 0
        attn.kv_group_size = group_size
        attn.v_group_size = value_group
        attn.kv_page_size = page
        attn.bit_v = int(bit_v)
        attn.forward = MethodType(make_forward(i), attn)
