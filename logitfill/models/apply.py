"""Install the Key-channel quantizer on a Qwen2 model.

Same shape as PM-KVQ ``apply_kivi``: build the quantizers, hang them on each
attention module, and replace ``forward``. Value is left in the original dtype.
"""
from __future__ import annotations

from types import MethodType

from logitfill.models.qwen2_attention import make_qwen2_forward
from logitfill.quantization.quantizer import make_quantizers


def apply_kv(model, group_size: int = 128) -> None:
    if model.__class__.__name__ != "Qwen2ForCausalLM":
        raise TypeError(
            "apply_kv currently supports Qwen2ForCausalLM "
            f"(DeepSeek-R1-Distill-Qwen-7B), got {model.__class__.__name__}"
        )
    quantizers = make_quantizers(group_size)
    make_forward = make_qwen2_forward(model)
    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        attn.kv_quantizers = quantizers
        attn.k_levels = None
        attn.k_quant_len = 0
        attn.kv_group_size = group_size
        attn.forward = MethodType(make_forward(i), attn)
