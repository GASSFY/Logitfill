"""Attach one baseline to a freshly loaded Hugging Face model.

The official KIVI checkout only ships Llama and Mistral modules, so the KIVI
arm uses the Qwen2 KIVI forward in the PM-KVQ checkout. MixKVQ is a separate
engine and is not attached here.
"""
from __future__ import annotations

from logitfill.baselines.repos import ensure_importable


def _quantizer_config(n_bits: int, group_size: int) -> dict:
    return {
        "n_bits": int(n_bits),
        "granularity": "per_group",
        "group_size": int(group_size),
        "symmetric": False,
        "round_zeros": False,
    }


def attach_logitfill(model, bit_v: int, page_size: int) -> None:
    from logitfill.models.deepseek_qwen import DeepSeekQwenKVAdapter

    DeepSeekQwenKVAdapter(model, group_size=page_size, page_size=page_size, bit_v=bit_v).attach()


def attach_kivi(model, bit_k: int, bit_v: int, group_size: int) -> None:
    ensure_importable("pm_kvq")
    from pm_kvq.quantization.methods.kivi.apply_kivi import apply_kivi

    apply_kivi(
        model,
        _quantizer_config(bit_k, group_size),
        _quantizer_config(bit_v, group_size),
    )


def attach_kvtuner(model, bit_k: int, bit_v: int, group_size: int, residual_length: int):
    import torch
    from flexible_quant.flexible_quantized_cache import FlexibleQuantizedCacheConfig, FlexibleVanillaQuantizedCache

    config = FlexibleQuantizedCacheConfig(
        backend="vanilla",
        nbits_key=int(bit_k),
        nbits_value=int(bit_v),
        axis_key=1,
        axis_value=0,
        asym=True,
        q_group_size=int(group_size),
        residual_length=int(residual_length),
        compute_dtype=torch.bfloat16,
        device="cuda",
    )
    return FlexibleVanillaQuantizedCache(cache_config=config)


def attach_pm_kvq(model, prompt_ids: list[int], budget_mb: float, settings: dict) -> None:
    ensure_importable("pm_kvq")
    from pm_kvq.quantization.methods.pm_kvq.progressive.apply_progressive import apply_progressive
    from pm_kvq.quantization.methods.pm_kvq.smoothattention.apply_smoothattention import (
        apply_smoothattention_rep,
        get_max_keys,
    )

    max_keys = get_max_keys(model, [{"input_ids": prompt_ids}])
    apply_smoothattention_rep(model, max_keys=max_keys)
    apply_progressive(
        model,
        float(budget_mb),
        int(settings.get("n_sink_token", 1)),
        int(settings.get("n_sink_token_bits", 16)),
        int(settings.get("n_window_token", 128)),
        int(settings.get("n_window_token_bits", 16)),
        int(settings.get("n_init_kv_bits", 16)),
    )
