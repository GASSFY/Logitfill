"""Checks that the LogitFill Key path matches the Stage 6 / Stage 9 buyer.

    python scripts/selfcheck.py
"""
from __future__ import annotations

import numpy as np
import torch

from logitfill.allocators.allocate import self_check as allocate_self_check
from logitfill.models.deepseek_qwen import DeepSeekQwenKVAdapter
from logitfill.quantizers.quantizer import GroupQuantizer, make_quantizers, quantize_along_time


def _close(a: torch.Tensor, b: torch.Tensor, tol: float = 1e-5) -> None:
    err = (a.float() - b.float()).abs().max().item()
    if err > tol:
        raise RuntimeError(f"max abs err {err} > {tol}")


def check_quantizer_matches_pm_kvq() -> None:
    from pm_kvq.quantization.quantizer.quantizer import UntrainableQuantizer

    torch.manual_seed(0)
    states = torch.randn(1, 4, 256, 128)
    for bits in (2, 4, 8):
        theirs = UntrainableQuantizer(
            n_bits=bits,
            granularity="per_group",
            group_size=128,
            symmetric=False,
            round_zeros=False,
        )
        ref = theirs(states.transpose(-1, -2)).transpose(-1, -2)
        got = quantize_along_time(states, GroupQuantizer(bits, 128))
        _close(got, ref, tol=1e-6)
    print("quantizer matches pm_kvq", flush=True)


def check_adapter() -> None:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            attention_dropout=0.0,
        )
    )
    model.eval()
    ids = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        raw = model(input_ids=ids, use_cache=True).past_key_values.key_cache[0].clone()
    adapter = DeepSeekQwenKVAdapter(model, group_size=4).attach()
    with torch.no_grad():
        quant = model(input_ids=ids, use_cache=True).past_key_values.key_cache[0]
    expect = quantize_along_time(raw[:, :, :4], make_quantizers(4)[2])
    _close(quant[:, :, :4], expect)
    _close(quant[:, :, 4:], raw[:, :, 4:])

    fresh = Qwen2ForCausalLM(model.config)
    fresh.load_state_dict(model.state_dict())
    fresh.eval()
    with torch.no_grad():
        raw2 = fresh(input_ids=ids, use_cache=True).past_key_values.key_cache[0].clone()
    adapter2 = DeepSeekQwenKVAdapter(fresh, group_size=4).attach()
    levels = np.zeros((1, 2, raw2.shape[-1]), dtype=np.int64)
    levels[:, :, 0] = 3
    adapter2.set_k_levels(list(levels))
    with torch.no_grad():
        mixed = fresh(input_ids=ids, use_cache=True).past_key_values.key_cache[0]
    _close(mixed[:, :, :4, 0], raw2[:, :, :4, 0])
    rest = quantize_along_time(raw2[:, :, :4], make_quantizers(4)[2])
    _close(mixed[:, :, :4, 1:], rest[:, :, :, 1:])
    adapter.detach()
    adapter2.detach()
    print("qwen2 adapter self-check ok", flush=True)


def main() -> None:
    allocate_self_check()
    check_quantizer_matches_pm_kvq()
    check_adapter()
    print("logitfill self-check ok", flush=True)


if __name__ == "__main__":
    main()
