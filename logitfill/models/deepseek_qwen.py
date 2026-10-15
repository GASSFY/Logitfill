"""KV adapter for DeepSeek-R1-Distill-Qwen-7B.

That checkpoint is a Qwen2 model (28 layers, 4 KV heads, head dim 128).
``attach`` installs the Qwen2 Key forward from ``apply_kv``. Every Key
channel starts at 2 bit. ``set_k_levels`` raises individual channels to 4, 8,
or 16 bit. Value is not quantized.
"""
from __future__ import annotations

from typing import Optional

from torch import nn

from logitfill.models.apply import apply_kv


class DeepSeekQwenKVAdapter:
    """Attach per-channel Key quantization to one loaded model."""

    def __init__(self, model: nn.Module, group_size: int = 128):
        if model.__class__.__name__ != "Qwen2ForCausalLM":
            raise TypeError(
                "DeepSeek-R1-Distill-Qwen-7B is Qwen2ForCausalLM, "
                f"got {model.__class__.__name__}"
            )
        if group_size < 1:
            raise ValueError("group_size must be positive")
        self.model = model
        self.group_size = group_size
        self.n_layers = int(model.config.num_hidden_layers)
        self.n_kv = int(model.config.num_key_value_heads)
        self.head_dim = int(model.config.hidden_size / model.config.num_attention_heads)
        self.capture = False
        self.skip_quant = False
        self.captured: dict = {"Q": {}, "K": {}, "V": {}}
        self.k_overrides: Optional[dict] = None
        self._originals: list = []
        self._attached = False

    def attach(self) -> "DeepSeekQwenKVAdapter":
        if self._attached:
            return self
        self._originals = [layer.self_attn.forward for layer in self.model.model.layers]
        self.model._kv_adapter = self
        apply_kv(self.model, group_size=self.group_size)
        self._attached = True
        return self

    def detach(self) -> None:
        if not self._attached:
            return
        for layer, original in zip(self.model.model.layers, self._originals):
            layer.self_attn.forward = original
        self._originals = []
        self._attached = False
        if getattr(self.model, "_kv_adapter", None) is self:
            self.model._kv_adapter = None

    def set_k_levels(self, levels) -> None:
        """``levels[layer]`` is ``[n_kv, head_dim]`` with 0=2bit, 1=4, 2=8, 3=16.

        ``None`` puts every channel back at 2 bit.
        """
        if not self._attached:
            raise RuntimeError("call attach() before set_k_levels()")
        if levels is None:
            for layer in self.model.model.layers:
                layer.self_attn.k_levels = None
            return
        if len(levels) != self.n_layers:
            raise ValueError(f"expected {self.n_layers} level maps, got {len(levels)}")
        for layer, level in zip(self.model.model.layers, levels):
            layer.self_attn.k_levels = level

    def clear_capture(self) -> None:
        self.captured = {"Q": {}, "K": {}, "V": {}}
