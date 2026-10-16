"""KV adapter for DeepSeek-R1-Distill-Qwen-7B.

That checkpoint is a Qwen2 model. ``attach`` installs the Qwen2 forward.
``set_k_levels`` installs one model-level Key map; online steps do not
recompute it. Value quantization is a uniform per-token width.

``backend="fake"`` stores dequantized values in the Hugging Face cache and is
the correctness path. ``backend="packed"`` stores history in the packed cache.
Neither backend quantizes a page before the attention that produced it.
"""
from __future__ import annotations

from typing import Optional

from torch import nn

from logitfill.models.apply import apply_kv
from logitfill.allocators.policy import PrecisionPolicy


class DeepSeekQwenKVAdapter:
    """Attach per-channel Key quantization and uniform Value quantization."""

    def __init__(
        self,
        model: nn.Module,
        group_size: int = 128,
        page_size: int | None = None,
        v_group_size: int | None = None,
        v_bits: int = 2,
        backend: str = "fake",
    ):
        if model.__class__.__name__ != "Qwen2ForCausalLM":
            raise TypeError(
                "DeepSeek-R1-Distill-Qwen-7B is Qwen2ForCausalLM, "
                f"got {model.__class__.__name__}"
            )
        if group_size < 1:
            raise ValueError("group_size must be positive")
        if backend not in ("fake", "packed"):
            raise ValueError(f"backend must be fake or packed, got {backend}")
        self.model = model
        self.group_size = group_size
        self.page_size = group_size if page_size is None else page_size
        self.v_group_size = group_size if v_group_size is None else v_group_size
        self.v_bits = int(v_bits)
        self.backend = backend
        self.n_layers = int(model.config.num_hidden_layers)
        self.n_kv = int(model.config.num_key_value_heads)
        self.n_q = int(model.config.num_attention_heads)
        self.head_dim = int(getattr(model.config, "head_dim", None) or model.config.hidden_size / self.n_q)
        self.capture = False
        self.skip_quant = False
        self.captured: dict = {"Q": {}, "K": {}, "V": {}}
        self.k_overrides: Optional[dict] = None
        self.policy: PrecisionPolicy | None = None
        self._originals: list = []
        self._attached = False
        self._packed = None

    def attach(self) -> "DeepSeekQwenKVAdapter":
        if self._attached:
            return self
        self._originals = [layer.self_attn.forward for layer in self.model.model.layers]
        self.model._kv_adapter = self
        apply_kv(
            self.model,
            group_size=self.group_size,
            v_group_size=self.v_group_size,
            page_size=self.page_size,
            v_bits=self.v_bits,
        )
        self._attached = True
        if self.policy is not None:
            self.set_k_levels(self.policy.levels)
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

        ``None`` puts every Key channel at 2 bit. Changing the map drops any
        packed history, because that history was encoded with the old map.
        """
        if not self._attached:
            raise RuntimeError("call attach() before set_k_levels()")
        if levels is None:
            for layer in self.model.model.layers:
                layer.self_attn.k_levels = None
        else:
            if len(levels) != self.n_layers:
                raise ValueError(f"expected {self.n_layers} level maps, got {len(levels)}")
            for layer, level in zip(self.model.model.layers, levels):
                if tuple(level.shape) != (self.n_kv, self.head_dim):
                    raise ValueError(
                        f"level map {tuple(level.shape)} does not match "
                        f"{(self.n_kv, self.head_dim)}"
                    )
                layer.self_attn.k_levels = level
        self._packed = None

    def load_policy(self, directory: str) -> PrecisionPolicy:
        policy = PrecisionPolicy.load(directory)
        policy.validate_model(self.model)
        if int(policy.metadata.get("v_bits", self.v_bits)) != self.v_bits:
            raise ValueError(
                f"policy v_bits={policy.metadata.get('v_bits')} but adapter v_bits={self.v_bits}"
            )
        self.policy = policy
        if self._attached:
            self.set_k_levels(policy.levels)
        return policy

    def packed_manager(self):
        if self._packed is None:
            from logitfill.packed.cache import PackedKVManager

            levels = [layer.self_attn.k_levels for layer in self.model.model.layers]
            device = next(self.model.parameters()).device
            self._packed = PackedKVManager(
                levels=levels,
                n_q=self.n_q,
                n_kv=self.n_kv,
                head_dim=self.head_dim,
                page_size=self.page_size,
                k_group_size=self.group_size,
                v_group_size=self.v_group_size,
                v_bits=self.v_bits,
                device=device,
            )
        return self._packed

    def make_cache(self):
        """Cache object accepted by ``model.generate(..., past_key_values=)``."""
        from logitfill.packed.cache import LogitFillCache

        return LogitFillCache(self.packed_manager())

    def clear_capture(self) -> None:
        self.captured = {"Q": {}, "K": {}, "V": {}}
