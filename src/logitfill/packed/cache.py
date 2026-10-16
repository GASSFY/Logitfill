"""Packed history cache plus the original-dtype residual page.

History pages are appended only after attention. The fixed channel map is
stored once per layer. This cache is the stage-2 representation; the fake
backend still uses the Hugging Face cache for algorithm checks.
"""
from __future__ import annotations

from typing import Optional

import torch

from logitfill.packed.layout import dequantize_key, dequantize_value, quantize_key_page, quantize_value_page, tier_tables


class _LayerStore:
    def __init__(self, levels: torch.Tensor, page_size: int, k_group_size: int, v_group_size: int, bit_v: int):
        self.page_size = page_size
        self.k_group_size = k_group_size
        self.v_group_size = v_group_size
        self.bit_v = bit_v
        built = tier_tables(levels)
        self.table = built["tiers"]
        self.table["tier"] = built["tier"]
        self.hist_k: dict[int, dict[str, torch.Tensor]] | None = None
        self.hist_v: dict[str, torch.Tensor] | None = None
        self.hist_len = 0
        self.tail_k: Optional[torch.Tensor] = None
        self.tail_v: Optional[torch.Tensor] = None

    @property
    def tail_len(self) -> int:
        if self.tail_k is None:
            return 0
        return int(self.tail_k.shape[1])

    @property
    def length(self) -> int:
        return self.hist_len + self.tail_len

    def reset(self) -> None:
        self.hist_k = None
        self.hist_v = None
        self.hist_len = 0
        self.tail_k = None
        self.tail_v = None

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        """Append new tokens ``[heads, tokens, dim]`` and flush full pages."""
        if key.shape[0] != value.shape[0] or key.shape[1] != value.shape[1]:
            raise ValueError("Key and Value pages do not match")
        self.tail_k = key if self.tail_k is None else torch.cat((self.tail_k, key), dim=1)
        self.tail_v = value if self.tail_v is None else torch.cat((self.tail_v, value), dim=1)
        while self.tail_len >= self.page_size:
            page_k = self.tail_k[:, : self.page_size]
            page_v = self.tail_v[:, : self.page_size]
            self.tail_k = self.tail_k[:, self.page_size :]
            self.tail_v = self.tail_v[:, self.page_size :]
            if self.tail_k.shape[1] == 0:
                self.tail_k = None
                self.tail_v = None
            self._flush(page_k, page_v)

    def _flush(self, key: torch.Tensor, value: torch.Tensor) -> None:
        packed_k = quantize_key_page(key, self.table, self.k_group_size)
        packed_v = quantize_value_page(value, self.bit_v, self.v_group_size)
        if self.hist_k is None:
            self.hist_k = packed_k
            self.hist_v = packed_v
        else:
            for code in (0, 1, 2):
                self.hist_k[code]["code"] = torch.cat((self.hist_k[code]["code"], packed_k[code]["code"]), dim=1)
                self.hist_k[code]["scale"] = torch.cat((self.hist_k[code]["scale"], packed_k[code]["scale"]), dim=2)
                self.hist_k[code]["min"] = torch.cat((self.hist_k[code]["min"], packed_k[code]["min"]), dim=2)
            self.hist_k[3]["value"] = torch.cat((self.hist_k[3]["value"], packed_k[3]["value"]), dim=1)
            if self.bit_v >= 16:
                self.hist_v["value"] = torch.cat((self.hist_v["value"], packed_v["value"]), dim=1)
            else:
                for name in ("code", "scale", "min"):
                    self.hist_v[name] = torch.cat((self.hist_v[name], packed_v[name]), dim=1)
        self.hist_len += int(key.shape[1])

    def dequantize(self) -> tuple[torch.Tensor, torch.Tensor]:
        dim = int(self.table["tier"].shape[1])
        if self.hist_k is None:
            empty = torch.zeros(self.table["tier"].shape[0], 0, dim, dtype=torch.float32, device=self.table["tier"].device)
            return empty, empty
        key = dequantize_key(self.hist_len, self.table, self.hist_k, self.k_group_size)
        value = dequantize_value(self.hist_v, self.hist_len, dim, self.bit_v, self.v_group_size)
        return key, value


class PackedKVManager:
    """Per-layer packed KV for one sequence."""

    def __init__(
        self,
        levels,
        n_q: int,
        n_kv: int,
        head_dim: int,
        page_size: int,
        k_group_size: int,
        v_group_size: int,
        bit_v: int,
        device,
    ):
        self.n_q = int(n_q)
        self.n_kv = int(n_kv)
        self.head_dim = int(head_dim)
        self.page_size = int(page_size)
        self.k_group_size = int(k_group_size)
        self.v_group_size = int(v_group_size)
        self.bit_v = int(bit_v)
        self.device = device
        self.seq_len = 0
        self.layers = []
        for level in levels:
            if level is None:
                values = torch.zeros(self.n_kv, head_dim, dtype=torch.int64, device=device)
            else:
                values = torch.as_tensor(level, dtype=torch.int64, device=device)
            self.layers.append(
                _LayerStore(values, page_size, k_group_size, v_group_size, bit_v)
            )

    def reset(self) -> None:
        self.seq_len = 0
        for layer in self.layers:
            layer.reset()

    def layer_length(self, layer_idx: int) -> int:
        return self.layers[layer_idx].length

    def commit(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> None:
        if key.shape[0] != 1:
            raise NotImplementedError("packed KV supports one sequence at a time")
        self.layers[layer_idx].append(key[0].detach(), value[0].detach())
        if layer_idx == len(self.layers) - 1:
            self.seq_len = self.layers[layer_idx].length


try:
    from transformers.cache_utils import Cache as _TransformersCache
except ImportError:  # pragma: no cover - exercised only when transformers is absent
    _TransformersCache = object


class LogitFillCache(_TransformersCache):
    """Small cache handle so ``generate`` can read the packed sequence length."""

    def __init__(self, manager: PackedKVManager):
        try:
            super().__init__()
        except TypeError:
            pass
        self.manager = manager
        self.key_cache = []
        self.value_cache = []

    def get_seq_length(self, layer_idx: int = 0) -> int:
        del layer_idx
        return int(self.manager.seq_len)

    def get_max_cache_shape(self):
        return None

    def get_mask_sizes(self, cache_position, layer_idx: int = 0):
        del layer_idx
        query_length = int(cache_position.shape[0])
        return self.manager.seq_len + query_length, 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        raise RuntimeError("packed attention writes the LogitFill manager directly")

    def reset(self) -> None:
        self.manager.reset()

    def note_layer(self, layer_idx: int, seq_len: int) -> None:
        del layer_idx, seq_len

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        raise NotImplementedError("packed LogitFill cache does not support beam search")
