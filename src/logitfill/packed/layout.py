"""Bit packing for a static Key policy and per-token Value groups.

Key codes are packed along the channel axis of one tier so a token's codes
stay contiguous. Value int2 codes are packed along the head dimension.
Scales use the same asymmetric min-max rule as ``GroupQuantizer``.
"""
from __future__ import annotations

import torch


def asymmetric_codes(grouped: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``grouped`` is ``[..., group_size]``. Returns codes, scale, min."""
    high = grouped.amax(dim=-1, keepdim=True)
    low = grouped.amin(dim=-1, keepdim=True)
    scale = torch.clamp((high - low) / float((1 << bits) - 1), min=1e-5, max=1e5)
    codes = torch.round((grouped - low) / scale).clamp(0, (1 << bits) - 1).to(torch.int32)
    return codes, scale.squeeze(-1), low.squeeze(-1)


def pack_ints(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack the last axis into int32 words. ``codes`` is non-negative."""
    if codes.numel() == 0:
        shape = codes.shape[:-1] + (0,)
        return torch.zeros(shape, dtype=torch.int32, device=codes.device)
    per = 32 // int(bits)
    pad = (-codes.shape[-1]) % per
    if pad:
        codes = torch.nn.functional.pad(codes, (0, pad))
    words = codes.shape[-1] // per
    view = codes.reshape(*codes.shape[:-1], words, per)
    packed = torch.zeros(*codes.shape[:-1], words, dtype=torch.int32, device=codes.device)
    for slot in range(per):
        packed = packed | (view[..., slot] << (slot * bits))
    return packed


def unpack_ints(packed: torch.Tensor, count: int, bits: int) -> torch.Tensor:
    """Inverse of ``pack_ints`` for the first ``count`` channels."""
    if count == 0:
        return torch.zeros(*packed.shape[:-1], 0, dtype=torch.int32, device=packed.device)
    per = 32 // int(bits)
    slots = torch.arange(count, device=packed.device)
    words = packed[..., slots // per]
    codes = (words >> ((slots % per) * bits)) & ((1 << bits) - 1)
    return codes.to(torch.int32)


def tier_tables(levels: torch.Tensor) -> dict[int, dict[str, torch.Tensor]]:
    """Build per-tier channel tables from ``levels`` of shape ``[heads, dim]``."""
    heads, dim = levels.shape
    tables = {}
    for code in (0, 1, 2, 3):
        rows = []
        for head in range(heads):
            rows.append(torch.nonzero(levels[head] == code, as_tuple=False).flatten())
        width = max((int(row.numel()) for row in rows), default=0)
        index = torch.zeros(heads, width, dtype=torch.int32, device=levels.device)
        count = torch.zeros(heads, dtype=torch.int32, device=levels.device)
        local = torch.full((heads, dim), -1, dtype=torch.int32, device=levels.device)
        for head, row in enumerate(rows):
            count[head] = int(row.numel())
            if row.numel():
                index[head, : row.numel()] = row.to(torch.int32)
                local[head, row] = torch.arange(row.numel(), device=levels.device, dtype=torch.int32)
        tables[code] = {"index": index, "count": count, "local": local, "width": width}
    tier = levels.to(torch.int32)
    return {"tiers": tables, "tier": tier}


def quantize_key_page(page: torch.Tensor, table: dict, group_size: int) -> dict[int, dict[str, torch.Tensor]]:
    """Quantize one Key page ``[heads, tokens, dim]`` into per-tier storage."""
    heads, tokens, _dim = page.shape
    if tokens % group_size != 0:
        raise ValueError(f"Key page length {tokens} is not a multiple of {group_size}")
    groups = tokens // group_size
    out = {}
    for code, bits in ((0, 2), (1, 4), (2, 8)):
        index = table[code]["index"]
        count = table[code]["count"]
        width = int(table[code]["width"])
        words = (width + (32 // bits) - 1) // (32 // bits) if width else 0
        code_out = torch.zeros(heads, tokens, words, dtype=torch.int32, device=page.device)
        scale = torch.ones(heads, width, groups, dtype=torch.float16, device=page.device)
        minimum = torch.zeros(heads, width, groups, dtype=torch.float16, device=page.device)
        for head in range(heads):
            used = int(count[head])
            if used == 0:
                continue
            channels = index[head, :used].long()
            gathered = page[head][:, channels].transpose(0, 1).contiguous()
            grouped = gathered.reshape(used, groups, group_size).float()
            codes, sc, low = asymmetric_codes(grouped, bits)
            flat = codes.reshape(used, tokens).transpose(0, 1).contiguous()
            padded = torch.zeros(tokens, width, dtype=torch.int32, device=page.device)
            padded[:, :used] = flat
            code_out[head] = pack_ints(padded, bits)
            scale[head, :used] = sc.to(torch.float16)
            minimum[head, :used] = low.to(torch.float16)
        out[code] = {"code": code_out, "scale": scale, "min": minimum}
    index = table[3]["index"]
    count = table[3]["count"]
    width = int(table[3]["width"])
    values = torch.zeros(heads, tokens, width, dtype=torch.float16, device=page.device)
    for head in range(heads):
        used = int(count[head])
        if used == 0:
            continue
        channels = index[head, :used].long()
        values[head, :, :used] = page[head][:, channels].to(torch.float16)
    out[3] = {"value": values}
    return out


def quantize_value_page(page: torch.Tensor, bits: int, group_size: int) -> dict[str, torch.Tensor]:
    """Per-token Value quantization. ``page`` is ``[heads, tokens, dim]``."""
    heads, tokens, dim = page.shape
    if bits >= 16:
        return {"value": page.to(torch.float16)}
    if dim % group_size != 0:
        raise ValueError(f"head dim {dim} is not divisible by value group {group_size}")
    grouped = page.float().reshape(heads, tokens, dim // group_size, group_size)
    codes, scale, low = asymmetric_codes(grouped, bits)
    flat = codes.reshape(heads, tokens, dim)
    return {
        "code": pack_ints(flat, bits),
        "scale": scale.to(torch.float16),
        "min": low.to(torch.float16),
    }


def dequantize_key(
    tokens: int,
    table: dict,
    packed: dict[int, dict[str, torch.Tensor]],
    group_size: int,
) -> torch.Tensor:
    """Rebuild a dense Key history ``[heads, tokens, dim]`` in float32."""
    tier = table["tier"]
    heads, dim = tier.shape
    out = torch.zeros(heads, tokens, dim, dtype=torch.float32, device=tier.device)
    if tokens == 0:
        return out
    for code, bits in ((0, 2), (1, 4), (2, 8)):
        index = table[code]["index"]
        count = table[code]["count"]
        block = packed[code]
        for head in range(heads):
            used = int(count[head])
            if used == 0:
                continue
            codes = unpack_ints(block["code"][head, :tokens], used, bits).float()
            groups = torch.arange(tokens, device=out.device) // group_size
            channels = index[head, :used].long()
            scale = block["scale"][head, :used].float()
            minimum = block["min"][head, :used].float()
            restored = codes * scale[:, groups].transpose(0, 1) + minimum[:, groups].transpose(0, 1)
            out[head, :, channels] = restored
    index = table[3]["index"]
    count = table[3]["count"]
    for head in range(heads):
        used = int(count[head])
        if used == 0:
            continue
        channels = index[head, :used].long()
        out[head, :, channels] = packed[3]["value"][head, :tokens, :used].float()
    return out


def dequantize_value(packed: dict[str, torch.Tensor], tokens: int, dim: int, bits: int, group_size: int) -> torch.Tensor:
    if tokens == 0:
        heads = packed["code"].shape[0] if bits < 16 else packed["value"].shape[0]
        return torch.zeros(heads, 0, dim, dtype=torch.float32, device=packed["code"].device if bits < 16 else packed["value"].device)
    if bits >= 16:
        return packed["value"][:, :tokens].float()
    heads = packed["code"].shape[0]
    codes = unpack_ints(packed["code"][:, :tokens], dim, bits).float()
    groups = torch.arange(dim, device=codes.device) // group_size
    scale = packed["scale"][:, :tokens].float()
    minimum = packed["min"][:, :tokens].float()
    restored = codes * scale[:, :, groups] + minimum[:, :, groups]
    return restored.reshape(heads, tokens, dim)
