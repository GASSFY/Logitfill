"""KV lifecycle.

Page flush timing still lives in ``logitfill.models.qwen2_attention``.
Packed history still lives in ``logitfill.packed``.
Those pieces stay where they are until a later change splits them.
"""
