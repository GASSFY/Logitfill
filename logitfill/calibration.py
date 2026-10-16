"""Long-CoT calibration records and equal-weight curve aggregation.

Each calibration sample is a prompt plus an already written response. The
response is teacher-forced in BF16; LogitFill does not generate the oracle
answer while fitting a policy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from logitfill.quantization.allocate import allocate_levels, kv_history_budget
from logitfill.quantization.distort import channel_curves, mean_curves, normalize_channel_curves
from logitfill.policy import PrecisionPolicy, build_policy

_CURVE_KEYS = ("q", "e2", "e4", "e8", "d2", "d4", "d8")


def load_records(path: str | Path) -> list[dict]:
    """Load JSONL, Parquet, or a directory containing those files."""
    source = Path(path)
    if source.is_dir():
        files = sorted(source.glob("*.jsonl")) + sorted(source.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no JSONL or Parquet records in {source}")
        records = []
        for item in files:
            records.extend(load_records(item))
        return records
    if source.suffix == ".jsonl":
        records = []
        for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                records.append(normalize_record(json.loads(line)))
            except (KeyError, json.JSONDecodeError) as exc:
                raise ValueError(f"{source}:{line_number} is not a calibration record") from exc
        return records
    if source.suffix == ".parquet":
        from datasets import load_dataset

        dataset = load_dataset("parquet", data_files=str(source), split="train")
        return [normalize_record(dict(row)) for row in dataset]
    raise ValueError(f"unsupported calibration file {source}")


def normalize_record(row: dict) -> dict:
    prompt = row.get("prompt") or row.get("problem") or row.get("Problem") or row.get("question")
    response = row.get("response") or row.get("output") or row.get("completion")
    if prompt is None or response is None:
        raise KeyError("record needs prompt/problem and response/output/completion")
    identifier = row.get("id", row.get("ID", row.get("problem_id")))
    gold = row.get("gold", row.get("answer", row.get("Answer")))
    return {
        "id": None if identifier is None else str(identifier),
        "prompt": str(prompt),
        "response": str(response),
        "gold": None if gold is None else str(gold),
    }


def encode_pair(tokenizer, prompt: str, response: str, use_chat_template: bool):
    """Return ``(input_ids [1, T], prompt_len)``."""
    if use_chat_template:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt").input_ids
    else:
        prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids
    response_ids = tokenizer(str(response), add_special_tokens=False, return_tensors="pt").input_ids
    return torch_cat(prompt_ids, response_ids), int(prompt_ids.shape[-1])


def torch_cat(prompt_ids, response_ids):
    import torch

    return torch.cat((prompt_ids, response_ids), dim=1)


def stack_layers(captured, quantizers, response_start: int, n_groups: int, n_query: int, n_key: int) -> dict:
    """Stack per-layer raw and length-normalized curves."""
    raw_layers = []
    normal_layers = []
    for layer in sorted(captured["K"]):
        curves = channel_curves(
            captured["Q"][layer],
            captured["K"][layer],
            quantizers,
            response_start,
            n_groups,
        )
        raw_layers.append(curves)
        normal_layers.append(normalize_channel_curves(curves, n_query, n_key))
    if not raw_layers:
        raise RuntimeError("the forward did not capture any Key states")
    return {
        "raw": {key: np.stack([layer[key] for layer in raw_layers]) for key in _CURVE_KEYS},
        "normalized": {key: np.stack([layer[key] for layer in normal_layers]) for key in _CURVE_KEYS},
    }


def capture_curves(model, adapter, input_ids, prompt_len: int, quantizers, n_groups: int, query_span: str):
    """BF16 teacher-forced Q/K curves.

    ``query_span`` is ``response`` for the offline objective or ``prompt`` for
    the prompt-only baseline. Key error always covers the keys that exist in
    ``input_ids``.
    """
    import torch

    if query_span not in ("response", "prompt"):
        raise ValueError(f"query_span must be response or prompt, got {query_span}")
    adapter.skip_quant = True
    adapter.capture = True
    adapter.clear_capture()
    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=False)
    total = int(input_ids.shape[-1])
    if query_span == "response":
        response_start = prompt_len
        n_query = total - prompt_len
    else:
        response_start = 0
        n_query = prompt_len
        captured_prompt = {
            name: {
                layer: value[:, :, :prompt_len]
                for layer, value in states.items()
            }
            for name, states in adapter.captured.items()
        }
        adapter.captured = captured_prompt
        total = prompt_len
    curves = stack_layers(adapter.captured, quantizers, response_start, n_groups, n_query, total)
    adapter.clear_capture()
    adapter.capture = False
    adapter.skip_quant = False
    curves["n_query"] = int(n_query)
    curves["n_key"] = int(total)
    return curves


def aggregate_curves(samples: list[dict]) -> dict[str, dict[str, np.ndarray]]:
    """Equal-weight means of the raw sums and of the normalized curves."""
    return {
        "raw": mean_curves([sample["raw"] for sample in samples]),
        "normalized": mean_curves([sample["normalized"] for sample in samples]),
    }


def policies_from_curves(
    curves: dict[str, np.ndarray],
    target_kv_bits: list[float],
    *,
    model_id: str,
    v_bits: int,
    k_group_size: int,
    v_group_size: int,
    page_size: int,
    calibration_datasets: list[str],
    n_samples: int,
) -> list[PrecisionPolicy]:
    """Allocate one static policy per K+V history budget from the same curves."""
    policies = []
    saved = {key: curves[key] for key in ("d2", "d4", "d8") if key in curves}
    head_dim = int(curves["d2"].shape[-1])
    for target in target_kv_bits:
        budget = kv_history_budget(target, v_bits, head_dim)
        levels = allocate_levels(curves["d2"], curves["d4"], curves["d8"], budget["extra_bits_per_head"])
        policies.append(
            build_policy(
                levels,
                model_id=model_id,
                target_kv_bit=target,
                v_bits=v_bits,
                k_group_size=k_group_size,
                v_group_size=v_group_size,
                page_size=page_size,
                calibration_datasets=calibration_datasets,
                n_samples=n_samples,
                curves=saved,
                normalized=True,
            )
        )
    return policies
