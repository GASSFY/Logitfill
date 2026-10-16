"""Fit one static Key policy per calibration set and K+V budget.

Examples:
  python scripts/calibrate_policy.py --model /path/to/model \
      --data c1=/path/to/longcot_a.jsonl --data c2=/path/to/longcot_b.jsonl \
      --target-kv-bit 2.5 3.0 --out artifacts/policies
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from logitfill.calibration import (
    aggregate_curves,
    capture_curves,
    encode_pair,
    load_records,
    policies_from_curves,
)
from logitfill.models.deepseek_qwen import DeepSeekQwenKVAdapter
from logitfill.quantization.quantizer import make_quantizers


def _named_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        path = Path(value)
        return path.stem, value
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("--data expects NAME=PATH or PATH")
    return name, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", action="append", required=True, type=_named_path)
    parser.add_argument("--target-kv-bit", type=float, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--v-bits", type=int, default=2)
    parser.add_argument("--k-group-size", type=int, default=128)
    parser.add_argument("--v-group-size", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=128)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    device = next(model.parameters()).device
    adapter = DeepSeekQwenKVAdapter(
        model,
        group_size=args.k_group_size,
        v_group_size=args.v_group_size,
        page_size=args.page_size,
        v_bits=args.v_bits,
    ).attach()
    quantizers = make_quantizers(args.k_group_size)
    n_groups = model.config.num_attention_heads // model.config.num_key_value_heads
    args.out.mkdir(parents=True, exist_ok=True)

    for name, path in args.data:
        records = load_records(path)
        if args.max_samples is not None:
            records = records[: args.max_samples]
        samples = []
        kept = []
        for record in records:
            input_ids, prompt_len = encode_pair(tokenizer, record["prompt"], record["response"], args.chat_template)
            input_ids = input_ids.to(device)
            if args.max_length and int(input_ids.shape[-1]) > args.max_length:
                input_ids = input_ids[:, : args.max_length]
            trim = int(input_ids.shape[-1]) % args.k_group_size
            if trim:
                input_ids = input_ids[:, : int(input_ids.shape[-1]) - trim]
            if prompt_len <= 0 or prompt_len >= int(input_ids.shape[-1]):
                print(f"skip {name} id={record['id']} after trimming", flush=True)
                continue
            curves = capture_curves(
                model, adapter, input_ids, prompt_len, quantizers, n_groups, query_span="response"
            )
            samples.append(curves)
            kept.append({"id": record["id"], "n_query": curves["n_query"], "n_key": curves["n_key"]})
            print(f"{name} id={record['id']} query={curves['n_query']} key={curves['n_key']}", flush=True)
        if not samples:
            raise RuntimeError(f"{name} produced no usable calibration samples")
        means = aggregate_curves(samples)
        policies = policies_from_curves(
            means["normalized"],
            args.target_kv_bit,
            model_id=args.model,
            v_bits=args.v_bits,
            k_group_size=args.k_group_size,
            v_group_size=args.v_group_size,
            page_size=args.page_size,
            calibration_datasets=[name],
            n_samples=len(samples),
        )
        dataset_dir = args.out / name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "dataset": name,
            "n_samples": len(samples),
            "samples": kept,
            "normalized_mean": {key: float(value.mean()) for key, value in means["normalized"].items()},
            "raw_mean": {key: float(value.mean()) for key, value in means["raw"].items()},
        }
        (dataset_dir / "curves_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        for policy in policies:
            target = f"{policy.metadata['target_kv_bit']:.4f}"
            policy.save(dataset_dir / f"kv{target}")
            print(
                f"saved {name} target_kv={target} achieved_kv={policy.metadata['achieved']['kv_bit']:.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
