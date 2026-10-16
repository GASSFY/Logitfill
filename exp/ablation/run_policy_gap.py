"""Compare offline, prompt-only, and oracle Key policies on held-out traces.

The offline policy is fit on a separate Long-CoT calibration set. Prompt-only
uses the evaluated prompt's own Q/K. Oracle uses the evaluated response's
query energy, so it is an upper bound rather than a deployable method.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from logitfill.calibration import (
    aggregate_curves,
    capture_curves,
    encode_pair,
    load_records,
    policies_from_curves,
)
from logitfill.allocators.allocate import allocate_levels, kv_history_budget
from logitfill.benchmarks.scoring import token_nll
from logitfill.models.deepseek_qwen import DeepSeekQwenKVAdapter
from logitfill.quantizers.quantizer import make_quantizers


def _levels_from_capture(curves, target_bit: float, bit_v: int) -> "object":
    budget = kv_history_budget(target_bit, bit_v, int(curves["d2"].shape[-1]))
    return allocate_levels(curves["d2"], curves["d4"], curves["d8"], budget["extra_budget_per_head"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--calib", required=True, help="Long-CoT JSONL or Parquet used only to fit the offline policy")
    parser.add_argument("--eval", required=True, dest="eval_data")
    parser.add_argument("--target-bit", type=float, required=True)
    parser.add_argument("--max-calib", type=int, default=None)
    parser.add_argument("--max-eval", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--bit-v", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--out", type=Path, default=ROOT / "exp" / "ablation" / "results" / "policy_gap.json")
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
    adapter = DeepSeekQwenKVAdapter(model, group_size=args.group_size, bit_v=args.bit_v).attach()
    quantizers = make_quantizers(args.group_size)
    n_groups = model.config.num_attention_heads // model.config.num_key_value_heads

    def prepare(record):
        input_ids, prompt_len = encode_pair(tokenizer, record["prompt"], record["response"], args.chat_template)
        input_ids = input_ids.to(device)
        if int(input_ids.shape[-1]) > args.max_length:
            input_ids = input_ids[:, : args.max_length]
        trim = int(input_ids.shape[-1]) % args.group_size
        if trim:
            input_ids = input_ids[:, : int(input_ids.shape[-1]) - trim]
        if prompt_len <= 0 or prompt_len >= int(input_ids.shape[-1]):
            return None
        return input_ids, prompt_len

    calib_records = load_records(args.calib)
    if args.max_calib is not None:
        calib_records = calib_records[: args.max_calib]
    calib_curves = []
    for record in calib_records:
        prepared = prepare(record)
        if prepared is None:
            continue
        input_ids, prompt_len = prepared
        calib_curves.append(
            capture_curves(model, adapter, input_ids, prompt_len, quantizers, n_groups, "response")
        )
    if not calib_curves:
        raise RuntimeError("calibration set produced no usable traces")
    offline = policies_from_curves(
        aggregate_curves(calib_curves)["normalized"],
        [args.target_bit],
        model_id=args.model,
        bit_v=args.bit_v,
        k_group_size=args.group_size,
        v_group_size=args.group_size,
        page_size=args.group_size,
        calibration_datasets=[str(args.calib)],
        n_samples=len(calib_curves),
    )[0]

    scores = {"offline": [], "prompt_only": [], "oracle": []}
    eval_records = load_records(args.eval_data)
    if args.max_eval is not None:
        eval_records = eval_records[: args.max_eval]
    for record in eval_records:
        prepared = prepare(record)
        if prepared is None:
            continue
        input_ids, prompt_len = prepared
        reference = token_nll(model, input_ids, prompt_len, quantized=False, page_size=adapter.page_size)
        prompt_tokens = (prompt_len // args.group_size) * args.group_size
        if prompt_tokens <= 0:
            continue
        prompt_ids = input_ids[:, :prompt_tokens]
        variants = {
            "offline": offline.levels,
            "prompt_only": _levels_from_capture(
                capture_curves(
                    model, adapter, prompt_ids, prompt_tokens, quantizers, n_groups, "prompt"
                )["normalized"],
                args.target_bit,
                args.bit_v,
            ),
            "oracle": _levels_from_capture(
                capture_curves(model, adapter, input_ids, prompt_len, quantizers, n_groups, "response")["normalized"],
                args.target_bit,
                args.bit_v,
            ),
        }
        for name, levels in variants.items():
            adapter.set_k_levels(levels)
            adapter.skip_quant = False
            scored = token_nll(model, input_ids, prompt_len, quantized=True, page_size=adapter.page_size)
            scores[name].append(scored - reference)
            print(f"id={record['id']} {name}={scores[name][-1]:.5f}", flush=True)
    summary = {name: None if not values else float(sum(values) / len(values)) for name, values in scores.items()}
    payload = {"target_bit": args.target_bit, "delta_nll": summary, "samples": scores}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
