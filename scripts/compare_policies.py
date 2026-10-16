"""Compare static policies from different Long-CoT calibration sets.

Exact agreement near one, at the same model and the same K+V budget, means
the level map is a property of the model rather than of a few samples.
Optional ``--eval-data`` reports held-out ΔNLL and, with ``--pass-at-1``,
AIME-style voting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from logitfill.allocators.allocate import assert_history_budget
from logitfill.allocators.policy import PrecisionPolicy, consensus_levels, overlap_report


def _named_dir(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--policy expects NAME=DIR")
    name, path = value.split("=", 1)
    return name, path


def _consensus(policies: list[PrecisionPolicy]) -> PrecisionPolicy:
    levels = consensus_levels(policies)
    base = dict(policies[0].metadata)
    budgets = {policy.metadata.get("target_bit") for policy in policies}
    bit_v = {int(policy.metadata.get("bit_v", 2)) for policy in policies}
    feasible = False
    if len(budgets) == 1 and len(bit_v) == 1 and None not in budgets:
        try:
            assert_history_budget(levels, float(next(iter(budgets))), int(next(iter(bit_v))))
            feasible = True
        except ValueError:
            feasible = False
    base.update(
        {
            "calibration_datasets": [policy.metadata.get("calibration_datasets", []) for policy in policies],
            "n_samples": sum(int(policy.metadata.get("n_samples", 0)) for policy in policies),
            "consensus": True,
            "feasible_for_budget": feasible,
            "enforce_budget": feasible,
        }
    )
    return PrecisionPolicy(levels, base)


def _evaluate(policies: dict[str, PrecisionPolicy], args) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from logitfill.benchmarks.aime import answers_equal, extract_answer, summarize_generations
    from logitfill.benchmarks.generate import generate_text
    from logitfill.benchmarks.scoring import token_nll
    from logitfill.calibration import capture_curves, encode_pair, load_records
    from logitfill.models.deepseek_qwen import DeepSeekQwenKVAdapter
    from logitfill.quantizers.quantizer import make_quantizers

    first = next(iter(policies.values()))
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
        group_size=int(first.metadata.get("k_group_size", 128)),
        v_group_size=int(first.metadata.get("v_group_size", 128)),
        page_size=int(first.metadata.get("page_size", 128)),
        bit_v=int(first.metadata.get("bit_v", 2)),
    ).attach()
    records = load_records(args.eval_data)
    if args.max_samples is not None:
        records = records[: args.max_samples]
    quantizers = make_quantizers(adapter.group_size)
    n_groups = model.config.num_attention_heads // model.config.num_key_value_heads
    report = {}
    for name, policy in policies.items():
        policy.validate_model(model)
        adapter.set_k_levels(policy.levels)
        deltas = []
        generations = []
        for index, record in enumerate(records):
            input_ids, prompt_len = encode_pair(tokenizer, record["prompt"], record["response"], args.chat_template)
            input_ids = input_ids.to(device)
            trim = int(input_ids.shape[-1]) % adapter.group_size
            if trim:
                input_ids = input_ids[:, : int(input_ids.shape[-1]) - trim]
            if prompt_len <= 0 or prompt_len >= int(input_ids.shape[-1]):
                continue
            reference = token_nll(model, input_ids, prompt_len, quantized=False, page_size=adapter.page_size)
            adapter.skip_quant = False
            scored = token_nll(model, input_ids, prompt_len, quantized=True, page_size=adapter.page_size)
            deltas.append(scored - reference)
            if args.pass_at_1:
                torch.manual_seed(args.seed + index)
                text, _ = generate_text(
                    model,
                    tokenizer,
                    input_ids[:, :prompt_len],
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                generations.append(
                    {
                        "id": record["id"] if record["id"] is not None else str(index),
                        "gold": record["gold"] if record.get("gold") else extract_answer(record["response"]),
                        "prediction": extract_answer(text),
                    }
                )
        entry = {"delta_nll": None if not deltas else float(sum(deltas) / len(deltas)), "n": len(deltas)}
        if generations:
            entry["generation"] = summarize_generations(generations)
            entry["exact_matches"] = [answers_equal(row["prediction"], row["gold"]) for row in generations]
        report[name] = entry
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", action="append", required=True, type=_named_dir)
    parser.add_argument("--out", type=Path, default=ROOT / "exp" / "policy_stability" / "results")
    parser.add_argument("--model", default=None)
    parser.add_argument("--eval-data", default=None)
    parser.add_argument("--pass-at-1", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chat-template", action="store_true")
    args = parser.parse_args()

    loaded = {name: PrecisionPolicy.load(path) for name, path in args.policy}
    names = list(loaded)
    pairwise = {}
    for left_index, left_name in enumerate(names):
        for right_name in names[left_index + 1 :]:
            pairwise[f"{left_name}__{right_name}"] = overlap_report(loaded[left_name], loaded[right_name])
    consensus = _consensus(list(loaded.values()))
    args.out.mkdir(parents=True, exist_ok=True)
    consensus.save(args.out / "consensus")
    payload = {
        "pairwise": pairwise,
        "consensus_feasible": bool(consensus.metadata.get("feasible_for_budget")),
        "note": (
            "High exact agreement is evidence of model structure only when "
            "same_budget and same_shape are true."
        ),
    }
    if args.eval_data:
        if not args.model:
            raise SystemExit("--eval-data requires --model")
        loaded["consensus"] = consensus
        payload["heldout"] = _evaluate(loaded, args)
    (args.out / "overlap.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["pairwise"], indent=2), flush=True)


if __name__ == "__main__":
    main()
