"""Generate with a saved policy and report pass@1, voting, and KV bytes.

``--backend fake`` checks the schedule numerically and does not reduce memory.
``--backend packed`` is the runtime whose throughput and peak memory can be claimed.
``--backend bf16`` runs the original model as the reference.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from logitfill.allocators.memory import runtime_footprint
from logitfill.allocators.policy import PrecisionPolicy
from logitfill.benchmarks.aime import extract_answer, summarize_generations
from logitfill.benchmarks.generate import generate_text


def _problems(path: str) -> list[dict]:
    source = Path(path)
    if source.suffix == ".jsonl":
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif source.suffix == ".parquet":
        from datasets import load_dataset

        rows = [dict(row) for row in load_dataset("parquet", data_files=str(source), split="train")]
    else:
        raise ValueError(f"unsupported eval file {source}")
    problems = []
    for index, row in enumerate(rows):
        prompt = row.get("prompt") or row.get("problem") or row.get("Problem") or row.get("question")
        gold = row.get("answer") or row.get("Answer") or row.get("gold")
        if prompt is None or gold is None:
            raise KeyError("eval records need a problem and an answer")
        problems.append(
            {
                "id": str(row.get("id", row.get("ID", index))),
                "prompt": str(prompt),
                "gold": str(gold),
            }
        )
    return problems


def _prompt_text(tokenizer, problem: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return problem
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": problem + "\nPlease reason step by step, and put your final answer within \\boxed{}."}],
        tokenize=False,
        add_generation_prompt=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--policy", default=None, help="Required unless --backend bf16")
    parser.add_argument("--backend", choices=("fake", "packed", "bf16"), default="fake")
    parser.add_argument("--n-responses", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.backend != "bf16" and not args.policy:
        raise SystemExit("--policy is required for fake and packed backends")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from logitfill.models.deepseek_qwen import DeepSeekQwenKVAdapter

    policy = None if args.policy is None else PrecisionPolicy.load(args.policy)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    device = next(model.parameters()).device
    adapter = None
    if args.backend != "bf16":
        adapter = DeepSeekQwenKVAdapter(
            model,
            group_size=int(policy.metadata.get("k_group_size", 128)),
            v_group_size=int(policy.metadata.get("v_group_size", 128)),
            page_size=int(policy.metadata.get("page_size", 128)),
            v_bits=int(policy.metadata.get("v_bits", 2)),
            backend=args.backend,
        ).attach()
        adapter.load_policy(args.policy)
    problems = _problems(args.data)
    if args.limit is not None:
        problems = problems[: args.limit]
    records = []
    lengths = []
    for problem in problems:
        prompt = _prompt_text(tokenizer, problem["prompt"], args.chat_template)
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        for response_index in range(args.n_responses):
            torch.manual_seed(args.seed + response_index)
            past = None
            if adapter is not None and adapter.backend == "packed":
                past = adapter.make_cache()
                past.reset()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            text, new_tokens = generate_text(
                model,
                tokenizer,
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                past_key_values=past,
            )
            seq_len = int(prompt_ids.shape[-1]) + new_tokens
            lengths.append(seq_len)
            row = {
                "id": problem["id"],
                "response_index": response_index,
                "gold": problem["gold"],
                "prediction": extract_answer(text),
                "completion": text,
                "seq_len": seq_len,
            }
            if torch.cuda.is_available():
                row["peak_memory_bytes"] = int(torch.cuda.max_memory_allocated())
            records.append(row)
            print(f"id={problem['id']} sample={response_index} tokens={new_tokens}", flush=True)
    summary = summarize_generations(records)
    if policy is not None and lengths:
        summary["footprint"] = [
            runtime_footprint(
                policy.levels,
                length,
                page_size=int(policy.metadata.get("page_size", 128)),
                k_group_size=int(policy.metadata.get("k_group_size", 128)),
                v_group_size=int(policy.metadata.get("v_group_size", 128)),
                v_bits=int(policy.metadata.get("v_bits", 2)),
            )
            for length in lengths
        ]
        summary["backend_note"] = (
            "fake stores dequantized BF16 values and does not reduce memory; "
            "packed is the backend that can reduce memory and decode bandwidth"
            if args.backend == "fake"
            else "effective_bit includes the BF16 residual page and metadata, so it is not target_kv_bit"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "records": records}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
