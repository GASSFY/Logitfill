"""Validity checks for the two LogitFill claims.

Distortion: the same water-filling buyer, three curves.
Allocation: the same measured distortion, three ways to spend one budget.
Allocation is scored on the other fold. Distortion is scored on the trajectory
that produced the curve, matching the earlier diagnostic.

This is not a comparison against KIVI, MiKV, RotateKV, or PM-KVQ.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]

from logitfill.allocators.allocate import allocate_levels, level_histogram, waterfill
from logitfill.benchmarks.scoring import token_nll
from logitfill.calibration.distort import channel_curves
from logitfill.models.deepseek_qwen import DeepSeekQwenKVAdapter
from logitfill.quantizers.quantizer import make_quantizers
# extra bits per head. 32 -> 2.25 bit, 62 -> 2.484 bit, 174 -> 3.359 bit.
DISTORTION_EXTRAS = (32, 62)
ALLOC_EXTRAS = (62, 174)
FIXED_COUNTS = {62: (3, 10), 174: (7, 38)}


def load_questions(path: str) -> list[str]:
    ds = load_dataset("parquet", data_files=path, split="train")
    return [row.get("problem") or row.get("question") or row["prompt"] for row in ds]


def build_chat(tokenizer, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": "Solve step by step and put the final numeric answer at the end.\n" + question,
        }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def nll(model, ids: torch.Tensor, prompt_len: int, quantized: bool = False, page_size: int = 128) -> float:
    # Quantized scoring follows the runtime: BF16 prefill, then page flushes
    # after attention. Value stays BF16 here so the comparison isolates Key levels.
    return token_nll(model, ids, prompt_len, quantized=quantized, page_size=page_size)


def random_curve(n: int, rng: np.random.Generator):
    draw = rng.random((3, n))
    d8 = draw[0]
    d4 = d8 + draw[1]
    d2 = d4 + draw[2]
    return d2, d4, d8


def uniform_levels(n: int, extra: int) -> np.ndarray:
    level = np.zeros(n, dtype=np.int64)
    costs = (2, 4, 8)
    spent = 0
    while True:
        choice = None
        best = None
        for i in range(n):
            src = int(level[i])
            if src >= 3 or spent + costs[src] > extra:
                continue
            key = (src, i)
            if best is None or key < best:
                best = key
                choice = i
        if choice is None:
            break
        spent += costs[int(level[choice])]
        level[choice] += 1
    return level


def fixed_levels(score: np.ndarray, n16: int, n4: int) -> np.ndarray:
    order = np.argsort(-np.asarray(score, dtype=np.float64), kind="stable")
    level = np.zeros(score.shape[0], dtype=np.int64)
    level[order[:n16]] = 3
    level[order[n16 : n16 + n4]] = 1
    return level


def stack_curves(captured, prompt_len: int, n_groups: int, quantizers) -> dict[str, np.ndarray]:
    names = ("d2", "d4", "d8", "e2", "e4", "e8", "s")
    bags = {name: [] for name in names}
    for layer in sorted(captured["K"]):
        q = captured["Q"][layer]
        k = captured["K"][layer]
        curves = channel_curves(q, k, quantizers, prompt_len, n_groups)
        for name in ("d2", "d4", "d8", "e2", "e4", "e8"):
            bags[name].append(curves[name])
        score = []
        n_kv = k.shape[1]
        for head in range(n_kv):
            q0 = head * n_groups
            qh = q[0, q0 : q0 + n_groups].float()
            weight = qh.abs().mean(dim=(0, 1))
            span = k[0, head].float().amax(0) - k[0, head].float().amin(0)
            score.append((weight * span).cpu().numpy())
        bags["s"].append(np.stack(score))
    return {name: np.stack(rows) for name, rows in bags.items()}


def levels_from(curves, kind: str, extra: int, rng: np.random.Generator | None = None) -> np.ndarray:
    n_layers, n_kv, width = curves["d2"].shape
    levels = np.zeros((n_layers, n_kv, width), dtype=np.int64)
    for layer in range(n_layers):
        for head in range(n_kv):
            if kind == "logit":
                levels[layer, head] = waterfill(
                    curves["d2"][layer, head],
                    curves["d4"][layer, head],
                    curves["d8"][layer, head],
                    extra,
                )
            elif kind == "no_q":
                levels[layer, head] = waterfill(
                    curves["e2"][layer, head],
                    curves["e4"][layer, head],
                    curves["e8"][layer, head],
                    extra,
                )
            elif kind == "random":
                d2, d4, d8 = random_curve(width, rng)
                levels[layer, head] = waterfill(d2, d4, d8, extra)
            elif kind == "uniform":
                levels[layer, head] = uniform_levels(width, extra)
            elif kind == "fixed":
                n16, n4 = FIXED_COUNTS[extra]
                levels[layer, head] = fixed_levels(curves["s"][layer, head], n16, n4)
            else:
                raise KeyError(kind)
    return levels


def apply_and_score(model, adapter, ids, prompt_len, nll_ref, levels) -> float:
    adapter.skip_quant = False
    adapter.capture = False
    adapter.set_k_levels(list(levels))
    value = nll(model, ids, prompt_len, quantized=True, page_size=adapter.page_size) - nll_ref
    torch.cuda.empty_cache()
    return value


def mean(xs):
    vals = [x for x in xs if np.isfinite(x)]
    return None if not vals else float(np.mean(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local model directory")
    parser.add_argument("--aime-a", required=True, help="Parquet file for fold A")
    parser.add_argument("--aime-b", required=True, help="Parquet file for fold B")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n_a", type=int, default=4)
    parser.add_argument("--n_b", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--min_gen", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.smoke:
        args.n_a, args.n_b = 1, 1
        args.max_new_tokens, args.min_gen = 128, 16
    out_dir = ROOT / "exp" / "ablation" / ("smoke" if args.smoke else "results")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    questions = [("A", i, q) for i, q in enumerate(load_questions(args.aime_a)[: args.n_a])]
    questions += [("B", i, q) for i, q in enumerate(load_questions(args.aime_b)[: args.n_b])]

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
    adapter = DeepSeekQwenKVAdapter(model, group_size=128, v_bits=16).attach()
    n_groups = model.config.num_attention_heads // model.config.num_key_value_heads
    quantizers = make_quantizers(128)
    print(
        f"layers={adapter.n_layers} kv={adapter.n_kv} head_dim={adapter.head_dim}",
        flush=True,
    )

    samples = []
    t0 = time.time()
    for fold, index, question in questions:
        print(f"generate fold={fold} idx={index}", flush=True)
        prompt = build_chat(tokenizer, question)
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        prompt_len = int(prompt_ids.shape[-1])
        adapter.detach()
        with torch.inference_mode():
            gen = model.generate(
                prompt_ids,
                attention_mask=torch.ones_like(prompt_ids),
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.6,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        adapter.attach()
        seq = gen
        trim = int(seq.shape[-1]) % 128
        if trim:
            seq = seq[:, : int(seq.shape[-1]) - trim]
        gen_len = int(seq.shape[-1]) - prompt_len
        if gen_len < args.min_gen:
            print(f"  skip gen_len={gen_len}", flush=True)
            continue
        adapter.skip_quant = True
        adapter.capture = True
        adapter.clear_capture()
        nll_ref = nll(model, seq, prompt_len)
        curves = stack_curves(adapter.captured, prompt_len, n_groups, quantizers)
        adapter.clear_capture()
        print(f"  gen_len={gen_len} nll_ref={nll_ref:.4f}", flush=True)

        distortion = {}
        rng = np.random.default_rng(args.seed + index + (0 if fold == "A" else 1000))
        for extra in DISTORTION_EXTRAS:
            distortion[str(extra)] = {}
            for kind in ("logit", "no_q", "random"):
                levels = levels_from(curves, kind, extra, rng)
                distortion[str(extra)][kind] = apply_and_score(
                    model, adapter, seq, prompt_len, nll_ref, levels
                )
                print(
                    f"  distortion extra={extra} {kind}={distortion[str(extra)][kind]:.5f}",
                    flush=True,
                )
        samples.append(
            {
                "fold": fold,
                "index": index,
                "gen_len": gen_len,
                "prompt_len": prompt_len,
                "nll_ref": nll_ref,
                "input_ids": seq[0].detach().cpu().numpy().astype(np.int64),
                "curves": {key: curves[key] for key in ("d2", "d4", "d8", "s")},
                "distortion": distortion,
            }
        )
        del curves
        torch.cuda.empty_cache()

    allocation = {}
    folds = sorted({s["fold"] for s in samples})
    if len(folds) >= 2:
        for calib_fold, eval_fold in ((folds[0], folds[1]), (folds[1], folds[0])):
            calib = [s for s in samples if s["fold"] == calib_fold]
            evalu = [s for s in samples if s["fold"] == eval_fold]
            direction = f"{calib_fold}->{eval_fold}"
            print(f"allocation {direction}", flush=True)
            mean_curves = {
                key: np.mean([s["curves"][key] for s in calib], axis=0)
                for key in ("d2", "d4", "d8", "s")
            }
            allocation[direction] = {}
            for extra in ALLOC_EXTRAS:
                allocation[direction][str(extra)] = {kind: [] for kind in ("uniform", "fixed", "logit")}
                packed = {
                    kind: levels_from(mean_curves, kind, extra)
                    for kind in ("uniform", "fixed", "logit")
                }
                hist = level_histogram(packed["logit"])
                for item in evalu:
                    ids = torch.as_tensor(item["input_ids"], dtype=torch.long, device=device).view(1, -1)
                    prompt_len = int(item["prompt_len"])
                    adapter.skip_quant = True
                    adapter.capture = False
                    ref = nll(model, ids, prompt_len)
                    for kind in ("uniform", "fixed", "logit"):
                        value = apply_and_score(
                            model, adapter, ids, prompt_len, ref, packed[kind]
                        )
                        allocation[direction][str(extra)][kind].append(value)
                        print(
                            f"  {direction} extra={extra} {kind}={value:.5f}",
                            flush=True,
                        )
                allocation[direction][str(extra)]["logit_hist"] = hist
                allocation[direction][str(extra)] = {
                    kind: mean(allocation[direction][str(extra)][kind])
                    if kind != "logit_hist"
                    else hist
                    for kind in ("uniform", "fixed", "logit", "logit_hist")
                }

    distortion_summary = {}
    for extra in DISTORTION_EXTRAS:
        key = str(extra)
        distortion_summary[key] = {
            kind: mean([s["distortion"][key][kind] for s in samples if key in s["distortion"]])
            for kind in ("logit", "no_q", "random")
        }
    payload = {
        "seconds": time.time() - t0,
        "n": len(samples),
        "smoke": args.smoke,
        "distortion": distortion_summary,
        "allocation": allocation,
        "samples": [
            {
                "fold": s["fold"],
                "index": s["index"],
                "gen_len": s["gen_len"],
                "distortion": s["distortion"],
            }
            for s in samples
        ],
    }
    (out_dir / "ablation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"distortion": distortion_summary, "allocation": allocation}, indent=2), flush=True)


if __name__ == "__main__":
    main()
