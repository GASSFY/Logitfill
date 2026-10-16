"""Score LogitFill and the HF baselines on the same frozen tokens."""
from __future__ import annotations

import gc
import json
import traceback
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from logitfill.baselines.environment import config_hash, git_commit, runtime_environment
from logitfill.baselines.methods import attach_kivi, attach_kvtuner, attach_logitfill, attach_pm_kvq
from logitfill.baselines.scoring import stepwise_nll

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}


def _load_model(model_id: str, dtype: torch.dtype, attn_impl: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl,
    )
    model.eval()
    return model


def _release(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _traces(config: dict, tokenizer) -> list[dict]:
    dataset = load_dataset(config["dataset"], split="train")
    length = int(config["length"])
    prompt_len = int(config["prompt_len"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    traces = []
    for row in list(dataset)[: int(config["n"])]:
        text = str(row.get("problem") or row.get("question")) + "\n" + str(row.get("solution") or row.get("answer") or "")
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids
        if int(ids.shape[-1]) < length:
            ids = ids.repeat(1, length // int(ids.shape[-1]) + 1)
        ids = ids[:, :length].contiguous().to(device)
        traces.append({"input_ids": ids, "prompt_len": prompt_len})
    return traces


def _score_method(name: str, config: dict, traces: list[dict], dtype, attn_impl: str) -> dict:
    model = _load_model(config["model"], dtype, attn_impl)
    past = None
    try:
        if name == "reference":
            pass
        elif name == "logitfill":
            attach_logitfill(model, int(config["bit_v"]), int(config["page_size"]))
        elif name == "kivi":
            attach_kivi(model, int(config["bit_k"]), int(config["bit_v"]), int(config["group_size"]))
        elif name == "kvtuner":
            past = attach_kvtuner(
                model,
                int(config["bit_k"]),
                int(config["bit_v"]),
                int(config["group_size"]),
                int(config["residual_length"]),
            )
        elif name == "pm_kvq":
            prompt = traces[0]["input_ids"][0, : traces[0]["prompt_len"]].tolist()
            attach_pm_kvq(model, prompt, float(config["pm_kvq_budget_mb"]), config)
        else:
            raise KeyError(name)
        values = [stepwise_nll(model, trace["input_ids"], trace["prompt_len"], past=past) for trace in traces]
    finally:
        _release(model)
    return {"nll": float(sum(values) / len(values)), "per_trace": values}


def run_compare(config_path: Path, out_path: Path | None = None) -> dict:
    text = Path(config_path).read_text(encoding="utf-8")
    config = yaml.safe_load(text)
    dtype = _DTYPES[config.get("dtype", "bfloat16")]
    attn_impl = config.get("attn_impl", "sdpa")
    repo = Path(__file__).resolve().parents[3]
    tokenizer = AutoTokenizer.from_pretrained(config["model"], trust_remote_code=True)
    traces = _traces(config, tokenizer)
    reference = _score_method("reference", config, traces, dtype, attn_impl)
    methods = []
    for name in config["methods"]:
        row = {"method": name, "status": "ok"}
        try:
            scored = _score_method(name, config, traces, dtype, attn_impl)
            row["nll"] = scored["nll"]
            row["delta_nll"] = scored["nll"] - reference["nll"]
            row["per_trace"] = scored["per_trace"]
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["traceback"] = traceback.format_exc(limit=8)
        methods.append(row)
        print(json.dumps({key: row[key] for key in row if key != "traceback"}), flush=True)
    payload = {
        "git_commit": git_commit(repo),
        "config_hash": config_hash(text),
        "seed": int(config.get("seed", 0)),
        "environment": runtime_environment(attn_impl, config.get("dtype", "bfloat16")),
        "reference_nll": reference["nll"],
        "target_bit": config.get("target_bit"),
        "bit_k": config.get("bit_k"),
        "bit_v": config.get("bit_v"),
        "methods": methods,
        "notes": {
            "kivi": "Qwen2 KIVI forward from the PM-KVQ checkout. The upstream KIVI repo has no Qwen2 model.",
            "logitfill": "No offline policy is loaded, so every Key channel stays at the int2 floor.",
            "mixkvq": "Not in this process. Importing it requires transformers>=4.51 and flash-attn, which the main env does not have.",
        },
    }
    destination = Path(out_path) if out_path else repo / "exp" / "baseline_compare" / "results" / "baseline.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {destination}", flush=True)
    return payload
