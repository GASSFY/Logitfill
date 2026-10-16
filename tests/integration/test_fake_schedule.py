"""Behavior checks for fake KV quantization on DeepSeek-R1-Distill-Qwen-7B.

One checkpoint per schedule rule, with Value at INT2:

- prefill attention reads the original K/V
- a full 128-token page leaves the residual and becomes quantized history
- the next decode attention reads that quantized history
- teacher-forced ΔNLL under this schedule is finite and not a no-op
"""
from __future__ import annotations

import unittest

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from logitfill.benchmarks.scoring import token_nll
from logitfill.models.deepseek_qwen import DeepSeekQwenKVAdapter
from logitfill.quantizers.quantizer import quantize_along_features, quantize_along_time

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
PAGE = 128
PREFILL_LEN = 200
SCORE_LEN = 256
PROMPT_LEN = 80


def _ids(tokenizer, length: int) -> torch.Tensor:
    row = load_dataset("HuggingFaceH4/aime_2024", split="train")[0]
    text = str(row["problem"]) + "\n" + str(row["solution"])
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids
    if int(ids.shape[-1]) < length:
        reps = length // int(ids.shape[-1]) + 1
        ids = ids.repeat(1, reps)
    return ids[:, :length].contiguous()


def _cache_tensor(past, layer: int, kind: str) -> torch.Tensor:
    bank = past.key_cache if kind == "k" else past.value_cache
    return bank[layer]


class _AttentionProbe:
    def __init__(self):
        self.records = []
        self._real = None

    def install(self) -> None:
        import logitfill.models.qwen2_attention as attention

        self._real = attention._run_dense_attention

        def wrapped(attn, query_states, key_states, value_states, attention_mask, kwargs):
            item = {
                "layer": int(attn.layer_idx),
                "k_quant_len": int(attn.k_quant_len),
                "v_quant_len": int(attn.v_quant_len),
            }
            if attn.layer_idx == 0:
                item["k"] = key_states.detach().clone()
                item["v"] = value_states.detach().clone()
            self.records.append(item)
            return self._real(
                attn, query_states, key_states, value_states, attention_mask, kwargs
            )

        attention._run_dense_attention = wrapped

    def remove(self) -> None:
        if self._real is None:
            return
        import logitfill.models.qwen2_attention as attention

        attention._run_dense_attention = self._real
        self._real = None


class FakeKVScheduleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required")
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        cls.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
        cls.model.eval()
        cls.device = next(cls.model.parameters()).device
        cls.adapter = DeepSeekQwenKVAdapter(
            cls.model,
            group_size=PAGE,
            page_size=PAGE,
            bit_v=2,
        ).attach()
        levels = np.zeros((cls.adapter.n_layers, cls.adapter.n_kv, cls.adapter.head_dim), dtype=np.int64)
        levels[..., 0] = 3
        cls.adapter.set_k_levels(levels)
        cls.levels = levels
        cls.probe = _AttentionProbe()
        cls.probe.install()

    @classmethod
    def tearDownClass(cls):
        if getattr(cls, "probe", None) is not None:
            cls.probe.remove()

    def setUp(self):
        self.probe.records.clear()
        self.adapter.skip_quant = False
        self.adapter.capture = False

    def _forward(self, ids, past=None):
        with torch.inference_mode():
            return self.model(input_ids=ids, past_key_values=past, use_cache=True)

    def test_prefill_reads_full_precision_and_flushes_full_pages(self):
        ids = _ids(self.tokenizer, PREFILL_LEN).to(self.device)
        self.adapter.skip_quant = True
        plain = self._forward(ids)
        self.probe.records.clear()
        self.adapter.skip_quant = False
        quantized = self._forward(ids)

        logits_err = (plain.logits.float() - quantized.logits.float()).abs().max().item()
        self.assertLess(logits_err, 1e-3, f"prefill logits changed by quantization, max abs {logits_err}")

        prefill_records = [item for item in self.probe.records if item["layer"] == 0]
        self.assertEqual(len(prefill_records), 1)
        seen = prefill_records[0]
        self.assertEqual(seen["k_quant_len"], 0)
        self.assertEqual(seen["v_quant_len"], 0)
        for layer in self.model.model.layers:
            self.assertEqual(layer.self_attn.k_quant_len, PAGE)
            self.assertEqual(layer.self_attn.v_quant_len, PAGE)

        cache_k = _cache_tensor(quantized.past_key_values, 0, "k")
        cache_v = _cache_tensor(quantized.past_key_values, 0, "v")
        self.assertEqual(int(cache_k.shape[-2]), PREFILL_LEN)
        self.assertTrue(torch.equal(cache_k[:, :, PAGE:, :], seen["k"][:, :, PAGE:, :]))
        self.assertTrue(torch.equal(cache_v[:, :, PAGE:, :], seen["v"][:, :, PAGE:, :]))

        key_gap = (cache_k[:, :, :PAGE, 1:].float() - seen["k"][:, :, :PAGE, 1:].float()).abs().max().item()
        value_gap = (cache_v[:, :, :PAGE, :].float() - seen["v"][:, :, :PAGE, :].float()).abs().max().item()
        self.assertGreater(key_gap, 1e-3)
        self.assertGreater(value_gap, 1e-3)
        self.assertTrue(torch.equal(cache_k[:, :, :PAGE, :1], seen["k"][:, :, :PAGE, :1]))

        attn = self.model.model.layers[0].self_attn
        expected_k = quantize_along_time(seen["k"][:, :, :PAGE, :], attn.kv_quantizers[2])
        expected_v = quantize_along_features(seen["v"][:, :, :PAGE, :], attn.v_quantizer)
        self.assertTrue(torch.equal(cache_k[:, :, :PAGE, 1:], expected_k[:, :, :, 1:]))
        self.assertTrue(torch.equal(cache_v[:, :, :PAGE, :], expected_v))
        self.assertEqual(int(attn.v_quantizer.n_bits), 2)

    def test_decode_reads_quantized_history(self):
        ids = _ids(self.tokenizer, PREFILL_LEN).to(self.device)
        prefill = self._forward(ids)
        history_k = _cache_tensor(prefill.past_key_values, 0, "k")[:, :, :PAGE, :].clone()
        history_v = _cache_tensor(prefill.past_key_values, 0, "v")[:, :, :PAGE, :].clone()
        original = next(item for item in self.probe.records if item["layer"] == 0)
        self.probe.records.clear()

        nxt = ids[:, -1:]
        self._forward(nxt, past=prefill.past_key_values)
        decode = next(item for item in self.probe.records if item["layer"] == 0)
        self.assertEqual(decode["k_quant_len"], PAGE)
        self.assertEqual(int(decode["k"].shape[-2]), PREFILL_LEN + 1)
        self.assertTrue(torch.equal(decode["k"][:, :, :PAGE, :], history_k))
        self.assertTrue(torch.equal(decode["v"][:, :, :PAGE, :], history_v))
        fresh_gap = (decode["k"][:, :, :PAGE, 1:].float() - original["k"][:, :, :PAGE, 1:].float()).abs().max().item()
        self.assertGreater(fresh_gap, 1e-3)
        self.assertTrue(torch.equal(decode["k"][:, :, PAGE:PREFILL_LEN, :], original["k"][:, :, PAGE:, :]))
        self.assertEqual(self.model.model.layers[0].self_attn.k_quant_len, PAGE)

    def test_value_int2_delta_nll_follows_the_same_pages(self):
        ids = _ids(self.tokenizer, SCORE_LEN).to(self.device)
        self.adapter.skip_quant = True
        reference = token_nll(self.model, ids, PROMPT_LEN, quantized=False, page_size=PAGE)
        self.adapter.skip_quant = False
        quantized = token_nll(self.model, ids, PROMPT_LEN, quantized=True, page_size=PAGE)
        delta = quantized - reference
        attn = self.model.model.layers[0].self_attn
        self.assertTrue(np.isfinite(reference))
        self.assertTrue(np.isfinite(quantized))
        self.assertGreater(abs(delta), 1e-4)
        self.assertEqual(attn.k_quant_len, PAGE)
        self.assertEqual(attn.v_quant_len, PAGE)
        self.assertEqual(int(attn.bit_v), 2)
        print(
            f"value_int2 delta_nll={delta:.6f} reference={reference:.6f} quantized={quantized:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
