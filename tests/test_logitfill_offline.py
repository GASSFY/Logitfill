"""CPU checks for the offline policy, plus schedule checks when torch is present.

These tests do not load a checkpoint or require a GPU. The packed CUDA parity
check is skipped unless CUDA is available.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from logitfill.policy import PrecisionPolicy, build_policy, consensus_levels, overlap_report, spearman
from logitfill.quantization.allocate import assert_history_budget, kv_history_budget, waterfill
from logitfill.quantization.distort import normalize_channel_curves
from logitfill.quantization.memory import runtime_footprint


class BudgetPolicyTest(unittest.TestCase):
    def test_kv_budget_converts_to_key_target(self):
        budget = kv_history_budget(2.5, v_bits=2, n_channels=128)
        self.assertEqual(budget["target_k_bit"], 3.0)
        self.assertEqual(budget["extra_bits_per_head"], 128)
        with self.assertRaises(ValueError):
            kv_history_budget(1.5, v_bits=2, n_channels=128)

    def test_waterfill_never_exceeds_extra_bits(self):
        rng = np.random.default_rng(0)
        curves = np.sort(rng.random((3, 32)), axis=0)[::-1]
        levels = waterfill(curves[0], curves[1], curves[2], extra=20)
        spent = int(np.array([0, 2, 6, 14], dtype=np.int64)[levels].sum())
        self.assertLessEqual(spent, 20)
        checked = assert_history_budget(levels, target_kv_bit=(2.0 + 20 / 32 + 2.0) / 2.0, v_bits=2)
        self.assertLessEqual(checked["max_extra_spent"], 20)

    def test_normalize_removes_length_scale(self):
        curves = {
            "q": np.array([[4.0]]),
            "e2": np.array([[6.0]]),
            "d2": np.array([[24.0]]),
        }
        normal = normalize_channel_curves(curves, n_query_tokens=2, n_key_tokens=3)
        self.assertAlmostEqual(float(normal["d2"]), 4.0)
        self.assertAlmostEqual(float(normal["q"] * normal["e2"]), 4.0)

    def test_policy_roundtrip_and_overlap(self):
        levels = np.zeros((2, 2, 8), dtype=np.int64)
        levels[0, 0, :2] = 1
        policy = build_policy(
            levels,
            model_id="unit-model",
            target_kv_bit=2.25,
            v_bits=2,
            calibration_datasets=["c1"],
            n_samples=3,
            # Constant gain has no rank variation, so Spearman is undefined.
            curves={
                "d2": np.arange(levels.size, dtype=np.float64).reshape(levels.shape) + 2.0,
                "d4": np.arange(levels.size, dtype=np.float64).reshape(levels.shape) * 0.25,
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            policy.save(directory)
            loaded = PrecisionPolicy.load(directory)
        self.assertTrue(np.array_equal(loaded.levels, levels))
        report = overlap_report(policy, loaded)
        self.assertEqual(report["exact_agreement"], 1.0)
        self.assertEqual(report["jaccard_int4"], 1.0)
        self.assertTrue(report["same_budget"])
        self.assertAlmostEqual(report["spearman_gain_2to4"], 1.0)
        other = build_policy(
            np.zeros_like(levels),
            model_id="unit-model",
            target_kv_bit=2.0,
            v_bits=2,
        )
        mismatch = overlap_report(policy, other)
        self.assertFalse(mismatch["same_budget"])
        self.assertLess(mismatch["exact_agreement"], 1.0)

    def test_consensus_tie_prefers_lower_bit(self):
        left = np.zeros((1, 1, 2), dtype=np.int64)
        right = np.ones((1, 1, 2), dtype=np.int64)
        policies = [
            build_policy(left, model_id="m", target_kv_bit=2.0, v_bits=2),
            build_policy(right, model_id="m", target_kv_bit=3.0, v_bits=2),
        ]
        chosen = consensus_levels(policies)
        self.assertTrue(np.array_equal(chosen, np.zeros((1, 1, 2), dtype=np.int64)))

    def test_effective_bit_counts_residual_separately(self):
        levels = np.zeros((1, 1, 4), dtype=np.int64)
        short = runtime_footprint(levels, seq_len=6, page_size=4, k_group_size=4, v_group_size=4, v_bits=2)
        self.assertEqual(short["history_tokens"], 4)
        self.assertEqual(short["residual_tokens"], 2)
        self.assertGreater(short["effective_bit"], short["history_kv_bit"])
        self.assertEqual(spearman(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])), 1.0)


class TorchScheduleTest(unittest.TestCase):
    def test_completed_page_keeps_tail(self):
        torch = self._torch()
        from logitfill.models.qwen2_attention import quantize_completed_pages
        from logitfill.quantization.quantizer import make_quantizers, make_value_quantizer

        class Attention:
            pass

        attn = Attention()
        attn.kv_page_size = 4
        attn.k_quant_len = 0
        attn.v_quant_len = 0
        attn.k_levels = None
        attn.kv_quantizers = make_quantizers(4)
        attn.v_bits = 2
        attn.v_quantizer = make_value_quantizer(2, 4)
        key = torch.randn(1, 1, 6, 4)
        value = torch.randn(1, 1, 6, 4)
        key_tail = key[..., 4:, :].clone()
        value_tail = value[..., 4:, :].clone()
        quantize_completed_pages(attn, key, value, past_len=0, step=6)
        self.assertEqual(attn.k_quant_len, 4)
        self.assertTrue(torch.equal(key[..., 4:, :], key_tail))
        self.assertTrue(torch.equal(value[..., 4:, :], value_tail))
        self.assertFalse(torch.equal(key[..., :4, :], key_tail.new_zeros(key[..., :4, :].shape)))

    def test_key_pack_matches_group_quantizer(self):
        torch = self._torch()
        from logitfill.packed.layout import dequantize_key, quantize_key_page, quantize_value_page, dequantize_value, tier_tables
        from logitfill.quantization.quantizer import GroupQuantizer, quantize_along_features, quantize_along_time

        torch.manual_seed(0)
        page = torch.randn(2, 16, 8)
        levels = torch.zeros(2, 8, dtype=torch.int64)
        built = tier_tables(levels)
        table = built["tiers"]
        table["tier"] = built["tier"]
        packed = quantize_key_page(page, table, group_size=8)
        restored = dequantize_key(16, table, packed, group_size=8)
        reference = quantize_along_time(page.unsqueeze(0), GroupQuantizer(2, 8))[0]
        self.assertTrue(torch.allclose(restored, reference.float(), atol=1e-2, rtol=1e-2))
        values = quantize_value_page(page, bits=2, group_size=8)
        restored_v = dequantize_value(values, 16, 8, bits=2, group_size=8)
        reference_v = quantize_along_features(page.unsqueeze(0), GroupQuantizer(2, 8))[0]
        self.assertTrue(torch.allclose(restored_v, reference_v.float(), atol=1e-2, rtol=1e-2))

    def test_prefill_logits_ignore_quantization_and_later_pages_flush(self):
        torch = self._torch()
        try:
            from transformers import Qwen2Config, Qwen2ForCausalLM
        except ImportError as exc:
            self.skipTest(f"transformers is not installed: {exc}")
        from logitfill.models.deepseek_qwen import DeepSeekQwenKVAdapter

        torch.manual_seed(0)
        config = Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            attention_dropout=0.0,
        )
        reference_model = Qwen2ForCausalLM(config).eval()
        quantized_model = Qwen2ForCausalLM(config).eval()
        quantized_model.load_state_dict(reference_model.state_dict())
        for model in (reference_model, quantized_model):
            model.config._attn_implementation = "eager"
        ids = torch.randint(0, 64, (1, 8))
        with torch.no_grad():
            reference = reference_model(input_ids=ids[:, :6], use_cache=True)
            adapter = DeepSeekQwenKVAdapter(quantized_model, group_size=4, v_bits=2).attach()
            first = quantized_model(input_ids=ids[:, :6], use_cache=True)
        self.assertTrue(torch.allclose(first.logits, reference.logits, atol=1e-5, rtol=1e-4))
        attention = quantized_model.model.layers[0].self_attn
        self.assertEqual(attention.k_quant_len, 4)
        with torch.no_grad():
            second = quantized_model(input_ids=ids[:, 6:7], past_key_values=first.past_key_values, use_cache=True)
        self.assertEqual(attention.k_quant_len, 4)
        with torch.no_grad():
            quantized_model(input_ids=ids[:, 7:8], past_key_values=second.past_key_values, use_cache=True)
        self.assertEqual(attention.k_quant_len, 8)
        del adapter

    def test_kernel_matches_reference_when_cuda_exists(self):
        torch = self._torch()
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available on this machine")
        from logitfill.packed.cache import PackedKVManager
        from logitfill.packed.kernel_loader import get_kernel
        from logitfill.packed.reference import packed_attention, reference_attention

        if get_kernel() is None:
            self.skipTest("the packed CUDA kernel did not compile")

        torch.manual_seed(0)
        levels = [torch.randint(0, 4, (1, 64), device="cuda")]
        manager = PackedKVManager(
            levels=levels,
            n_q=1,
            n_kv=1,
            head_dim=64,
            page_size=64,
            k_group_size=64,
            v_group_size=64,
            v_bits=2,
            device="cuda",
        )
        key = torch.randn(1, 1, 64, 64, device="cuda")
        value = torch.randn(1, 1, 64, 64, device="cuda")
        manager.commit(0, key, value)
        query = torch.randn(1, 1, 1, 64, device="cuda")
        new_k = torch.randn(1, 1, 1, 64, device="cuda")
        new_v = torch.randn(1, 1, 1, 64, device="cuda")
        fast = packed_attention(manager, 0, query, new_k, new_v, scaling=64 ** -0.5)
        slow = reference_attention(manager, 0, query, new_k, new_v, scaling=64 ** -0.5)
        self.assertTrue(torch.allclose(fast.float(), slow.float(), atol=1e-3, rtol=1e-3))

    def _torch(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest(f"torch is not installed: {exc}")
        return torch


if __name__ == "__main__":
    unittest.main()
