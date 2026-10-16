# LogitFill

LogitFill 给 Key cache 做静态混合精度，给 Value cache 做统一的 per-token INT2。每个模型离线校准一次，保存一张 `[layers, kv_heads, head_dim]` 档位表。在线推理只读取这张表，不再计算曲线，也不再运行注水。

历史区的目标比特 `target_kv_bit` 是 Key 和 Value 码字的平均值。Value 固定 2 bit，因此

```text
target_k_bit = 2 * target_kv_bit - 2
```

Key 从全员 INT2 起步，按注意力 logit 的二阶对角失真把剩余比特逐档买到 4、8 或 16 bit。BF16 residual page、scale/min 和 packing padding 不占用这个预算；评测时另报 effective bit 和 bytes/token。

在线顺序是：

1. Prefill 用原始精度做 attention。
2. Attention 结束后，按固定档位压缩已经凑满的 128-token 页。
3. Decode 的当前页留在 BF16，参与本次 attention。
4. 当前页满 128 后，再压缩进历史区。

`fake` 后端把反量化结果写回浮点 cache，只用于核对数值和任务准确率，不减少显存，也不能作为吞吐结果。`packed` 后端按档位打包历史 K/V，decode 优先走 fused kernel，编译失败时退回 dense reference。

## 目录

| 路径 | 内容 |
| --- | --- |
| `src/logitfill/allocators/policy.py` | 模型级档位表的保存、加载、重叠率和投票 |
| `src/logitfill/calibration/` | Long-CoT 样本读取和等权曲线聚合 |
| `src/logitfill/allocators/allocate.py` | K+V 预算换算和反向注水 |
| `src/logitfill/allocators/memory.py` | 历史 payload、residual 和 effective bit |
| `src/logitfill/models/qwen2_attention.py` | BF16 attention 之后再压缩完整页 |
| `src/logitfill/packed/` | 打包 cache、reference attention 和 CUDA decode kernel |
| `scripts/calibrate_policy.py` | 用一份或多份校准数据生成 policy |
| `scripts/compare_policies.py` | 比较 policy 重叠，并可在留出数据上测 ΔNLL |
| `scripts/evaluate.py` | pass@1、majority voting 和 footprint |
| `exp/ablation/run_policy_gap.py` | offline、prompt-only、oracle 三种档位的差距 |

## 校准和评测

校准文件是 JSONL 或 Parquet，至少包含 `prompt` 和 `response`。`response` 必须是已经写好的 Long-CoT，脚本不会在校准集上重新采样答案。

```bash
python scripts/calibrate_policy.py \
    --model /path/to/DeepSeek-R1-Distill-Qwen-7B \
    --data c1=/path/to/longcot_a.jsonl \
    --data c2=/path/to/longcot_b.jsonl \
    --target-kv-bit 2.5 3.0 \
    --out artifacts/policies

python scripts/compare_policies.py \
    --policy c1=artifacts/policies/c1/kv2.5000 \
    --policy c2=artifacts/policies/c2/kv2.5000 \
    --model /path/to/model \
    --eval-data /path/to/heldout.jsonl

python scripts/evaluate.py \
    --model /path/to/model \
    --policy artifacts/policies/c1/kv2.5000 \
    --data /path/to/aime.parquet \
    --backend fake \
    --n-responses 16 \
    --chat-template \
    --out artifacts/aime_fake.json
```

同一模型、同一 `target_kv_bit` 下，两张表的 `exact_agreement` 高，说明档位主要反映模型结构。预算或形状不同的表不能这样解释。

`exp/ablation/run_ablation.py` 仍是 Key 分配机制实验，Value 保持 BF16。部署路径的 Value 默认是 INT2。
