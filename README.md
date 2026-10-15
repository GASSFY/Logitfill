# LogitFill

对 Key cache 做混合精度。每个 channel 先放在 2 bit，再用反向注水把剩余比特买到下一档性价比最高的 channel 上。Value 保持原精度。当前只接 DeepSeek-R1-Distill-Qwen-7B，它的结构是 Qwen2。

失真是

\[
D_c(b)=q_c e_c(b)
\]

\(e_c(b)\) 是该 channel 在比特 \(b\) 下的分组量化误差，\(q_c\) 是生成段 query 的能量。购买一次只升一档，性价比是 \((D(b)-D(b'))/(b'-b)\)。档位是 2→4、4→8、8→16，花费 2、4、8 bit。16 bit 的失真为 0。

## 目录

| 路径 | 内容 |
| --- | --- |
| `logitfill/models/deepseek_qwen.py` | `DeepSeekQwenKVAdapter`，把量化挂到模型上，并用 `set_k_levels` 设置通道档位 |
| `logitfill/models/qwen2_attention.py` | Qwen2 attention：满组的 Key 按通道量化，尾部保持原精度 |
| `logitfill/models/apply.py` | `apply_kv`，换掉每一层的 forward |
| `logitfill/quantization/quantizer.py` | 非对称、按序列分组的伪量化 |
| `logitfill/quantization/distort.py` | 由 Q、K 计算 \(D_c(b)\) |
| `logitfill/quantization/allocate.py` | `waterfill`：由 `target_bit` 得到剩余比特，并做逐档注水 |
| `scripts/selfcheck.py` | 自检 |
| `experiments/` | 已完成的 KV 实验和讨论记录 |

权重量化的旧代码已经不在这个仓库里。

## 环境

自检用过的版本写在 `requirements.txt`：Python 3.10、torch 2.5.1（CUDA 12.4）、transformers 4.49.0、numpy 2.2.6、datasets 3.3.2、accelerate 1.4.0。

```bash
conda create -n logitfill python=3.10 -y
conda activate logitfill
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install -e .
```

自检会临时对照本机的 PM-KVQ 量化器，并核对 `experiments/validation_chain` 里 Stage 9 保存的注水直方图。PM-KVQ 不在上面的安装列表里。

```bash
cd /root/autodl-tmp/quantization/PRISM
python scripts/selfcheck.py
```

通过时打印 `logitfill self-check ok`。

## 实验记录

`experiments/validation_chain/` 是 Stage 0 到 Stage 9 的脚本和结果。论文用到的数字在：

- `experiments/validation_chain/exp_battery/results/stage6_waterfill/`
- `experiments/validation_chain/exp_battery/results/stage8_budget_curve/`
- `experiments/validation_chain/exp_battery/results/stage9_alloc/`

`experiments/notes/` 是方法讨论。`experiments/archive/` 是更早的 KV 探索。这些脚本记录的是当时的运行，不作为现在的入口。
