# 有效性实验

这两组只检查 LogitFill 自己的两个部分，不和 KIVI、MiKV、RotateKV、PM-KVQ 比。那些系统对照以后放在 `exp/baselines/`。

模型是 DeepSeek-R1-Distill-Qwen-7B。数据是 AIME 2024 和 AIME 2026 各 4 题。指标是冻结轨迹上的 teacher-forcing ΔNLL，越低越好。

## 失真

买家固定为 `waterfill`。只换曲线：

- `logit`：\(D_c(b)=q_c e_c(b)\)
- `no_q`：只用量化误差 \(e_c(b)\)
- `random`：随机单调曲线

预算是 2.25 bit 和 2.48 bit。打分和评测在同一条轨迹上。

## 分配

失真固定为 \(q_c e_c\)。三种花法使用同一笔额外比特：

- `uniform`：按 channel 轮流升级
- `fixed`：按 MixKVQ 的名次，先留 FP16 再留 INT4。2.48 bit 是 3 条 FP16 加 10 条 INT4，3.36 bit 是 7 条 FP16 加 38 条 INT4
- `logit`：逐档注水

档位用一折轨迹的平均失真算好，在另一折上测 ΔNLL。

```bash
cd /root/autodl-tmp/quantization/PRISM
python exp/ablation/run_ablation.py
```

短检查：

```bash
python exp/ablation/run_ablation.py --smoke
```

结果写到 `exp/ablation/results/ablation.json`。冒烟写到 `exp/ablation/smoke/`。
